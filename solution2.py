import argparse
import logging
import os
import time
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer


CONFIG = {
    # 路径
    "DATA_DIR": ".",
    "TRAIN_CSV": "train.csv",
    "TEST_CSV": "test.csv",
    "IMAGE_DIR": "Partial image dataset",
    "OUTPUT_DIR": "output",

    # 模型
    "TEXT_MODEL": "answerdotai/ModernBERT-base",
    "VISION_MODEL": "google/vit-base-patch16-224",  # 仅保留AutoImageProcessor路径，当前不喂入LGBM

    # 设备与日志
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "LOG_LEVEL": "INFO",
    "SUPPRESS_HF_WARNINGS": True,

    # CV
    "N_FOLDS": 5,
    "REPEATS": 1,
    "SEEDS": [42],

    # 批大小
    "TEXT_BATCH_SIZE": 64,
    "IMAGE_BATCH_SIZE": 16,

    # 训练轮次（每个模型 + 最终主模型）
    "LGB_EPOCHS": 1,
    "CATBOOST_EPOCHS": 1,
    "META_RIDGE_EPOCHS": 1,
    "META_CATBOOST_EPOCHS": 0,

    # 每轮采样样本数
    "LGB_SAMPLES_PER_EPOCH": None,
    "CATBOOST_SAMPLES_PER_EPOCH": None,
    "META_RIDGE_SAMPLES_PER_EPOCH": None,
    "META_CATBOOST_SAMPLES_PER_EPOCH": None,

    # 轻量文本降维维度
    "TEXT_EMBED_DIM": 64,
    "META_ALPHA": 1.0,
}


logging.basicConfig(
    level=getattr(logging, CONFIG["LOG_LEVEL"]),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if CONFIG["SUPPRESS_HF_WARNINGS"]:
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")


TEXT_COLS = [
    "exploration_log",
    "environmental_report",
    "incident_report",
    "description",
    "image_prompt",
]
ORDINAL_MAPS = {
    "storm_frequency": {"rare": 0, "seasonal": 1, "frequent": 2, "constant": 3, "extreme": 4},
    "seismic_activity": {"minimal": 0, "low": 1, "moderate": 2, "high": 3, "extreme": 4},
    "magnetic_field": {"none": 0, "weak": 1, "moderate": 2, "strong": 3, "very strong": 4},
    "radiation_level": {"low": 0, "moderate": 1, "high": 2, "extreme": 3},
}
RISK_TERMS = ["radiation", "storm", "seismic", "toxic", "hazard", "predator", "extreme", "danger"]
RESOURCE_TERMS = ["water", "mineral", "energy", "fertile", "metal", "resource", "geothermal"]


@dataclass
class FeatureBundle:
    catboost_df: pd.DataFrame
    lgb_df: pd.DataFrame
    target: Optional[np.ndarray]


def read_data(data_dir: str, train_csv: Optional[str], test_csv: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = train_csv if train_csv else os.path.join(data_dir, "train.csv")
    test_path = test_csv if test_csv else os.path.join(data_dir, "test.csv")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def resolve_image_paths(df: pd.DataFrame, image_dir: str) -> np.ndarray:
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    has = []
    for idx in df["id"]:
        found = False
        for ext in exts:
            if os.path.exists(os.path.join(image_dir, f"{idx}{ext}")):
                found = True
                break
        has.append(1 if found else 0)
    return np.asarray(has, dtype=np.int8)


def make_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "environmental_report",
        "incident_report",
        "description",
        "environment_type",
        "oxygen_percent",
        "terraforming_difficulty",
        "colonization_cost_index",
    ]
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c in df.columns:
            out[f"is_missing_{c}"] = df[c].isna().astype(np.int8)
    out["missing_count_key"] = out.sum(axis=1)
    return out


def make_ordinal_dual_view(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            raw = df[col].astype(str).str.lower().str.strip()
            out[f"{col}_ord"] = raw.map(mapping).fillna(-1).astype(np.int16)
    return out


def make_physics_priors(df: pd.DataFrame, ord_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    oxy = pd.to_numeric(df.get("oxygen_percent", np.nan), errors="coerce")
    g = pd.to_numeric(df.get("gravity_g", np.nan), errors="coerce")
    p = pd.to_numeric(df.get("atmospheric_pressure_atm", np.nan), errors="coerce")
    t_mean = pd.to_numeric(df.get("mean_temp_c", np.nan), errors="coerce")
    t_range = pd.to_numeric(df.get("temp_range_c", np.nan), errors="coerce")
    rad_ord = ord_df.get("radiation_level_ord", pd.Series(-1, index=df.index))
    mag_ord = ord_df.get("magnetic_field_ord", pd.Series(-1, index=df.index))

    out["score_oxygen_suitability"] = np.exp(-((oxy - 21.0) ** 2) / (2 * 6.0 ** 2))
    out["penalty_gravity_1g"] = np.abs(g - 1.0)
    out["score_pressure_band"] = np.exp(-((p - 1.0) ** 2) / (2 * 0.6 ** 2))
    out["score_temp_comfort"] = np.exp(-((t_mean - 15.0) ** 2) / (2 * 18.0 ** 2)) * np.exp(-(t_range / 45.0))
    out["score_magnetosphere_shield"] = (mag_ord + 1) - (rad_ord + 1)
    out["score_risk_energy"] = pd.to_numeric(df.get("energy_harvest_potential", np.nan), errors="coerce") - (rad_ord + 1)
    return out.replace([np.inf, -np.inf], np.nan)


def make_text_rule_features(df: pd.DataFrame) -> pd.DataFrame:
    merged = df[TEXT_COLS].fillna("").astype(str).agg(" | ".join, axis=1)

    def _count_terms(text: str, terms: Sequence[str]) -> int:
        t = text.lower()
        return sum(t.count(term) for term in terms)

    out = pd.DataFrame(index=df.index)
    out["text_total_len"] = merged.str.len().astype(np.int32)
    out["text_sent_cnt"] = merged.str.count(r"[.!?]+") + (merged.str.len() > 0).astype(np.int8)
    out["text_num_density"] = (merged.str.count(r"\d") / np.maximum(1, merged.str.len())).astype(np.float32)
    out["text_risk_kw"] = merged.apply(lambda x: _count_terms(x, RISK_TERMS)).astype(np.int16)
    out["text_resource_kw"] = merged.apply(lambda x: _count_terms(x, RESOURCE_TERMS)).astype(np.int16)
    out["text_missing_count"] = df[TEXT_COLS].isna().sum(axis=1).astype(np.int8)
    return out


def encode_text_embedding(df: pd.DataFrame, model_name: str, batch_size: int, device: str) -> np.ndarray:
    texts = df[TEXT_COLS].fillna("").astype(str).agg(" | ".join, axis=1).tolist()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    embs: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            last = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (last * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1)
        embs.append(pooled.cpu().numpy())
    return np.vstack(embs)


def _build_lgb_dataframe(train_np: np.ndarray, test_np: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = [f"f_{i:04d}" for i in range(train_np.shape[1])]
    X_train = pd.DataFrame(train_np, columns=cols)
    X_test = pd.DataFrame(test_np, columns=cols)
    assert X_train.columns.is_unique
    assert X_test.columns.is_unique
    return X_train, X_test


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame, args: argparse.Namespace) -> Tuple[FeatureBundle, FeatureBundle]:
    all_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

    miss = make_missing_indicators(all_df)
    ord_df = make_ordinal_dual_view(all_df)
    priors = make_physics_priors(all_df, ord_df)
    text_rules = make_text_rule_features(all_df)
    has_image = resolve_image_paths(all_df, args.image_dir)
    has_image_df = pd.DataFrame({"has_image": has_image}, index=all_df.index)

    excluded = set(TEXT_COLS + ["settlement_index", "id"])
    base_cols = [c for c in all_df.columns if c not in excluded]
    base_df = all_df[base_cols].copy()

    cat_cols = [c for c in base_df.columns if base_df[c].dtype == object]
    num_cols = [c for c in base_df.columns if c not in cat_cols]
    for c in num_cols:
        base_df[c] = pd.to_numeric(base_df[c], errors="coerce")

    feature_df = pd.concat([base_df, miss, ord_df, priors, has_image_df], axis=1)
    cat_cols = [c for c in feature_df.columns if feature_df[c].dtype == object]
    num_cols = [c for c in feature_df.columns if c not in cat_cols]

    for c in num_cols:
        feature_df[c] = pd.to_numeric(feature_df[c], errors="coerce")
        feature_df[c] = feature_df[c].fillna(feature_df[c].median())
    for c in cat_cols:
        feature_df[c] = feature_df[c].fillna("<MISSING>").astype(str)

    # LGB 数值视图
    lgb_tab = feature_df.copy()
    for c in cat_cols:
        lgb_tab[c] = lgb_tab[c].astype("category").cat.codes.astype(np.int32)

    text_emb = encode_text_embedding(all_df, args.text_model, args.text_batch_size, args.device)
    svd_dim = min(args.text_embed_dim, text_emb.shape[1] - 1) if text_emb.shape[1] > 1 else 1
    reducer = TruncatedSVD(n_components=svd_dim, random_state=args.seeds[0])
    text_emb_small = reducer.fit_transform(text_emb)

    lgb_np = np.hstack([
        lgb_tab.values.astype(np.float32),
        text_rules.values.astype(np.float32),
        text_emb_small.astype(np.float32),
    ])

    n_train = len(train_df)
    X_lgb_train, X_lgb_test = _build_lgb_dataframe(lgb_np[:n_train], lgb_np[n_train:])

    train_bundle = FeatureBundle(
        catboost_df=feature_df.iloc[:n_train].reset_index(drop=True),
        lgb_df=X_lgb_train.reset_index(drop=True),
        target=train_df["settlement_index"].values,
    )
    test_bundle = FeatureBundle(
        catboost_df=feature_df.iloc[n_train:].reset_index(drop=True),
        lgb_df=X_lgb_test.reset_index(drop=True),
        target=None,
    )
    return train_bundle, test_bundle


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _sample_training_data(X: pd.DataFrame, y: np.ndarray, sample_size: Optional[int], rng: np.random.RandomState) -> Tuple[pd.DataFrame, np.ndarray]:
    if sample_size is None or sample_size <= 0 or sample_size >= len(X):
        return X, y
    idx = rng.choice(len(X), size=sample_size, replace=False)
    return X.iloc[idx], y[idx]


def train_catboost_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    seeds: Sequence[int],
    n_folds: int,
    epochs: int,
    samples_per_epoch: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, float]:
    cat_features = [c for c in X.columns if X[c].dtype == object]
    cat_indices = [X.columns.get_loc(c) for c in cat_features]

    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)

    for seed in seeds:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        rep_oof = np.zeros(len(X), dtype=np.float64)
        rep_test = np.zeros(len(X_test), dtype=np.float64)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_va, y_va = X.iloc[va_idx], y[va_idx]

            pred_va = np.zeros(len(X_va), dtype=np.float64)
            pred_te = np.zeros(len(X_test), dtype=np.float64)
            for epoch in range(epochs):
                rng = np.random.RandomState(seed + fold * 100 + epoch)
                X_epoch, y_epoch = _sample_training_data(X_tr, y_tr, samples_per_epoch, rng)
                model = CatBoostRegressor(
                    iterations=2000,
                    learning_rate=0.03,
                    depth=8,
                    loss_function="RMSE",
                    eval_metric="RMSE",
                    random_state=seed + fold + epoch,
                    verbose=False,
                )
                model.fit(X_epoch, y_epoch, eval_set=(X_va, y_va), use_best_model=True, cat_features=cat_indices)
                pred_va += model.predict(X_va) / epochs
                pred_te += model.predict(X_test) / epochs

            rep_oof[va_idx] = pred_va
            rep_test += pred_te / n_folds

        oof += rep_oof / len(seeds)
        test_pred += rep_test / len(seeds)

    return oof, test_pred, rmse(y, oof)


def train_lgb_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    seeds: Sequence[int],
    n_folds: int,
    epochs: int,
    samples_per_epoch: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, float]:
    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)

    base_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=2200,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
        force_col_wise=True,
    )

    for seed in seeds:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        rep_oof = np.zeros(len(X), dtype=np.float64)
        rep_test = np.zeros(len(X_test), dtype=np.float64)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_va, y_va = X.iloc[va_idx], y[va_idx]

            assert X_tr.columns.is_unique
            assert X_va.columns.is_unique
            assert X_test.columns.is_unique
            assert list(X_tr.columns) == list(X_va.columns) == list(X_test.columns)

            pred_va = np.zeros(len(X_va), dtype=np.float64)
            pred_te = np.zeros(len(X_test), dtype=np.float64)
            for epoch in range(epochs):
                rng = np.random.RandomState(seed + fold * 100 + epoch)
                X_epoch, y_epoch = _sample_training_data(X_tr, y_tr, samples_per_epoch, rng)
                model = clone(base_model)
                model.set_params(random_state=seed + fold + epoch)
                model.fit(X_epoch, y_epoch)
                pred_va += model.predict(X_va) / epochs
                pred_te += model.predict(X_test) / epochs

            rep_oof[va_idx] = pred_va
            rep_test += pred_te / n_folds

        oof += rep_oof / len(seeds)
        test_pred += rep_test / len(seeds)

    return oof, test_pred, rmse(y, oof)


def train_ridge_stack(
    base_oof: pd.DataFrame,
    y: np.ndarray,
    base_test: pd.DataFrame,
    seeds: Sequence[int],
    n_folds: int,
    alpha: float,
    epochs: int,
    samples_per_epoch: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, float]:
    oof = np.zeros(len(base_oof), dtype=np.float64)
    test_pred = np.zeros(len(base_test), dtype=np.float64)

    for seed in seeds:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        rep_oof = np.zeros(len(base_oof), dtype=np.float64)
        rep_test = np.zeros(len(base_test), dtype=np.float64)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(base_oof), 1):
            X_tr, y_tr = base_oof.iloc[tr_idx], y[tr_idx]
            X_va, y_va = base_oof.iloc[va_idx], y[va_idx]

            pred_va = np.zeros(len(X_va), dtype=np.float64)
            pred_te = np.zeros(len(base_test), dtype=np.float64)
            for epoch in range(epochs):
                rng = np.random.RandomState(seed + fold * 100 + epoch)
                X_epoch, y_epoch = _sample_training_data(X_tr, y_tr, samples_per_epoch, rng)
                model = Ridge(alpha=alpha)
                model.fit(X_epoch, y_epoch)
                pred_va += model.predict(X_va) / epochs
                pred_te += model.predict(base_test) / epochs

            rep_oof[va_idx] = pred_va
            rep_test += pred_te / n_folds

        oof += rep_oof / len(seeds)
        test_pred += rep_test / len(seeds)

    return oof, test_pred, rmse(y, oof)


def _touch_auto_image_processor(model_name: str) -> None:
    """保留ViT使用AutoImageProcessor路径，避免AutoProcessor噪音。"""
    _ = AutoImageProcessor.from_pretrained(model_name)


def main(args: argparse.Namespace) -> None:
    print(f"[CONFIG] device={args.device} | folds={args.n_folds} | seed={args.seeds[0]}")

    train_df, test_df = read_data(args.data_dir, args.train_csv, args.test_csv)
    _touch_auto_image_processor(args.vision_model)
    train_bundle, test_bundle = build_features(train_df, test_df, args)

    y = train_bundle.target
    assert y is not None

    t0 = time.time()
    cat_oof, cat_test, cat_rmse = train_catboost_oof(
        train_bundle.catboost_df,
        y,
        test_bundle.catboost_df,
        seeds=args.seeds,
        n_folds=args.n_folds,
        epochs=args.catboost_epochs,
        samples_per_epoch=args.catboost_samples_per_epoch,
    )
    print(f"[CatBoost] OOF RMSE={cat_rmse:.4f} | loss={cat_rmse:.4f} | time={time.time() - t0:.1f}s")

    t1 = time.time()
    lgb_oof, lgb_test, lgb_rmse = train_lgb_oof(
        train_bundle.lgb_df,
        y,
        test_bundle.lgb_df,
        seeds=args.seeds,
        n_folds=args.n_folds,
        epochs=args.lgb_epochs,
        samples_per_epoch=args.lgb_samples_per_epoch,
    )
    print(f"[LGBM_Aux] OOF RMSE={lgb_rmse:.4f} | loss={lgb_rmse:.4f} | time={time.time() - t1:.1f}s")

    base_oof = pd.DataFrame({"cat": cat_oof, "lgb": lgb_oof})
    base_test = pd.DataFrame({"cat": cat_test, "lgb": lgb_test})

    t2 = time.time()
    stack_oof, stack_test, stack_rmse = train_ridge_stack(
        base_oof,
        y,
        base_test,
        seeds=args.seeds,
        n_folds=args.n_folds,
        alpha=args.meta_alpha,
        epochs=args.meta_ridge_epochs,
        samples_per_epoch=args.meta_ridge_samples_per_epoch,
    )
    print(f"[Stacking-Ridge] OOF RMSE={stack_rmse:.4f} | loss={stack_rmse:.4f} | time={time.time() - t2:.1f}s")

    print("\nFINAL RESULTS")
    print(f"CatBoost  {cat_rmse:.4f}")
    print(f"LGBM_Aux  {lgb_rmse:.4f}")
    print(f"Stacking  {stack_rmse:.4f}")

    submission = pd.DataFrame({"id": test_df["id"], "settlement_index": stack_test})
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "submission.csv")
    submission.to_csv(output_file, index=False)
    logger.info("Saved submission: %s", output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slim multimodal pipeline: CatBoost + LGBM + Ridge stacking")
    parser.add_argument("--data_dir", type=str, default=CONFIG["DATA_DIR"])
    parser.add_argument("--train_csv", type=str, default=CONFIG["TRAIN_CSV"])
    parser.add_argument("--test_csv", type=str, default=CONFIG["TEST_CSV"])
    parser.add_argument("--image_dir", type=str, default=CONFIG["IMAGE_DIR"])
    parser.add_argument("--output_dir", type=str, default=CONFIG["OUTPUT_DIR"])

    parser.add_argument("--text_model", type=str, default=CONFIG["TEXT_MODEL"])
    parser.add_argument("--vision_model", type=str, default=CONFIG["VISION_MODEL"])
    parser.add_argument("--device", type=str, default=CONFIG["DEVICE"])
    parser.add_argument("--text_batch_size", type=int, default=CONFIG["TEXT_BATCH_SIZE"])
    parser.add_argument("--image_batch_size", type=int, default=CONFIG["IMAGE_BATCH_SIZE"])
    parser.add_argument("--text_embed_dim", type=int, default=CONFIG["TEXT_EMBED_DIM"])

    parser.add_argument("--n_folds", type=int, default=CONFIG["N_FOLDS"])
    parser.add_argument("--repeats", type=int, default=CONFIG["REPEATS"])
    parser.add_argument("--seeds", nargs="+", type=int, default=CONFIG["SEEDS"])

    parser.add_argument("--lgb_epochs", type=int, default=CONFIG["LGB_EPOCHS"])
    parser.add_argument("--catboost_epochs", type=int, default=CONFIG["CATBOOST_EPOCHS"])
    parser.add_argument("--meta_ridge_epochs", type=int, default=CONFIG["META_RIDGE_EPOCHS"])
    parser.add_argument("--meta_catboost_epochs", type=int, default=CONFIG["META_CATBOOST_EPOCHS"])

    parser.add_argument("--lgb_samples_per_epoch", type=int, default=CONFIG["LGB_SAMPLES_PER_EPOCH"])
    parser.add_argument("--catboost_samples_per_epoch", type=int, default=CONFIG["CATBOOST_SAMPLES_PER_EPOCH"])
    parser.add_argument("--meta_ridge_samples_per_epoch", type=int, default=CONFIG["META_RIDGE_SAMPLES_PER_EPOCH"])
    parser.add_argument("--meta_catboost_samples_per_epoch", type=int, default=CONFIG["META_CATBOOST_SAMPLES_PER_EPOCH"])

    parser.add_argument("--meta_alpha", type=float, default=CONFIG["META_ALPHA"])
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
