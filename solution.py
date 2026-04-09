import argparse
import logging
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from catboost import CatBoostRegressor
import lightgbm as lgb


CONFIG = {
    # 路径与地址统一管理
    "DATA_DIR": ".",
    "IMAGE_DIR": "Partial image dataset",
    "OUTPUT_DIR": "output",
    "TEXT_MODEL": "answerdotai/ModernBERT-base",
    "VL_MODEL": "google/siglip2-base-patch16-224",

    # 设备与日志
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "LOG_LEVEL": "INFO",
    "SUPPRESS_HF_WARNINGS": True,  # 是否抑制Hugging Face无关警告

    # 训练基础参数
    "N_FOLDS": 5,
    "REPEATS": 2,
    "SEEDS": [42, 3407],
    "TEXT_BATCH_SIZE": 24,
    "IMAGE_BATCH_SIZE": 16,
    "META_ALPHA": 1.0,

    # 各模型与最终主模型的训练轮次配置
    "LGB_EPOCHS": 1,
    "CATBOOST_EPOCHS": 1,
    "META_RIDGE_EPOCHS": 1,
    "META_CATBOOST_EPOCHS": 1,

    # 每轮训练随机抽样样本数（None表示使用全部样本）
    "LGB_SAMPLES_PER_EPOCH": None,
    "CATBOOST_SAMPLES_PER_EPOCH": None,
    "META_RIDGE_SAMPLES_PER_EPOCH": None,
    "META_CATBOOST_SAMPLES_PER_EPOCH": None,
}

logging.basicConfig(
    level=getattr(logging, CONFIG["LOG_LEVEL"]),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
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
POSITIVE_TERMS = ["stable", "safe", "viable", "abundant", "promising", "habitable"]
NEGATIVE_TERMS = ["hostile", "risk", "hazard", "volatile", "danger", "toxic"]


@dataclass
class FeatureBundle:
    tab_df: pd.DataFrame
    tab_num: pd.DataFrame
    text_embeddings: np.ndarray
    text_rules: pd.DataFrame
    image_embeddings: np.ndarray
    image_stats: pd.DataFrame
    reliability: pd.DataFrame
    target: Optional[np.ndarray]


def read_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train, test


def resolve_image_paths(df: pd.DataFrame, image_dir: str) -> List[Optional[str]]:
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    out = []
    for idx in df["id"]:
        found = None
        for ext in exts:
            p = os.path.join(image_dir, f"{idx}{ext}")
            if os.path.exists(p):
                found = p
                break
        out.append(found)
    return out


def _count_terms(text: str, terms: Sequence[str]) -> int:
    t = text.lower()
    return sum(t.count(term) for term in terms)


def make_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [
        "environmental_report",
        "incident_report",
        "description",
        "environment_type",
        "oxygen_percent",
        "terraforming_difficulty",
        "colonization_cost_index",
    ]
    out = pd.DataFrame(index=df.index)
    for c in missing_cols:
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


def make_physics_priors(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    oxy = pd.to_numeric(df.get("oxygen_percent", np.nan), errors="coerce")
    g = pd.to_numeric(df.get("gravity_g", np.nan), errors="coerce")
    p = pd.to_numeric(df.get("atmospheric_pressure_atm", np.nan), errors="coerce")
    t_mean = pd.to_numeric(df.get("mean_temp_c", np.nan), errors="coerce")
    t_range = pd.to_numeric(df.get("temp_range_c", np.nan), errors="coerce")
    rad_ord = make_ordinal_dual_view(df).get("radiation_level_ord", pd.Series(-1, index=df.index))
    mag_ord = make_ordinal_dual_view(df).get("magnetic_field_ord", pd.Series(-1, index=df.index))
    storm_ord = make_ordinal_dual_view(df).get("storm_frequency_ord", pd.Series(-1, index=df.index))
    seismic_ord = make_ordinal_dual_view(df).get("seismic_activity_ord", pd.Series(-1, index=df.index))
    terraform = pd.to_numeric(df.get("terraforming_difficulty", np.nan), errors="coerce")
    cost = pd.to_numeric(df.get("colonization_cost_index", np.nan), errors="coerce")
    rare = pd.to_numeric(df.get("rare_mineral_index", np.nan), errors="coerce")
    energy = pd.to_numeric(df.get("energy_harvest_potential", np.nan), errors="coerce")
    albedo = pd.to_numeric(df.get("albedo", np.nan), errors="coerce")
    cloud = pd.to_numeric(df.get("cloud_coverage_percent", np.nan), errors="coerce")

    out["score_oxygen_suitability"] = np.exp(-((oxy - 21.0) ** 2) / (2 * 6.0**2))
    out["penalty_gravity_1g"] = np.abs(g - 1.0)
    out["score_pressure_band"] = np.exp(-((p - 1.0) ** 2) / (2 * 0.6**2))
    out["score_temp_comfort"] = np.exp(-((t_mean - 15.0) ** 2) / (2 * 18.0**2)) * np.exp(-(t_range / 45.0))
    out["score_magnetosphere_shield"] = (mag_ord + 1) - (rad_ord + 1)
    out["score_resource_composite"] = np.nanmean(np.vstack([rare, energy]), axis=0)
    out["score_cost_joint"] = np.nanmean(np.vstack([terraform, cost]), axis=0)
    out["score_risk_composite"] = np.nanmean(np.vstack([storm_ord, seismic_ord, rad_ord]), axis=0)
    out["score_aesthetic_proxy"] = np.nanmean(np.vstack([1 - np.abs(albedo - 0.35), cloud / 100.0]), axis=0)

    water = df.get("water_presence", pd.Series("", index=df.index)).astype(str).str.lower()
    out["water_binary"] = water.str.contains("abundant|surface|subsurface|ocean|river|lake").astype(np.int8)
    return out.replace([np.inf, -np.inf], np.nan)


def make_category_crosses(df: pd.DataFrame) -> pd.DataFrame:
    combos = [
        ("planet_type", "dominant_biome"),
        ("star_type", "atmosphere_primary_gases"),
        ("primary_hazard", "environment_type"),
        ("viability_class", "native_biology"),
        ("water_presence", "surface_composition"),
    ]
    out = pd.DataFrame(index=df.index)
    for a, b in combos:
        if a in df.columns and b in df.columns:
            out[f"cross_{a}_{b}"] = df[a].astype(str) + "__" + df[b].astype(str)
    return out


def make_text_rule_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in TEXT_COLS:
        col = df[c].fillna("").astype(str)
        out[f"{c}_len"] = col.str.len()
        out[f"{c}_sent"] = col.str.count(r"[.!?]+") + (col.str.len() > 0).astype(np.int8)
        out[f"{c}_num_density"] = (col.str.count(r"\d") / np.maximum(1, col.str.len())).astype(np.float32)
        out[f"{c}_risk_kw"] = col.apply(lambda x: _count_terms(x, RISK_TERMS)).astype(np.int16)
        out[f"{c}_resource_kw"] = col.apply(lambda x: _count_terms(x, RESOURCE_TERMS)).astype(np.int16)
        pos = col.apply(lambda x: _count_terms(x, POSITIVE_TERMS))
        neg = col.apply(lambda x: _count_terms(x, NEGATIVE_TERMS))
        out[f"{c}_pos_neg_ratio"] = (pos + 1) / (neg + 1)

    out["text_total_len"] = out[[f"{c}_len" for c in TEXT_COLS]].sum(axis=1)
    out["text_missing_count"] = df[TEXT_COLS].isna().sum(axis=1)
    return out


def encode_text_by_field(
    df: pd.DataFrame,
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    field_embs = []
    for col in TEXT_COLS:
        texts = df[col].fillna("").astype(str).tolist()
        col_embs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                last = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                mean_pool = (last * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1)
            col_embs.append(mean_pool.cpu().numpy())
        field_embs.append(np.vstack(col_embs))

    return np.hstack(field_embs)


def _image_tta_embeddings(model, processor, image: Image.Image, device: str) -> np.ndarray:
    variants = [image, image.transpose(Image.FLIP_LEFT_RIGHT)]
    embs = []
    for v in variants:
        inputs = processor(images=v, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        embs.append(out[0].cpu().numpy())
    return np.vstack(embs)


def encode_image_and_alignment(
    df: pd.DataFrame,
    image_paths: Sequence[Optional[str]],
    model_name: str,
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    text_prompts = df["image_prompt"].fillna("").astype(str).tolist()
    text_embs: List[np.ndarray] = []
    for i in range(0, len(text_prompts), batch_size):
        batch = text_prompts[i : i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
        text_embs.append(feats.cpu().numpy())
    prompt_emb = np.vstack(text_embs)

    img_emb_list: List[np.ndarray] = []
    stats = {"has_image": [], "img_tta_var": [], "img_txt_align": []}
    emb_dim = prompt_emb.shape[1]

    for i, p in enumerate(image_paths):
        if p is None:
            img_vec = np.zeros(emb_dim, dtype=np.float32)
            stats["has_image"].append(0)
            stats["img_tta_var"].append(0.0)
            stats["img_txt_align"].append(0.0)
            img_emb_list.append(img_vec)
            continue

        try:
            img = Image.open(p).convert("RGB")
            tta = _image_tta_embeddings(model, processor, img, device)
            img_vec = tta.mean(axis=0)
            tta_var = float(tta.var(axis=0).mean())
            num = float(np.dot(img_vec, prompt_emb[i]))
            den = float(np.linalg.norm(img_vec) * np.linalg.norm(prompt_emb[i]) + 1e-9)
            align = num / den
            stats["has_image"].append(1)
            stats["img_tta_var"].append(tta_var)
            stats["img_txt_align"].append(align)
            img_emb_list.append(img_vec)
        except Exception:
            img_emb_list.append(np.zeros(emb_dim, dtype=np.float32))
            stats["has_image"].append(0)
            stats["img_tta_var"].append(0.0)
            stats["img_txt_align"].append(0.0)

    return np.vstack(img_emb_list), pd.DataFrame(stats, index=df.index)


def build_features(df: pd.DataFrame, image_dir: str, args: argparse.Namespace) -> FeatureBundle:
    miss_df = make_missing_indicators(df)
    ord_df = make_ordinal_dual_view(df)
    prior_df = make_physics_priors(df)
    cross_df = make_category_crosses(df)
    text_rule_df = make_text_rule_features(df)

    excluded = set(TEXT_COLS + ["settlement_index", "id"])
    base_cols = [c for c in df.columns if c not in excluded]
    base_df = df[base_cols].copy()

    cat_cols = [c for c in base_df.columns if base_df[c].dtype == object]
    num_cols = [c for c in base_df.columns if c not in cat_cols]

    for c in num_cols:
        base_df[c] = pd.to_numeric(base_df[c], errors="coerce")

    tab_df = pd.concat([base_df, cross_df, ord_df, prior_df, miss_df], axis=1)
    cat_cols = [c for c in tab_df.columns if tab_df[c].dtype == object]
    num_cols = [c for c in tab_df.columns if c not in cat_cols]

    for c in num_cols:
        tab_df[c] = pd.to_numeric(tab_df[c], errors="coerce").fillna(tab_df[c].median())
    for c in cat_cols:
        tab_df[c] = tab_df[c].astype(str).fillna("<MISSING>")

    tab_num = tab_df.copy()
    for c in cat_cols:
        tab_num[c] = tab_num[c].astype("category").cat.codes.astype(np.int32)

    image_paths = resolve_image_paths(df, image_dir)
    text_embeddings = encode_text_by_field(df, args.text_model, args.batch_size, args.device)
    image_embeddings, image_stats = encode_image_and_alignment(df, image_paths, args.vl_model, args.img_batch_size, args.device)

    # 跨字段语义一致性：exploration_log vs environmental_report 的余弦相似度
    e0 = text_embeddings[:, : text_embeddings.shape[1] // len(TEXT_COLS)]
    e1 = text_embeddings[:, text_embeddings.shape[1] // len(TEXT_COLS) : 2 * text_embeddings.shape[1] // len(TEXT_COLS)]
    num = np.sum(e0 * e1, axis=1)
    den = np.linalg.norm(e0, axis=1) * np.linalg.norm(e1, axis=1) + 1e-9
    text_rule_df["txt_consistency_explore_env"] = num / den

    reliability = pd.concat([image_stats, miss_df, text_rule_df[["text_total_len", "text_missing_count"]]], axis=1)

    target = df["settlement_index"].values if "settlement_index" in df.columns else None
    return FeatureBundle(tab_df, tab_num, text_embeddings, text_rule_df, image_embeddings, image_stats, reliability, target)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _sample_training_data(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_size: Optional[int],
    rng: np.random.RandomState,
) -> Tuple[pd.DataFrame, np.ndarray]:
    if sample_size is None or sample_size <= 0 or sample_size >= len(X):
        return X, y
    idx = rng.choice(len(X), size=sample_size, replace=False)
    return X.iloc[idx], y[idx]


def cv_oof_predict(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    seeds: Sequence[int],
    n_folds: int,
    fit_params: Optional[dict] = None,
    epochs: int = 1,
    samples_per_epoch: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    fit_params = fit_params or {}
    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    fold_scores: List[float] = []

    for rep, seed in enumerate(seeds):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        rep_oof = np.zeros(len(X), dtype=np.float64)
        rep_test = np.zeros(len(X_test), dtype=np.float64)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
            X_tr = X.iloc[tr_idx]
            y_tr = y[tr_idx]
            X_va = X.iloc[va_idx]
            y_va = y[va_idx]

            epoch_va = np.zeros(len(X_va), dtype=np.float64)
            epoch_te = np.zeros(len(X_test), dtype=np.float64)
            epoch_losses: List[float] = []
            for epoch in range(epochs):
                model = clone(estimator)
                if hasattr(model, "random_state"):
                    model.set_params(random_state=seed + fold + epoch)
                rng = np.random.RandomState(seed + rep * 1000 + fold * 100 + epoch)
                X_epoch, y_epoch = _sample_training_data(X_tr, y_tr, samples_per_epoch, rng)
                model.fit(X_epoch, y_epoch, **fit_params)
                pred_va_epoch = model.predict(X_va)
                pred_te_epoch = model.predict(X_test)
                epoch_va += pred_va_epoch / epochs
                epoch_te += pred_te_epoch / epochs
                epoch_loss = rmse(y_va, pred_va_epoch)
                epoch_losses.append(epoch_loss)
                logger.info(
                    "[%s] rep=%d fold=%d epoch=%d loss_rmse=%.5f sample_size=%d",
                    model.__class__.__name__,
                    rep + 1,
                    fold + 1,
                    epoch + 1,
                    epoch_loss,
                    len(X_epoch),
                )
                print(f"[{model.__class__.__name__}] rep={rep + 1} fold={fold + 1} epoch={epoch + 1} loss={epoch_loss:.5f}")

            pred_va = epoch_va
            pred_te = epoch_te
            rep_oof[va_idx] = pred_va
            rep_test += pred_te / n_folds
            fold_scores.append(float(np.mean(epoch_losses)))
            logger.info("[%s] rep=%d fold=%d avg_loss_rmse=%.5f", model.__class__.__name__, rep + 1, fold + 1, fold_scores[-1])

        oof += rep_oof / len(seeds)
        test_pred += rep_test / len(seeds)

    return oof, test_pred, fold_scores


def train_catboost_oof(
    X_tab: pd.DataFrame,
    y: np.ndarray,
    X_tab_test: pd.DataFrame,
    cat_features: List[str],
    seeds: Sequence[int],
    n_folds: int,
    epochs: int = 1,
    samples_per_epoch: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(X_tab), dtype=np.float64)
    test_pred = np.zeros(len(X_tab_test), dtype=np.float64)

    for rep, seed in enumerate(seeds):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        rep_oof = np.zeros(len(X_tab), dtype=np.float64)
        rep_test = np.zeros(len(X_tab_test), dtype=np.float64)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tab)):
            X_tr = X_tab.iloc[tr_idx]
            y_tr = y[tr_idx]
            X_va = X_tab.iloc[va_idx]
            y_va = y[va_idx]

            epoch_va = np.zeros(len(X_va), dtype=np.float64)
            epoch_te = np.zeros(len(X_tab_test), dtype=np.float64)
            epoch_losses: List[float] = []
            for epoch in range(epochs):
                rng = np.random.RandomState(seed + rep * 1000 + fold * 100 + epoch)
                X_epoch, y_epoch = _sample_training_data(X_tr, y_tr, samples_per_epoch, rng)
                model = CatBoostRegressor(
                    iterations=2500,
                    learning_rate=0.03,
                    depth=8,
                    loss_function="RMSE",
                    eval_metric="RMSE",
                    random_state=seed + fold + epoch,
                    verbose=False,
                )
                model.fit(
                    X_epoch,
                    y_epoch,
                    eval_set=(X_va, y_va),
                    use_best_model=True,
                    cat_features=cat_features,
                )
                pred_va_epoch = model.predict(X_va)
                pred_te_epoch = model.predict(X_tab_test)
                epoch_va += pred_va_epoch / epochs
                epoch_te += pred_te_epoch / epochs
                epoch_loss = rmse(y_va, pred_va_epoch)
                epoch_losses.append(epoch_loss)
                logger.info(
                    "[CatBoost] rep=%d fold=%d epoch=%d loss_rmse=%.5f sample_size=%d",
                    rep + 1,
                    fold + 1,
                    epoch + 1,
                    epoch_loss,
                    len(X_epoch),
                )
                print(f"[CatBoost] rep={rep + 1} fold={fold + 1} epoch={epoch + 1} loss={epoch_loss:.5f}")

            rep_oof[va_idx] = epoch_va
            rep_test += epoch_te / n_folds
            logger.info("[CatBoost] rep=%d fold=%d avg_loss_rmse=%.5f", rep + 1, fold + 1, float(np.mean(epoch_losses)))

        oof += rep_oof / len(seeds)
        test_pred += rep_test / len(seeds)
        logger.info("[CatBoost] repeat=%d oof_rmse=%.5f", rep + 1, rmse(y, rep_oof))

    return oof, test_pred


def _try_tabm_branch(X: np.ndarray, y: np.ndarray, X_test: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        from tabm import TabMRegressor  # type: ignore
    except Exception:
        logger.info("TabM is not available; skip TabM branch.")
        return None

    model = TabMRegressor(random_state=42)
    model.fit(X, y)
    return model.predict(X), model.predict(X_test)


def _try_tabpfn_branch(X: np.ndarray, y: np.ndarray, X_test: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        from tabpfn import TabPFNRegressor  # type: ignore
    except Exception:
        logger.info("TabPFN is not available; skip TabPFN branch.")
        return None

    model = TabPFNRegressor(device="cpu")
    model.fit(X, y)
    return model.predict(X), model.predict(X_test)


def strict_meta_cv(
    meta_X: pd.DataFrame,
    y: np.ndarray,
    meta_test_X: pd.DataFrame,
    seeds: Sequence[int],
    n_folds: int,
    alpha: float,
    ridge_epochs: int = 1,
    ridge_samples_per_epoch: Optional[int] = None,
    catboost_epochs: int = 1,
    catboost_samples_per_epoch: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    ridge = Ridge(alpha=alpha)
    cb = CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=6, random_state=42, verbose=False)

    ridge_oof, ridge_test, _ = cv_oof_predict(
        ridge,
        meta_X,
        y,
        meta_test_X,
        seeds=seeds,
        n_folds=n_folds,
        epochs=ridge_epochs,
        samples_per_epoch=ridge_samples_per_epoch,
    )
    cb_oof, cb_test = train_catboost_oof(
        meta_X,
        y,
        meta_test_X,
        cat_features=[],
        seeds=seeds,
        n_folds=n_folds,
        epochs=catboost_epochs,
        samples_per_epoch=catboost_samples_per_epoch,
    )

    blend_oof = 0.5 * ridge_oof + 0.5 * cb_oof
    blend_test = 0.5 * ridge_test + 0.5 * cb_test
    logger.info("[Meta] strict cv blended rmse=%.5f", rmse(y, blend_oof))
    return blend_oof, blend_test


def main(args: argparse.Namespace) -> None:
    train_df, test_df = read_data(args.data_dir)
    train_bundle = build_features(train_df, args.image_dir, args)
    test_bundle = build_features(test_df, args.image_dir, args)

    y = train_bundle.target
    assert y is not None

    X_tab = train_bundle.tab_df
    X_tab_test = test_bundle.tab_df
    cat_features = [c for c in X_tab.columns if X_tab[c].dtype == object]

    X_tab_num = train_bundle.tab_num
    X_tab_num_test = test_bundle.tab_num

    X_text_only = pd.DataFrame(
        np.hstack([train_bundle.text_embeddings, train_bundle.text_rules.values]),
        index=X_tab.index,
    )
    X_text_only_test = pd.DataFrame(
        np.hstack([test_bundle.text_embeddings, test_bundle.text_rules.values]),
        index=X_tab_test.index,
    )

    X_img_only = pd.DataFrame(
        np.hstack([train_bundle.image_embeddings, train_bundle.image_stats.values]),
        index=X_tab.index,
    )
    X_img_only_test = pd.DataFrame(
        np.hstack([test_bundle.image_embeddings, test_bundle.image_stats.values]),
        index=X_tab_test.index,
    )

    X_tab_text = pd.concat([X_tab_num, X_text_only], axis=1)
    X_tab_text_test = pd.concat([X_tab_num_test, X_text_only_test], axis=1)

    X_tab_img = pd.concat([X_tab_num, X_img_only], axis=1)
    X_tab_img_test = pd.concat([X_tab_num_test, X_img_only_test], axis=1)

    X_all = pd.concat([X_tab_num, X_text_only, X_img_only], axis=1)
    X_all_test = pd.concat([X_tab_num_test, X_text_only_test, X_img_only_test], axis=1)

    # 有图样本模态 dropout 标记（提高缺模态鲁棒性）
    rng = np.random.RandomState(42)
    drop_mask = (train_bundle.image_stats["has_image"].values == 1) & (rng.rand(len(X_all)) < 0.15)
    X_all.loc[drop_mask, X_img_only.columns] = 0.0

    cat_oof, cat_test = train_catboost_oof(
        X_tab,
        y,
        X_tab_test,
        cat_features=cat_features,
        seeds=args.seeds,
        n_folds=args.n_folds,
        epochs=args.catboost_epochs,
        samples_per_epoch=args.catboost_samples_per_epoch,
    )

    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    tab_oof, tab_test, _ = cv_oof_predict(
        lgb_model, X_tab_num, y, X_tab_num_test, seeds=args.seeds, n_folds=args.n_folds, epochs=args.lgb_epochs, samples_per_epoch=args.lgb_samples_per_epoch
    )
    text_oof, text_test, _ = cv_oof_predict(
        clone(lgb_model), X_text_only, y, X_text_only_test, seeds=args.seeds, n_folds=args.n_folds, epochs=args.lgb_epochs, samples_per_epoch=args.lgb_samples_per_epoch
    )
    img_oof, img_test, _ = cv_oof_predict(
        clone(lgb_model), X_img_only, y, X_img_only_test, seeds=args.seeds, n_folds=args.n_folds, epochs=args.lgb_epochs, samples_per_epoch=args.lgb_samples_per_epoch
    )
    tt_oof, tt_test, _ = cv_oof_predict(
        clone(lgb_model), X_tab_text, y, X_tab_text_test, seeds=args.seeds, n_folds=args.n_folds, epochs=args.lgb_epochs, samples_per_epoch=args.lgb_samples_per_epoch
    )
    ti_oof, ti_test, _ = cv_oof_predict(
        clone(lgb_model), X_tab_img, y, X_tab_img_test, seeds=args.seeds, n_folds=args.n_folds, epochs=args.lgb_epochs, samples_per_epoch=args.lgb_samples_per_epoch
    )
    all_oof, all_test, _ = cv_oof_predict(
        clone(lgb_model), X_all, y, X_all_test, seeds=args.seeds, n_folds=args.n_folds, epochs=args.lgb_epochs, samples_per_epoch=args.lgb_samples_per_epoch
    )

    # Optional branches
    tabm_preds = _try_tabm_branch(X_all.values.astype(np.float32), y.astype(np.float32), X_all_test.values.astype(np.float32))
    tabpfn_preds = _try_tabpfn_branch(X_tab_num.values.astype(np.float32), y.astype(np.float32), X_tab_num_test.values.astype(np.float32))

    level1_oof = {
        "tabular_only": tab_oof,
        "text_only": text_oof,
        "image_only": img_oof,
        "tab_text": tt_oof,
        "tab_image": ti_oof,
        "tab_text_image": all_oof,
        "catboost_tab": cat_oof,
    }
    level1_test = {
        "tabular_only": tab_test,
        "text_only": text_test,
        "image_only": img_test,
        "tab_text": tt_test,
        "tab_image": ti_test,
        "tab_text_image": all_test,
        "catboost_tab": cat_test,
    }

    if tabm_preds is not None:
        level1_oof["tabm"] = tabm_preds[0]
        level1_test["tabm"] = tabm_preds[1]
    if tabpfn_preds is not None:
        level1_oof["tabpfn"] = tabpfn_preds[0]
        level1_test["tabpfn"] = tabpfn_preds[1]

    oof_df = pd.DataFrame(level1_oof)
    test_df_preds = pd.DataFrame(level1_test)

    # reliability-aware meta features
    rel_train = train_bundle.reliability.copy()
    rel_test = test_bundle.reliability.copy()

    oof_df["pred_std"] = oof_df.std(axis=1)
    oof_df["pred_range"] = oof_df.max(axis=1) - oof_df.min(axis=1)
    test_df_preds["pred_std"] = test_df_preds.std(axis=1)
    test_df_preds["pred_range"] = test_df_preds.max(axis=1) - test_df_preds.min(axis=1)

    meta_X = pd.concat([oof_df, rel_train.reset_index(drop=True)], axis=1)
    meta_X_test = pd.concat([test_df_preds, rel_test.reset_index(drop=True)], axis=1)

    meta_oof, meta_test = strict_meta_cv(
        meta_X,
        y,
        meta_X_test,
        seeds=args.seeds,
        n_folds=args.n_folds,
        alpha=args.meta_alpha,
        ridge_epochs=args.meta_ridge_epochs,
        ridge_samples_per_epoch=args.meta_ridge_samples_per_epoch,
        catboost_epochs=args.meta_catboost_epochs,
        catboost_samples_per_epoch=args.meta_catboost_samples_per_epoch,
    )
    logger.info("Final strict meta OOF RMSE: %.5f", rmse(y, meta_oof))

    submission = pd.DataFrame({"id": test_df["id"], "settlement_index": meta_test})
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "submission.csv")
    submission.to_csv(output_file, index=False)
    logger.info("Saved: %s", output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured multimodal stacking pipeline for settlement index prediction")
    parser.add_argument("--data_dir", type=str, default=CONFIG["DATA_DIR"])
    parser.add_argument("--image_dir", type=str, default=CONFIG["IMAGE_DIR"])
    parser.add_argument("--output_dir", type=str, default=CONFIG["OUTPUT_DIR"])
    parser.add_argument("--n_folds", type=int, default=CONFIG["N_FOLDS"])
    parser.add_argument("--repeats", type=int, default=CONFIG["REPEATS"])
    parser.add_argument("--seeds", nargs="+", type=int, default=CONFIG["SEEDS"])
    parser.add_argument("--text_model", type=str, default=CONFIG["TEXT_MODEL"])
    parser.add_argument("--vl_model", type=str, default=CONFIG["VL_MODEL"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["TEXT_BATCH_SIZE"])
    parser.add_argument("--img_batch_size", type=int, default=CONFIG["IMAGE_BATCH_SIZE"])
    parser.add_argument("--device", type=str, default=CONFIG["DEVICE"])
    parser.add_argument("--meta_alpha", type=float, default=CONFIG["META_ALPHA"])
    parser.add_argument("--lgb_epochs", type=int, default=CONFIG["LGB_EPOCHS"])
    parser.add_argument("--catboost_epochs", type=int, default=CONFIG["CATBOOST_EPOCHS"])
    parser.add_argument("--meta_ridge_epochs", type=int, default=CONFIG["META_RIDGE_EPOCHS"])
    parser.add_argument("--meta_catboost_epochs", type=int, default=CONFIG["META_CATBOOST_EPOCHS"])
    parser.add_argument("--lgb_samples_per_epoch", type=int, default=CONFIG["LGB_SAMPLES_PER_EPOCH"])
    parser.add_argument("--catboost_samples_per_epoch", type=int, default=CONFIG["CATBOOST_SAMPLES_PER_EPOCH"])
    parser.add_argument("--meta_ridge_samples_per_epoch", type=int, default=CONFIG["META_RIDGE_SAMPLES_PER_EPOCH"])
    parser.add_argument("--meta_catboost_samples_per_epoch", type=int, default=CONFIG["META_CATBOOST_SAMPLES_PER_EPOCH"])
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
