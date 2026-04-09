import argparse
import logging
import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
import lightgbm as lgb

# ================= 统一配置参数 =================
CONFIG = {
    # 数据路径
    "DATA_DIR": r"G:\PythonProject\Info5558\app-of-gen-ai-deep-learning-wustl-spring-2026",
    "IMAGE_DIR": r"G:\PythonProject\Info5558\app-of-gen-ai-deep-learning-wustl-spring-2026\images_dataset",
    "OUTPUT_DIR": "output",
    "N_FOLDS": 5,
    "RANDOM_STATE": 42,
    # 设备与日志
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "LOG_LEVEL": "INFO",
    "SUPPRESS_HF_WARNINGS": True,
    # 模态编码模型
    "TEXT_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "VISION_MODEL": "google/vit-base-patch16-224",
    "TEXT_BATCH_SIZE": 64,
    "VISION_BATCH_SIZE": 16,
    # CatBoost 训练控制
    "CATBOOST_ITERATIONS": 2000,
    "CATBOOST_VERBOSE": 200,
    "CATBOOST_SUBSAMPLE": 0.8,
    # TabNet 训练控制
    "TABNET_MAX_EPOCHS": 200,
    "TABNET_PATIENCE": 30,
    "TABNET_BATCH_SIZE": 1024,
    "TABNET_VIRTUAL_BATCH_SIZE": 128,
    "TABNET_SAMPLE_SIZE": None,  # 若为整数，则每折训练前随机抽取指定数量样本；为None则使用全量数据
    # LightGBM 训练控制
    "LIGHTGBM_NUM_ROUNDS": 10000,
    "LIGHTGBM_EARLY_STOPPING_ROUNDS": 300,
    "LIGHTGBM_VERBOSE_EVAL": 500,
    "LIGHTGBM_SUBSAMPLE": 0.7,
    # 元模型控制
    "META_ALPHA": 1.0
}

# 日志与警告抑制初始化
logging.basicConfig(
    level=getattr(logging, CONFIG["LOG_LEVEL"]),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

if CONFIG["SUPPRESS_HF_WARNINGS"]:
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*")


def read_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Expected train.csv and test.csv in {data_dir}")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def resolve_image_paths(df: pd.DataFrame, image_dir: str) -> List[str]:
    paths = []
    for idx in df["id"]:
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            p = os.path.join(image_dir, f"{idx}{ext}")
            if os.path.exists(p):
                paths.append(p)
                break
        else:
            paths.append(None)
    return paths


def identify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    text_cols = ["exploration_log", "environmental_report", "incident_report", "image_prompt", "description"]
    numeric_cols, categorical_cols = [], []
    for col in df.columns:
        if col in text_cols or col in ["settlement_index", "id"]:
            continue
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            numeric_cols.append(col)
        else:
            try:
                coerced = pd.to_numeric(df[col], errors="coerce")
                if coerced.isna().sum() < 0.3 * len(df):
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            except Exception:
                categorical_cols.append(col)
    return numeric_cols, categorical_cols, text_cols


def preprocess_tabular(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str], is_train: bool = True,
                       label_encoders: dict = None) -> Tuple[pd.DataFrame, dict]:
    df_proc = df.copy()
    encoders = {} if label_encoders is None else label_encoders
    for col in numeric_cols:
        if df_proc[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
            df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce")
        df_proc[col] = df_proc[col].fillna(df_proc[col].median())
    for col in categorical_cols:
        if is_train:
            le = LabelEncoder()
            df_proc[col] = df_proc[col].astype(str).fillna("unknown")
            df_proc[col] = le.fit_transform(df_proc[col])
            encoders[col] = le
        else:
            le = encoders[col]
            df_proc[col] = df_proc[col].astype(str).fillna("unknown")
            df_proc[col] = df_proc[col].apply(lambda x: x if x in le.classes_ else "<UNK>")
            if "<UNK>" not in le.classes_:
                le.classes_ = np.append(le.classes_, "<UNK>")
            df_proc[col] = le.transform(df_proc[col])
    return df_proc, encoders


def concatenate_text_fields(df: pd.DataFrame, text_cols: List[str]) -> List[str]:
    return df[text_cols].fillna("").apply(lambda row: " \n ".join(row.values.astype(str)), axis=1).tolist()


def get_hf_text_embeddings(texts: List[str], model_name: str = CONFIG["TEXT_MODEL"],
                           batch_size: int = CONFIG["TEXT_BATCH_SIZE"], device: str = CONFIG["DEVICE"]) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            embeddings.append((sum_embeddings / sum_mask).cpu().numpy())
    return np.vstack(embeddings)


def get_hf_image_embeddings(image_paths: List[str], model_name: str = CONFIG["VISION_MODEL"],
                            batch_size: int = CONFIG["VISION_BATCH_SIZE"], device: str = CONFIG["DEVICE"]) -> Tuple[
    np.ndarray, np.ndarray]:
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    has_image_flags = []
    embeddings = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        valid_imgs, valid_pos = [], []
        for i, p in enumerate(batch_paths):
            if p and os.path.exists(p):
                try:
                    valid_imgs.append(Image.open(p).convert("RGB"))
                    valid_pos.append(i)
                    has_image_flags.append(1)
                except Exception:
                    has_image_flags.append(0)
            else:
                has_image_flags.append(0)
        dim = model.config.hidden_size
        batch_emb = np.zeros((len(batch_paths), dim))
        if valid_imgs:
            inputs = processor(images=valid_imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                feats = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[
                    :, 0, :]
                feats_np = feats.cpu().numpy()
            for pos, vec in zip(valid_pos, feats_np):
                batch_emb[pos] = vec
        embeddings.append(batch_emb)
    return np.vstack(embeddings), np.array(has_image_flags)


def train_catboost(X: pd.DataFrame, y: np.ndarray, categorical_features: List[str], X_test: pd.DataFrame,
                   n_folds: int = CONFIG["N_FOLDS"], random_state: int = CONFIG["RANDOM_STATE"]) -> Tuple[
    np.ndarray, np.ndarray]:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_preds, test_preds = np.zeros(len(X)), np.zeros(len(X_test))
    cat_features_indices = [X.columns.get_loc(col) for col in categorical_features if col in X.columns]
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        logger.info(f"[CatBoost] Fold {fold + 1}/{n_folds} training started.")
        model = CatBoostRegressor(
            iterations=CONFIG["CATBOOST_ITERATIONS"],
            learning_rate=0.03,
            depth=8,
            subsample=CONFIG["CATBOOST_SUBSAMPLE"],
            loss_function="RMSE",
            eval_metric="RMSE",
            random_state=random_state + fold,
            verbose=CONFIG["CATBOOST_VERBOSE"]
        )
        model.fit(
            X.iloc[train_idx], y[train_idx],
            eval_set=(X.iloc[valid_idx], y[valid_idx]),
            cat_features=cat_features_indices,
            use_best_model=True
        )
        oof_preds[valid_idx] = model.predict(X.iloc[valid_idx])
        test_preds += model.predict(X_test) / n_folds
        logger.info(f"[CatBoost] Fold {fold + 1} completed. Best Iteration: {model.best_iteration_}")
    return oof_preds, test_preds


def train_tabnet(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, categorical_dims: List[int], cat_idxs: List[int],
                 n_folds: int = CONFIG["N_FOLDS"], random_state: int = CONFIG["RANDOM_STATE"]) -> Tuple[
    np.ndarray, np.ndarray]:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_preds, test_preds = np.zeros(X.shape[0]), np.zeros(X_test.shape[0])
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        logger.info(f"[TabNet] Fold {fold + 1}/{n_folds} training started.")
        X_fold, y_fold = X[train_idx], y[train_idx].reshape(-1, 1)
        X_valid, y_valid = X[valid_idx], y[valid_idx].reshape(-1, 1)

        # 随机采样控制
        if CONFIG["TABNET_SAMPLE_SIZE"] is not None and CONFIG["TABNET_SAMPLE_SIZE"] < len(X_fold):
            rng = np.random.RandomState(random_state + fold)
            sample_idx = rng.choice(len(X_fold), size=CONFIG["TABNET_SAMPLE_SIZE"], replace=False)
            X_fold, y_fold = X_fold[sample_idx], y_fold[sample_idx]
            logger.info(f"[TabNet] Sampled {CONFIG['TABNET_SAMPLE_SIZE']} instances for this epoch.")

        model = TabNetRegressor(
            cat_idxs=cat_idxs,
            cat_dims=categorical_dims,
            n_d=16, n_a=16, n_steps=5, gamma=1.3, lambda_sparse=1e-5,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-3),
            mask_type="sparsemax", n_shared=1, n_independent=1,
            seed=random_state + fold,
            verbose=0
        )
        model.fit(
            X_train=X_fold, y_train=y_fold,
            eval_set=[(X_valid, y_valid)],
            max_epochs=CONFIG["TABNET_MAX_EPOCHS"],
            patience=CONFIG["TABNET_PATIENCE"],
            batch_size=CONFIG["TABNET_BATCH_SIZE"],
            virtual_batch_size=CONFIG["TABNET_VIRTUAL_BATCH_SIZE"],
            num_workers=0, drop_last=False
        )

        # 打印TabNet损失曲线
        history = model.history
        if "loss" in history and len(history["loss"]) > 0:
            logger.info(
                f"[TabNet] Fold {fold + 1} Train Loss: {history['loss'][-1]:.6f} | Val Loss: {history.get('val_loss', history['loss'])[-1]:.6f} | Epochs: {len(history['loss'])}")

        oof_preds[valid_idx] = model.predict(X_valid).squeeze()
        test_preds += model.predict(X_test).squeeze() / n_folds
    return oof_preds, test_preds


def train_lightgbm(X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, n_folds: int = CONFIG["N_FOLDS"],
                   random_state: int = CONFIG["RANDOM_STATE"]) -> Tuple[np.ndarray, np.ndarray]:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_preds, test_preds = np.zeros(len(X)), np.zeros(len(X_test))
    params = {
        "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
        "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
        "subsample": CONFIG["LIGHTGBM_SUBSAMPLE"], "subsample_freq": 1,
        "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1,
        "seed": random_state, "verbosity": -1
    }
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        logger.info(f"[LightGBM] Fold {fold + 1}/{n_folds} training started.")
        lgb_train = lgb.Dataset(X.iloc[train_idx], label=y[train_idx])
        lgb_valid = lgb.Dataset(X.iloc[valid_idx], label=y[valid_idx])
        model = lgb.train(
            params, lgb_train,
            num_boost_round=CONFIG["LIGHTGBM_NUM_ROUNDS"],
            valid_sets=[lgb_train, lgb_valid],
            valid_names=["train", "valid"],
            early_stopping_rounds=CONFIG["LIGHTGBM_EARLY_STOPPING_ROUNDS"],
            verbose_eval=CONFIG["LIGHTGBM_VERBOSE_EVAL"]
        )
        oof_preds[valid_idx] = model.predict(X.iloc[valid_idx], num_iteration=model.best_iteration)
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_folds
        logger.info(f"[LightGBM] Fold {fold + 1} completed. Best Iteration: {model.best_iteration}")
    return oof_preds, test_preds


def train_stacking_model(base_oof: np.ndarray, y: np.ndarray, base_test_preds: np.ndarray,
                         alpha: float = CONFIG["META_ALPHA"]) -> Tuple[np.ndarray, np.ndarray]:
    logger.info(f"[Stacking] Training Ridge meta-model with alpha={alpha}")
    meta_model = Ridge(alpha=alpha, random_state=CONFIG["RANDOM_STATE"])
    meta_model.fit(base_oof, y)
    meta_oof = meta_model.predict(base_oof)
    meta_test_preds = meta_model.predict(base_test_preds)
    meta_train_loss = mean_squared_error(y, meta_oof, squared=False)
    logger.info(f"[Stacking] Meta-model train RMSE: {meta_train_loss:.4f}")
    return meta_oof, meta_test_preds


def main(args: argparse.Namespace) -> None:
    logger.info(f"Initializing pipeline with device: {args.device}")
    train_df, test_df = read_data(args.data_dir)
    numeric_cols, categorical_cols, text_cols = identify_columns(train_df)

    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    full_df_proc, encoders = preprocess_tabular(full_df, numeric_cols, categorical_cols, is_train=True)
    train_proc = full_df_proc.iloc[:len(train_df)].copy()
    test_proc = full_df_proc.iloc[len(train_df):].copy().reset_index(drop=True)

    logger.info("Computing text embeddings via HF Transformers...")
    train_texts = concatenate_text_fields(train_df, text_cols)
    test_texts = concatenate_text_fields(test_df, text_cols)
    text_emb_train = get_hf_text_embeddings(train_texts, model_name=args.text_model, batch_size=args.batch_size,
                                            device=args.device)
    text_emb_test = get_hf_text_embeddings(test_texts, model_name=args.text_model, batch_size=args.batch_size,
                                           device=args.device)

    logger.info("Computing image embeddings via HF Transformers...")
    train_img_paths = resolve_image_paths(train_df, args.image_dir)
    test_img_paths = resolve_image_paths(test_df, args.image_dir)
    img_emb_train, has_img_train = get_hf_image_embeddings(train_img_paths, model_name=args.vision_model,
                                                           batch_size=args.img_batch_size, device=args.device)
    img_emb_test, has_img_test = get_hf_image_embeddings(test_img_paths, model_name=args.vision_model,
                                                         batch_size=args.img_batch_size, device=args.device)

    drop_cols = [c for c in (text_cols + ["settlement_index", "id"]) if c in train_proc.columns]
    X_tab = train_proc.drop(columns=drop_cols)
    X_tab_test = test_proc.drop(columns=drop_cols)
    X_tab["has_image"] = has_img_train
    X_tab_test["has_image"] = has_img_test
    y = train_df["settlement_index"].values

    cat_feature_indices = [X_tab.columns.get_loc(col) for col in categorical_cols if col in X_tab.columns]
    cat_dims = [int(X_tab[col].nunique()) for col in categorical_cols if col in X_tab.columns]

    X_tab_np = X_tab.values
    X_tab_test_np = X_tab_test.values
    X_tab_text = np.hstack([X_tab_np, text_emb_train, img_emb_train])
    X_tab_text_test = np.hstack([X_tab_test_np, text_emb_test, img_emb_test])

    X_lgb = pd.DataFrame(X_tab_text, columns=[f"f{i}" for i in range(X_tab_text.shape[1])])
    X_lgb_test = pd.DataFrame(X_tab_text_test, columns=[f"f{i}" for i in range(X_tab_text_test.shape[1])])

    cat_oof, cat_test = train_catboost(X_tab.reset_index(drop=True), y,
                                       [col for col in categorical_cols if col in X_tab.columns],
                                       X_tab_test.reset_index(drop=True), n_folds=args.n_folds)
    logger.info(f"CatBoost OOF RMSE: {np.sqrt(mean_squared_error(y, cat_oof)):.4f}")

    tabnet_oof, tabnet_test = train_tabnet(X_tab_text.astype(np.float32), y.astype(np.float32),
                                           X_tab_text_test.astype(np.float32), categorical_dims=cat_dims,
                                           cat_idxs=cat_feature_indices, n_folds=args.n_folds)
    logger.info(f"TabNet OOF RMSE: {np.sqrt(mean_squared_error(y, tabnet_oof)):.4f}")

    lgb_oof, lgb_test = train_lightgbm(X_lgb, y, X_lgb_test, n_folds=args.n_folds)
    logger.info(f"LightGBM OOF RMSE: {np.sqrt(mean_squared_error(y, lgb_oof)):.4f}")

    base_oof = np.vstack([cat_oof, tabnet_oof, lgb_oof]).T
    base_test_preds = np.vstack([cat_test, tabnet_test, lgb_test]).T
    meta_oof, meta_test = train_stacking_model(base_oof, y, base_test_preds, alpha=args.meta_alpha)
    logger.info(f"Stacking OOF RMSE: {np.sqrt(mean_squared_error(y, meta_oof)):.4f}")

    submission = pd.DataFrame({"id": test_df["id"], "settlement_index": meta_test})
    os.makedirs(args.output_dir, exist_ok=True)
    submission.to_csv(os.path.join(args.output_dir, "submission.csv"), index=False)
    logger.info(f"Submission saved to: {os.path.join(args.output_dir, 'submission.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HF-based Exoplanet Settlement Viability Pipeline")
    parser.add_argument("--data_dir", type=str, default=CONFIG["DATA_DIR"])
    parser.add_argument("--image_dir", type=str, default=CONFIG["IMAGE_DIR"])
    parser.add_argument("--output_dir", type=str, default=CONFIG["OUTPUT_DIR"])
    parser.add_argument("--n_folds", type=int, default=CONFIG["N_FOLDS"])
    parser.add_argument("--text_model", type=str, default=CONFIG["TEXT_MODEL"])
    parser.add_argument("--vision_model", type=str, default=CONFIG["VISION_MODEL"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["TEXT_BATCH_SIZE"])
    parser.add_argument("--img_batch_size", type=int, default=CONFIG["VISION_BATCH_SIZE"])
    parser.add_argument("--device", type=str, default=CONFIG["DEVICE"])
    parser.add_argument("--meta_alpha", type=float, default=CONFIG["META_ALPHA"])
    args = parser.parse_args()
    main(args)