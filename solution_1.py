import argparse
import os
import sys
import warnings
import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.exceptions import NotFittedError
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, logging as hf_logging
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
import lightgbm as lgb
from tqdm import tqdm

# ================= 统一配置参数 =================
CONFIG = {
    # 数据路径
    "DATA_DIR": r"G:\PythonProject\Info5558\app-of-gen-ai-deep-learning-wustl-spring-2026",
    "IMAGE_DIR": r"G:\PythonProject\Info5558\app-of-gen-ai-deep-learning-wustl-spring-2026\images_dataset",
    "OUTPUT_DIR": "output",

    # 训练超参数
    "N_FOLDS": 5,
    "EPOCHS": 200,  # TabNet训练轮次
    "SAMPLES_PER_EPOCH": None,  # 每轮随机采样样本数，None表示使用全部数据
    "BATCH_SIZE": 64,  # 文本嵌入批大小
    "IMG_BATCH_SIZE": 16,  # 图像嵌入批大小
    "TABNET_BATCH_SIZE": 1024,  # TabNet训练批大小
    "TABNET_VIRTUAL_BATCH_SIZE": 128,  # TabNet虚拟批大小
    "CATBOOST_ITERATIONS": 2000,  # CatBoost迭代次数
    "LIGHTGBM_ROUNDS": 5000,  # LightGBM最大迭代轮次

    # 模型配置
    "TEXT_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "VISION_MODEL": "google/vit-base-patch16-224",
    "META_ALPHA": 1.0,  # Ridge元模型正则化系数
    "RANDOM_STATE": 42,

    # 设备与日志
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "LOG_LEVEL": "INFO",
    "SUPPRESS_HF_WARNINGS": True,  # 是否抑制Hugging Face无关警告
}


# ================= 日志与警告配置 =================
def setup_logging():
    """配置日志输出格式与级别，抑制无关警告"""
    # 设置基础日志
    logging.basicConfig(
        level=getattr(logging, CONFIG["LOG_LEVEL"]),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout
    )

    # 抑制Hugging Face无关警告
    if CONFIG["SUPPRESS_HF_WARNINGS"]:
        warnings.filterwarnings("ignore", message=".*symlinks.*")
        warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
        hf_logging.set_verbosity_error()

    # 抑制PyTorch TabNet设备提示
    warnings.filterwarnings("ignore", message=".*Device used.*")

    # 设置环境变量避免CUDA异步错误难以定位
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


setup_logging()
logger = logging.getLogger(__name__)


# ================= 核心工具函数 =================
def read_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读取训练集与测试集，校验文件完整性"""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing train.csv or test.csv in {data_dir}")
    logger.info(
        f"Loaded train.csv ({os.path.getsize(train_path) // 1024} KB) and test.csv ({os.path.getsize(test_path) // 1024} KB)")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def resolve_image_paths(df: pd.DataFrame, image_dir: str) -> List[str]:
    """为每个样本解析图像路径，缺失则返回None"""
    paths = []
    missing_count = 0
    for idx in df["id"]:
        resolved = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = os.path.join(image_dir, f"{idx}{ext}")
            if os.path.exists(candidate):
                resolved = candidate
                break
        if resolved is None:
            missing_count += 1
        paths.append(resolved)
    if missing_count > 0:
        logger.warning(f"{missing_count}/{len(df)} samples have no corresponding image file")
    return paths


def identify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """自动识别数值型、类别型与文本型字段"""
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
    logger.info(
        f"Identified {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, {len(text_cols)} text columns")
    return numeric_cols, categorical_cols, text_cols


def preprocess_tabular(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str],
                       is_train: bool = True, label_encoders: dict = None) -> Tuple[pd.DataFrame, dict]:
    """预处理表格数据：数值型填中位数，类别型LabelEncode并保证索引安全"""
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
            if col not in encoders:
                raise NotFittedError(f"LabelEncoder for '{col}' not fitted")
            le = encoders[col]
            df_proc[col] = df_proc[col].astype(str).fillna("unknown")
            # 关键修复：未知类别映射为0而非新增类，避免索引越界
            df_proc[col] = df_proc[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df_proc[col] = le.transform(df_proc[col])
            # 二次校验：确保所有值在[0, n_classes)范围内
            max_valid = len(le.classes_) - 1
            df_proc[col] = df_proc[col].clip(0, max_valid)

    return df_proc, encoders


def concatenate_text_fields(df: pd.DataFrame, text_cols: List[str]) -> List[str]:
    """拼接多个文本字段为单字符串"""
    return df[text_cols].fillna("").apply(lambda row: " | ".join(row.values.astype(str)), axis=1).tolist()


def get_hf_text_embeddings(texts: List[str], model_name: str, batch_size: int, device: str) -> np.ndarray:
    """使用Hugging Face Transformers提取文本嵌入，带进度条"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    embeddings = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Text Embedding", unit="batch"):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            embeddings.append((summed / counts).cpu().numpy())

    return np.vstack(embeddings)


def get_hf_image_embeddings(image_paths: List[str], model_name: str, batch_size: int, device: str) -> Tuple[
    np.ndarray, np.ndarray]:
    """提取图像嵌入，缺失图片用零向量填充并记录掩码"""
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    hidden_dim = model.config.hidden_size
    embeddings, has_image = [], []

    for start in tqdm(range(0, len(image_paths), batch_size), desc="Image Embedding", unit="batch"):
        batch_paths = image_paths[start:start + batch_size]
        valid_imgs, valid_pos = [], []
        batch_mask = []

        for i, p in enumerate(batch_paths):
            if p and os.path.exists(p):
                try:
                    valid_imgs.append(Image.open(p).convert("RGB"))
                    valid_pos.append(i)
                    batch_mask.append(1)
                except Exception:
                    batch_mask.append(0)
            else:
                batch_mask.append(0)

        has_image.extend(batch_mask)
        batch_emb = np.zeros((len(batch_paths), hidden_dim), dtype=np.float32)

        if valid_imgs:
            inputs = processor(images=valid_imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                feats = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
            for pos, vec in zip(valid_pos, feats):
                batch_emb[pos] = vec
        embeddings.append(batch_emb)

    return np.vstack(embeddings), np.array(has_image, dtype=np.float32)


# ================= 基模型训练函数 =================
def train_catboost(X: pd.DataFrame, y: np.ndarray, categorical_features: List[str],
                   X_test: pd.DataFrame, n_folds: int, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """CatBoost五折训练，带简洁进度输出"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_preds, test_preds = np.zeros(len(X)), np.zeros(len(X_test))
    cat_indices = [X.columns.get_loc(c) for c in categorical_features if c in X.columns]

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X), 1):
        model = CatBoostRegressor(
            iterations=CONFIG["CATBOOST_ITERATIONS"], learning_rate=0.03, depth=8,
            loss_function="RMSE", eval_metric="RMSE", random_state=random_state + fold,
            verbose=0  # 关闭冗余输出
        )
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_set=(X.iloc[vl_idx], y[vl_idx]),
            cat_features=cat_indices, use_best_model=True
        )
        oof_preds[vl_idx] = model.predict(X.iloc[vl_idx])
        test_preds += model.predict(X_test) / n_folds
        logger.info(
            f"  Fold {fold}/{n_folds} - Best iter: {model.best_iteration_}, Val RMSE: {np.sqrt(mean_squared_error(y[vl_idx], oof_preds[vl_idx])):.4f}")

    return oof_preds, test_preds


def train_tabnet(X: np.ndarray, y: np.ndarray, X_test: np.ndarray,
                 categorical_dims: List[int], cat_idxs: List[int],
                 n_folds: int, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """TabNet训练，关键修复：确保类别索引严格在[0, dim)范围内"""
    # 核心修复：对输入数据进行二次校验，防止越界
    for idx, dim in zip(cat_idxs, categorical_dims):
        if idx < X.shape[1]:
            X[:, idx] = np.clip(X[:, idx], 0, dim - 1)
            X_test[:, idx] = np.clip(X_test[:, idx], 0, dim - 1)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_preds, test_preds = np.zeros(X.shape[0]), np.zeros(X_test.shape[0])

    # 采样配置
    samples_per_epoch = CONFIG["SAMPLES_PER_EPOCH"] if CONFIG["SAMPLES_PER_EPOCH"] else X.shape[0]

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X), 1):
        model = TabNetRegressor(
            cat_idxs=cat_idxs, cat_dims=categorical_dims,
            n_d=16, n_a=16, n_steps=5, gamma=1.3, lambda_sparse=1e-5,
            optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-3),
            mask_type='sparsemax', seed=random_state + fold, device_name=CONFIG["DEVICE"]
        )
        model.fit(
            X_train=X[tr_idx], y_train=y[tr_idx].reshape(-1, 1),
            eval_set=[(X[vl_idx], y[vl_idx].reshape(-1, 1))],
            max_epochs=CONFIG["EPOCHS"], patience=30,
            batch_size=CONFIG["TABNET_BATCH_SIZE"],
            virtual_batch_size=CONFIG["TABNET_VIRTUAL_BATCH_SIZE"],
            num_workers=0, drop_last=False
        )
        oof_preds[vl_idx] = model.predict(X[vl_idx]).squeeze()
        test_preds += model.predict(X_test).squeeze() / n_folds
        val_rmse = np.sqrt(mean_squared_error(y[vl_idx], oof_preds[vl_idx]))
        logger.info(f"  Fold {fold}/{n_folds} - Epochs: {CONFIG['EPOCHS']}, Val RMSE: {val_rmse:.4f}")

    return oof_preds, test_preds


def train_lightgbm(X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame,
                   n_folds: int, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """LightGBM训练，简洁日志输出"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_preds, test_preds = np.zeros(len(X)), np.zeros(len(X_test))
    params = {
        'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
        'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 20,
        'subsample': 0.7, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 0.1, 'seed': random_state, 'verbosity': -1
    }

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X), 1):
        lgb_train = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx])
        lgb_valid = lgb.Dataset(X.iloc[vl_idx], label=y[vl_idx])
        model = lgb.train(
            params, lgb_train, num_boost_round=CONFIG["LIGHTGBM_ROUNDS"],
            valid_sets=[lgb_valid], early_stopping_rounds=300, verbose_eval=False
        )
        oof_preds[vl_idx] = model.predict(X.iloc[vl_idx], num_iteration=model.best_iteration)
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_folds
        val_rmse = np.sqrt(mean_squared_error(y[vl_idx], oof_preds[vl_idx]))
        logger.info(f"  Fold {fold}/{n_folds} - Best iter: {model.best_iteration}, Val RMSE: {val_rmse:.4f}")

    return oof_preds, test_preds


def train_stacking_model(base_oof: np.ndarray, y: np.ndarray, base_test_preds: np.ndarray, alpha: float) -> Tuple[
    np.ndarray, np.ndarray]:
    """Ridge元模型训练"""
    meta = Ridge(alpha=alpha, random_state=CONFIG["RANDOM_STATE"])
    meta.fit(base_oof, y)
    return meta.predict(base_oof), meta.predict(base_test_preds)


# ================= 主流程 =================
def main(args: argparse.Namespace) -> None:
    logger.info("=" * 60)
    logger.info("Starting Exoplanet Settlement Viability Prediction Pipeline")
    logger.info(f"Device: {CONFIG['DEVICE']} | Folds: {CONFIG['N_FOLDS']} | TabNet Epochs: {CONFIG['EPOCHS']}")
    if CONFIG["SAMPLES_PER_EPOCH"]:
        logger.info(f"TabNet samples per epoch: {CONFIG['SAMPLES_PER_EPOCH']} (random subset)")
    logger.info("=" * 60)

    # 1. 数据加载
    train_df, test_df = read_data(args.data_dir)
    numeric_cols, categorical_cols, text_cols = identify_columns(train_df)

    # 2. 表格预处理
    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    full_proc, encoders = preprocess_tabular(full_df, numeric_cols, categorical_cols, is_train=True)
    train_proc = full_proc.iloc[:len(train_df)].copy()
    test_proc = full_proc.iloc[len(train_df):].copy().reset_index(drop=True)

    # 3. 文本嵌入
    logger.info("Computing text embeddings...")
    train_texts = concatenate_text_fields(train_df, text_cols)
    test_texts = concatenate_text_fields(test_df, text_cols)
    text_emb_train = get_hf_text_embeddings(train_texts, args.text_model, args.batch_size, args.device)
    text_emb_test = get_hf_text_embeddings(test_texts, args.text_model, args.batch_size, args.device)

    # 4. 图像嵌入
    logger.info("Computing image embeddings...")
    train_img_paths = resolve_image_paths(train_df, args.image_dir)
    test_img_paths = resolve_image_paths(test_df, args.image_dir)
    img_emb_train, has_img_train = get_hf_image_embeddings(train_img_paths, args.vision_model, args.img_batch_size,
                                                           args.device)
    img_emb_test, has_img_test = get_hf_image_embeddings(test_img_paths, args.vision_model, args.img_batch_size,
                                                         args.device)

    # 5. 特征组装
    drop_cols = [c for c in text_cols + ["settlement_index", "id"] if c in train_proc.columns]
    X_tab = train_proc.drop(columns=drop_cols).reset_index(drop=True)
    X_tab_test = test_proc.drop(columns=drop_cols).reset_index(drop=True)
    X_tab["has_image"] = has_img_train
    X_tab_test["has_image"] = has_img_test
    y = train_df["settlement_index"].values

    cat_indices = [X_tab.columns.get_loc(c) for c in categorical_cols if c in X_tab.columns]
    cat_dims = [int(X_tab[c].nunique()) for c in categorical_cols if c in X_tab.columns]

    X_tab_np = X_tab.values.astype(np.float32)
    X_tab_test_np = X_tab_test.values.astype(np.float32)
    X_fused_train = np.hstack([X_tab_np, text_emb_train, img_emb_train])
    X_fused_test = np.hstack([X_tab_test_np, text_emb_test, img_emb_test])
    X_lgb = pd.DataFrame(X_fused_train, columns=[f"f{i}" for i in range(X_fused_train.shape[1])])
    X_lgb_test = pd.DataFrame(X_fused_test, columns=[f"f{i}" for i in range(X_fused_test.shape[1])])

    # 6. 基模型训练
    logger.info("\nTraining base models...")

    logger.info("→ CatBoost (tabular only)")
    cat_oof, cat_test = train_catboost(
        X_tab, y, [c for c in categorical_cols if c in X_tab.columns],
        X_tab_test, n_folds=args.n_folds, random_state=CONFIG["RANDOM_STATE"]
    )
    cat_rmse = np.sqrt(mean_squared_error(y, cat_oof))

    logger.info("→ TabNet (tabular + text + image)")
    tabnet_oof, tabnet_test = train_tabnet(
        X_fused_train, y.astype(np.float32), X_fused_test,
        categorical_dims=cat_dims, cat_idxs=cat_indices,
        n_folds=args.n_folds, random_state=CONFIG["RANDOM_STATE"]
    )
    tabnet_rmse = np.sqrt(mean_squared_error(y, tabnet_oof))

    logger.info("→ LightGBM (tabular + text + image)")
    lgb_oof, lgb_test = train_lightgbm(
        X_lgb, y, X_lgb_test, n_folds=args.n_folds, random_state=CONFIG["RANDOM_STATE"]
    )
    lgb_rmse = np.sqrt(mean_squared_error(y, lgb_oof))

    # 7. 堆叠集成
    logger.info("\nTraining stacking meta-model...")
    base_oof = np.vstack([cat_oof, tabnet_oof, lgb_oof]).T
    base_test = np.vstack([cat_test, tabnet_test, lgb_test]).T
    meta_oof, meta_test = train_stacking_model(base_oof, y, base_test, alpha=args.meta_alpha)
    meta_rmse = np.sqrt(mean_squared_error(y, meta_oof))

    # 8. 结果汇总
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"{'Model':<12} {'OOF RMSE':>12}")
    logger.info("-" * 24)
    logger.info(f"{'CatBoost':<12} {cat_rmse:>12.4f}")
    logger.info(f"{'TabNet':<12} {tabnet_rmse:>12.4f}")
    logger.info(f"{'LightGBM':<12} {lgb_rmse:>12.4f}")
    logger.info(f"{'Stacking':<12} {meta_rmse:>12.4f}")
    logger.info("=" * 60)

    # 9. 保存提交
    submission = pd.DataFrame({"id": test_df["id"], "settlement_index": meta_test})
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "submission.csv")
    submission.to_csv(save_path, index=False)
    logger.info(f"Submission saved to: {save_path}")
    logger.info(f"Preview (first 5 rows):\n{submission.head().to_string(index=False)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exoplanet Settlement Prediction - HF Multi-modal Pipeline")
    parser.add_argument("--data_dir", type=str, default=CONFIG["DATA_DIR"])
    parser.add_argument("--image_dir", type=str, default=CONFIG["IMAGE_DIR"])
    parser.add_argument("--output_dir", type=str, default=CONFIG["OUTPUT_DIR"])
    parser.add_argument("--n_folds", type=int, default=CONFIG["N_FOLDS"])
    parser.add_argument("--text_model", type=str, default=CONFIG["TEXT_MODEL"])
    parser.add_argument("--vision_model", type=str, default=CONFIG["VISION_MODEL"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["BATCH_SIZE"])
    parser.add_argument("--img_batch_size", type=int, default=CONFIG["IMG_BATCH_SIZE"])
    parser.add_argument("--device", type=str, default=CONFIG["DEVICE"])
    parser.add_argument("--meta_alpha", type=float, default=CONFIG["META_ALPHA"])
    args = parser.parse_args()

    # 允许命令行覆盖CONFIG中的训练参数
    if args.n_folds != CONFIG["N_FOLDS"]:
        CONFIG["N_FOLDS"] = args.n_folds

    main(args)