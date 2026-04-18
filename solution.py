import argparse
import contextlib
import io
import itertools
import logging
import os
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from scipy.optimize import nnls
except Exception:
    nnls = None
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

CONFIG = {
    # ========================================================================
    # 训练轮数总览（所有模型训练轮数的硬性规定入口，不使用任何早停机制）
    # ------------------------------------------------------------------------
    #   CatBoost          CatBoost_epoch
    #   LightGBM          LightGBM_epoch
    #   XGBoost           XGBoost_epoch
    #   TextModel         TextModel_epoch
    #   TabM              TabM_epoch
    #   KNN               惰性学习，无训练轮数
    # 备注：8000 训练样本 + 5 折 = 每折 6400 训练量，轮数为固定训练轮数
    # ========================================================================

    # ------ 每个模型的硬性训练轮数（无早停；LR 降了对应训练轮数要补齐以免欠拟合）------
    "CatBoost_epoch": 3500,             # CatBoost 固定迭代数（LR 0.018→0.015，多训 500 轮）
    "LightGBM_epoch": 2500,             # LightGBM 固定 boosting 轮数（LR 0.025→0.018，补齐）
    "XGBoost_epoch": 2000,              # XGBoost 固定 boosting 轮数（LR 0.03→0.02，补齐）
    "TextModel_epoch": 20,              # TextModel / AdapterRegressor 固定训练 epoch 数
    "TabM_epoch": 25,                   # TabM 固定训练 epoch 数

    # ------ 路径与基础设置 ------
    "DATA_DIR": r"G:\PythonProject\Info5558\app-of-gen-ai-deep-learning-wustl-spring-2026",  # 训练/测试 CSV 所在目录
    "OUTPUT_DIR": r"G:\PythonProject\Info5558\app-of-gen-ai-deep-learning-wustl-spring-2026\result",  # 提交结果输出目录
    "N_FOLDS": 5,                       # 交叉验证折数
    "RANDOM_STATE": 42,                 # 全局随机种子（fold 划分、PCA、ElasticNet 用）
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",  # 训练设备，优先 GPU
    "LOG_LEVEL": "INFO",                # 日志级别（DEBUG / INFO / WARNING）
    "SUPPRESS_HF_WARNINGS": True,       # 是否屏蔽 HuggingFace 和 warnings 的冗余输出

    # ------ 文本编码器（HF Transformers）------
    "TEXT_MODEL": "sentence-transformers/all-mpnet-base-v2",   # 主文本编码器，MPNet-base 768 维
    "TEXT_MODEL_FALLBACK": "BAAI/bge-small-en-v1.5",           # 下载失败时降级模型，BGE-small 384 维
    "TEXT_BATCH_SIZE": 32,              # 文本编码推理 batch（MPNet batch=32 约占 2G 显存）
    "TEXT_MAX_LENGTH": 256,             # tokenizer 最大截断长度
    "L2_NORMALIZE_EMBEDDINGS": True,    # 是否对文本向量做 L2 归一化
    "TEXT_PCA_DIM": 96,                 # 文本嵌入 PCA 降维后的维度

    # ------ CatBoost（梯度提升树主力，稳定性为先：小步长 + 强正则 + 低随机性）------
    "CATBOOST_LEARNING_RATE": 0.015,    # 学习率（降一档，波动更小）
    "CATBOOST_DEPTH": 6,                # 树深度
    "CATBOOST_L2_LEAF_REG": 20.0,       # L2 叶节点正则系数（加强）
    "CATBOOST_SUBSAMPLE": 0.85,         # 行采样比例
    "CATBOOST_RSM": 0.8,                # 列采样比例（rsm = Random Subspace Method）
    "CATBOOST_RANDOM_STRENGTH": 0.8,    # 分裂点选择随机强度（降低，fold 间更稳）
    "CATBOOST_BAGGING_TEMPERATURE": 0.5, # Bayesian bootstrap 温度（降低，降方差）
    "CATBOOST_VERBOSE": 0,              # CatBoost 打印详细度（0=静默）

    # ------ LightGBM（Huber 损失 + 较大叶子数，与 CatBoost 解耦；配强正则抑制 fold 间波动）------
    "LIGHTGBM_OBJECTIVE": "huber",      # Huber 损失，与 CatBoost 的 L2 解耦，降低预测相关性
    "LIGHTGBM_HUBER_ALPHA": 0.9,        # Huber 的分位阈值 δ 控制
    "LIGHTGBM_LEARNING_RATE": 0.018,    # 学习率（降一档，波动更小）
    "LIGHTGBM_NUM_LEAVES": 47,          # 单棵树最大叶子数（63 偏大易抖动，降到 47）
    "LIGHTGBM_MIN_CHILD_SAMPLES": 28,   # 叶节点最少样本数（加大，避免叶子过细）
    "LIGHTGBM_MAX_DEPTH": 8,            # 限制最大深度（-1 无限会增加方差）
    "LIGHTGBM_SUBSAMPLE": 0.75,         # 行采样比例
    "LIGHTGBM_COLSAMPLE": 0.7,          # 列采样比例（feature_fraction）
    "LIGHTGBM_L1": 0.1,                 # L1 正则（加强）
    "LIGHTGBM_L2": 3.0,                 # L2 正则（加强）

    # ------ XGBoost（L2 损失，配小步长 + 强正则以稳住 fold 间方差）------
    "XGBOOST_LEARNING_RATE": 0.02,      # 学习率（0.03 偏高，降一档更稳）
    "XGBOOST_MAX_DEPTH": 5,             # 树深度
    "XGBOOST_MIN_CHILD_WEIGHT": 6,      # 叶节点最小 hessian 和（加大，避免过拟合小叶）
    "XGBOOST_SUBSAMPLE": 0.8,           # 行采样比例
    "XGBOOST_COLSAMPLE": 0.7,           # 列采样比例
    "XGBOOST_REG_ALPHA": 0.1,           # L1 正则
    "XGBOOST_REG_LAMBDA": 5.0,          # L2 正则（加强）

    # ------ KNN（惰性近邻学习器，在 comp 子分空间捕捉局部一致性）------
    "KNN_K": 12,                        # 近邻数 k（原 25 过平滑，改小以保留局部细节）

    # ------ 文本头（TextModel，AdapterRegressor）------
    "ADAPTER_HIDDEN": 256,              # 隐藏层维度
    "ADAPTER_BOTTLENECK": 64,           # 瓶颈层维度（兼作传给 LightGBM 的辅助特征维度）
    "ADAPTER_BATCH_SIZE": 256,          # 训练 batch size
    "ADAPTER_LR": 1.0e-3,               # AdamW 初始学习率（1.4e-3 偏高，降到 1e-3 更稳）
    "ADAPTER_WEIGHT_DECAY": 4e-4,       # AdamW 权重衰减（加强）
    "ADAPTER_AUX_WEIGHT": 0.006,        # 瓶颈层 L2 辅助损失权重（正则）
    "ADAPTER_WARMUP_RATIO": 0.15,       # 学习率 warmup 占总 steps 比例（拉长，前期更稳）

    # ------ TabM（MoE 风格表格网络，fold 间波动最大，主要稳定化对象）------
    "TABM_HIDDEN": 160,                 # 隐藏层维度
    "TABM_EMBED_DIM": 16,               # 类别 embedding 维度上限
    "TABM_K": 3,                        # expert 分支数（MoE 的 K）
    "TABM_BATCH_SIZE": 512,             # 训练 batch size
    "TABM_LR": 3.0e-4,                  # AdamW 初始学习率（4.5e-4 抖动大，降到 3e-4）
    "TABM_WEIGHT_DECAY": 5e-4,          # AdamW 权重衰减（翻倍，抑制方差）
    "TABM_AUX_WEIGHT": 0.0025,          # 门控均衡辅助损失权重
    "TABM_WARMUP_RATIO": 0.20,          # 学习率 warmup 占总 steps 比例（拉长到 20%）
    "TABM_TEXT_AUX_DIM": 16,            # 拼到 TabM 数值输入的文本 PCA 前 N 维
    "TABM_DROP_NUM_COLS": [],           # TabM 需要排除的数值列名列表

    # ------ 困难样本加权 ------
    "HARD_WEIGHT_ALPHA": 2.0,           # 困难样本权重上限的增益系数（w = 1 + α·pct^β）
    "HARD_WEIGHT_POWER": 2.0,           # 困难程度百分位的幂次 β
    "USE_HARD_SAMPLE_WEIGHT": True,     # 是否启用困难样本加权（关闭则 w 恒为 1）

    # ------ Stacking 子集搜索（subset search）------
    "STACK_MAX_MODELS": 6,              # 参与 stacking 的最多 base 模型数
    "STACK_MIN_MODELS": 2,              # 参与 stacking 的最少 base 模型数
    "STACK_FORCE_INCLUDE": ["CatBoost", "XGBoost"],  # 强制包含的 base 模型（搜索时不可剔除）
    "STACK_SOLVERS": ["convex", "nnls", "ridge"],    # 要尝试的 solver 列表（convex 优先）
    "STACK_CONVEX_MAXITER": 500,        # 凸组合 SLSQP 求解器最大迭代
    "STACK_ELASTICNET_MAXITER": 20000,  # ElasticNetCV 坐标下降最大迭代
    "STACK_RIDGE_ALPHAS_COUNT": 32,     # RidgeCV alpha 网格大小
    "STACK_CORR_PENALTY": 0.0015,       # 子集残差相关度惩罚系数（越大越偏爱多样性）
    "STACK_HARD_GAIN_PENALTY": 0.0040,  # 子集内负向 hard_gain 的惩罚系数
    "STACK_GLOBAL_GAIN_FLOOR": -5.0,    # 全局 gain 下界（放宽：让 LightGBM/TabM/KNN 都能进候选池）
    "STACK_HARD_GAIN_FLOOR": -5.0,      # 困难样本 gain 下界（放宽：同上）
    "DIAG_TOPK_FEATURES": 30,           # CatBoost feature importance 打印前 K 项

    # ------ 目标编码（Target Encoding）------
    "TE_SMOOTHING": 20.0,               # Bayesian smoothing 系数（越大越偏向全局均值）

    # ------ 单种子训练（不做多-seed 平均）------
    # 所有模型复用 RANDOM_STATE 作为 seed，单次训练完成。
}

if CONFIG["SUPPRESS_HF_WARNINGS"]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=getattr(logging, CONFIG["LOG_LEVEL"]),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
for _name in ["httpx", "httpcore", "huggingface_hub", "transformers", "urllib3", "PIL", "filelock"]:
    _logger = logging.getLogger(_name)
    _logger.setLevel(logging.ERROR)
    _logger.propagate = False


@contextlib.contextmanager
def suppress_stdout_stderr():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def log_stage(name: str, y_true: np.ndarray, y_pred: np.ndarray, elapsed: float) -> None:
    logger.info(f"[{name}] OOF RMSE={rmse(y_true, y_pred):.4f} | loss={mse(y_true, y_pred):.6f} | time={elapsed:.1f}s")


def log_prediction_diagnostics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    bias = float(np.mean(y_pred - y_true))
    pred_std = float(np.std(y_pred))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 else 0.0
    logger.info(
        f"[{name}::Diag] RMSE={rmse(y_true, y_pred):.4f} | MSE={mse(y_true, y_pred):.6f} | "
        f"MAE={mae:.4f} | Bias={bias:.4f} | PredStd={pred_std:.4f} | Corr={corr:.4f}"
    )


def make_target_bins(y: np.ndarray, n_bins: int = 12) -> np.ndarray:
    q = min(n_bins, max(4, len(np.unique(y)) // 200 + 6))
    try:
        bins = pd.qcut(y, q=q, labels=False, duplicates="drop")
    except Exception:
        bins = pd.cut(y, bins=q, labels=False)
    return np.asarray(bins, dtype=np.int32)


def make_folds(y: np.ndarray, n_splits: int, random_state: int):
    bins = make_target_bins(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros(len(y)), bins))


def compute_hard_weights(error_signal: np.ndarray) -> np.ndarray:
    abs_e = np.abs(error_signal).astype(np.float32)
    order = np.argsort(abs_e)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(abs_e), dtype=np.float32)
    pct = ranks / max(1.0, float(len(abs_e) - 1))
    weights = 1.0 + CONFIG["HARD_WEIGHT_ALPHA"] * np.power(pct, CONFIG["HARD_WEIGHT_POWER"])
    return weights.astype(np.float32)



def weighted_mse_tensor(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2 * weight).mean()


def read_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Expected train.csv and test.csv in {data_dir}")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def identify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    text_cols = [c for c in ["exploration_log", "image_prompt", "description"] if c in df.columns]
    numeric_cols, categorical_cols = [], []
    for col in df.columns:
        if col in text_cols or col in ["settlement_index", "id"]:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols, text_cols


def preprocess_tabular(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str], is_train: bool,
                       label_encoders: Optional[Dict[str, LabelEncoder]] = None) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    df_proc = df.copy()
    encoders = {} if label_encoders is None else label_encoders
    for col in numeric_cols:
        df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce")
        med = df_proc[col].median()
        if pd.isna(med):
            med = 0.0
        df_proc[col] = df_proc[col].fillna(med)
    for col in categorical_cols:
        df_proc[col] = df_proc[col].astype(str).fillna("unknown")
        if is_train:
            le = LabelEncoder()
            vals = df_proc[col].tolist()
            if "<UNK>" not in vals:
                vals.append("<UNK>")
            le.fit(vals)
            df_proc[col] = le.transform(df_proc[col])
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df_proc[col] = df_proc[col].apply(lambda x: x if x in known else "<UNK>")
            df_proc[col] = le.transform(df_proc[col])
    return df_proc, encoders


def concatenate_text_fields(df: pd.DataFrame, text_cols: List[str]) -> List[str]:
    if not text_cols:
        return [""] * len(df)
    return df[text_cols].fillna("").apply(lambda row: " \n ".join(row.values.astype(str)), axis=1).tolist()


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-8, None)


def get_hf_text_embeddings(texts: List[str], model_name: str, batch_size: int, device: str, max_length: int) -> np.ndarray:
    start = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    except Exception as e:
        fallback = CONFIG.get("TEXT_MODEL_FALLBACK")
        if fallback and fallback != model_name:
            logger.warning(f"Failed to load TEXT_MODEL={model_name} ({e}); falling back to {fallback}")
            tokenizer = AutoTokenizer.from_pretrained(fallback)
            model = AutoModel.from_pretrained(fallback).to(device)
        else:
            raise
    model.eval()
    amp_enabled = device.startswith("cuda")
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1)
                token_embeddings = outputs.last_hidden_state
                pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = pooled.detach().float().cpu().numpy().astype(np.float32)
        all_emb.append(pooled)
    emb = np.vstack(all_emb).astype(np.float32)
    if CONFIG["L2_NORMALIZE_EMBEDDINGS"]:
        emb = _l2_normalize(emb).astype(np.float32)
    del model, tokenizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    logger.info(f"Text embeddings shape={emb.shape} | time={time.time() - start:.1f}s")
    return emb


def fit_pca_features(train_arr: np.ndarray, test_arr: np.ndarray, n_components: int, name: str) -> Tuple[np.ndarray, np.ndarray]:
    n_components = int(min(n_components, train_arr.shape[1], max(2, train_arr.shape[0] - 1)))
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_arr)
    test_scaled = scaler.transform(test_arr)
    pca = PCA(n_components=n_components, random_state=CONFIG["RANDOM_STATE"])
    train_pca = pca.fit_transform(train_scaled).astype(np.float32)
    test_pca = pca.transform(test_scaled).astype(np.float32)
    logger.info(f"{name} PCA shape train={train_pca.shape} test={test_pca.shape} explained={pca.explained_variance_ratio_.sum():.4f}")
    return train_pca, test_pca




NEG_WORDS = [
    "ominous", "hazard", "toxic", "lethal", "hostile", "dangerous", "storm",
    "radiation", "unstable", "collapse", "contamin", "failure", "anomal",
    "warning", "threat", "risk", "volatile", "extreme", "deadly", "corros",
]
POS_WORDS = [
    "pristine", "habitable", "stable", "abundant", "mild", "temperate",
    "breathable", "fertile", "promising", "suitable", "ideal", "rich",
    "calm", "gentle", "lush", "thriving",
]


def text_keyword_features(texts: List[str]) -> np.ndarray:
    """Hand-crafted neg/pos keyword counts + net sentiment + length features.
    5 cols: kw_neg, kw_pos, kw_net, kw_net_ratio, kw_len."""
    feats = np.zeros((len(texts), 5), dtype=np.float32)
    for i, t in enumerate(texts):
        tl = str(t).lower()
        neg = sum(tl.count(w) for w in NEG_WORDS)
        pos = sum(tl.count(w) for w in POS_WORDS)
        n_words = max(len(tl.split()), 1)
        feats[i, 0] = float(neg)
        feats[i, 1] = float(pos)
        feats[i, 2] = float(pos - neg)
        feats[i, 3] = float(pos - neg) / float(n_words)
        feats[i, 4] = float(n_words)
    return feats


def _safe_num(s: pd.Series, default: float = 0.0) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    return out.fillna(default).astype(np.float32)


def _map_levels(s: pd.Series, mapping: Dict[str, float], default: float = 0.0,
                name: Optional[str] = None, debug: bool = True) -> pd.Series:
    miss_counter = {"n": 0, "total": 0}
    def one(x):
        miss_counter["total"] += 1
        x = str(x).strip().lower()
        for k, v in mapping.items():
            if k in x:
                return float(v)
        miss_counter["n"] += 1
        return float(default)
    out = s.fillna("").map(one).astype(np.float32)
    if debug and name is not None and miss_counter["n"] > 0:
        logger.info(
            f"[_map_levels] column={name:<28s} fell_to_default={miss_counter['n']}/{miss_counter['total']} "
            f"({100.0 * miss_counter['n'] / max(1, miss_counter['total']):.2f}%) default={default}"
        )
    return out


def kfold_target_encode(series_train: pd.Series, y_train: np.ndarray, series_test: pd.Series,
                        folds: List, smoothing: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """Out-of-fold target encoding with Bayesian smoothing. Returns (train_oof, test_full)."""
    s_tr = series_train.astype(str).fillna("unknown").str.strip().values
    s_te = series_test.astype(str).fillna("unknown").str.strip().values
    y_tr = np.asarray(y_train, dtype=np.float64)
    global_mean = float(y_tr.mean())
    train_enc = np.full(len(s_tr), global_mean, dtype=np.float32)
    for tr_idx, va_idx in folds:
        grp: Dict[str, List[float]] = {}
        for i in tr_idx:
            grp.setdefault(s_tr[i], []).append(y_tr[i])
        mapping: Dict[str, float] = {}
        for k, vals in grp.items():
            cnt = len(vals)
            mean = float(np.mean(vals))
            mapping[k] = (cnt * mean + smoothing * global_mean) / (cnt + smoothing)
        for i in va_idx:
            train_enc[i] = float(mapping.get(s_tr[i], global_mean))
    # Full-train mapping for test
    grp_full: Dict[str, List[float]] = {}
    for i in range(len(s_tr)):
        grp_full.setdefault(s_tr[i], []).append(y_tr[i])
    full_map: Dict[str, float] = {}
    for k, vals in grp_full.items():
        cnt = len(vals)
        mean = float(np.mean(vals))
        full_map[k] = (cnt * mean + smoothing * global_mean) / (cnt + smoothing)
    test_enc = np.array([full_map.get(k, global_mean) for k in s_te], dtype=np.float32)
    return train_enc, test_enc


def engineer_target_components(df_raw: pd.DataFrame, env_type_score: pd.Series) -> pd.DataFrame:
    """Build 6 target-formula sub-scores (0-100 scale) + priors + interactions.

    settlement_index ~= 0.40*H + 0.25*E + 0.15*R + 0.10*S + 0.10*Ec + 0.05*A
      H  = environmental habitability (temperature / oxygen / gravity / pressure / day / stability / toxicity)
      E  = environment_type target-encoded score
      R  = resources (rare minerals, energy, water)
      S  = safety (magnetic, radiation, seismic, storm, native biology threat)
      Ec = economic / strategic (strategy vs terraforming+cost)
      A  = aesthetic (albedo, cloud coverage)
    """
    df = pd.DataFrame(index=df_raw.index)
    temp = _safe_num(df_raw.get("mean_temp_c", 0))
    oxy = _safe_num(df_raw.get("oxygen_percent", 0))
    grav = _safe_num(df_raw.get("gravity_g", 0))
    press = _safe_num(df_raw.get("atmospheric_pressure_atm", 0))
    day = _safe_num(df_raw.get("day_length_hours", 24))
    temp_range = _safe_num(df_raw.get("temp_range_c", 0))
    rare = _safe_num(df_raw.get("rare_mineral_index", 0))
    strategic = _safe_num(df_raw.get("strategic_value_rating", 0))
    terra = _safe_num(df_raw.get("terraforming_difficulty", 0))
    cost = _safe_num(df_raw.get("colonization_cost_index", 0))
    albedo = _safe_num(df_raw.get("albedo", 0.35))
    cloud = _safe_num(df_raw.get("cloud_coverage_percent", 45.0))
    co2 = _safe_num(df_raw.get("co2_percent", 0))
    ch4 = _safe_num(df_raw.get("methane_percent", 0))

    water = _map_levels(
        df_raw.get("water_presence", pd.Series(index=df_raw.index, dtype=object)),
        {"global ocean": 1.0, "ocean": 0.95, "seas": 0.9, "subsurface": 0.7,
         "trace": 0.3, "ice": 0.5, "vapor": 0.4, "none": 0.0, "no": 0.0},
        default=0.2, name="water_presence",
    )
    energy = _map_levels(
        df_raw.get("energy_harvest_potential", pd.Series(index=df_raw.index, dtype=object)),
        {"extreme": 1.3, "very high": 1.2, "high": 1.0, "moderate": 0.6,
         "medium": 0.6, "low": 0.25, "none": 0.0},
        default=0.4, name="energy_harvest_potential",
    )
    radiation = _map_levels(
        df_raw.get("radiation_level", pd.Series(index=df_raw.index, dtype=object)),
        {"extreme": 1.4, "very high": 1.2, "high": 1.0, "moderate": 0.55,
         "medium": 0.55, "low": 0.2, "none": 0.0},
        default=0.5, name="radiation_level",
    )
    magnetic = _map_levels(
        df_raw.get("magnetic_field", pd.Series(index=df_raw.index, dtype=object)),
        {"very strong": 1.3, "strong": 1.0, "moderate": 0.7, "weak": 0.3,
         "none": 0.0, "no": 0.0},
        default=0.5, name="magnetic_field",
    )
    seismic = _map_levels(
        df_raw.get("seismic_activity", pd.Series(index=df_raw.index, dtype=object)),
        {"extreme": 1.4, "high": 1.0, "moderate": 0.55, "medium": 0.55,
         "low": 0.2, "none": 0.0},
        default=0.5, name="seismic_activity",
    )
    storm = _map_levels(
        df_raw.get("storm_frequency", pd.Series(index=df_raw.index, dtype=object)),
        {"constant": 1.4, "frequent": 1.0, "high": 1.0, "seasonal": 0.6,
         "occasional": 0.5, "moderate": 0.5, "rare": 0.15, "low": 0.15,
         "none": 0.0},
        default=0.5, name="storm_frequency",
    )
    bio_threat = _map_levels(
        df_raw.get("native_biology", pd.Series(index=df_raw.index, dtype=object)),
        {"complex": 1.0, "invertebrate": 0.85, "multicellular": 0.8,
         "extremophile": 0.55, "chemosynth": 0.5, "microbial": 0.5,
         "unknown": 0.4, "none": 0.0, "no": 0.0},
        default=0.4, name="native_biology",
    )

    # H: two-sided Gaussian comfort zones + toxic gas damping
    h_temp = np.exp(-((temp - 15.0) / 18.0) ** 2)
    h_oxy = np.exp(-((oxy - 21.0) / 6.0) ** 2)
    h_grav = np.exp(-((grav - 1.0) / 0.45) ** 2)
    h_press = np.exp(-((press - 1.0) / 0.6) ** 2)
    h_day = np.exp(-((day - 24.0) / 14.0) ** 2)
    h_stab = np.exp(-(temp_range / 30.0))
    h_toxic = np.exp(-(co2 / 20.0)) * np.exp(-(ch4 / 15.0))
    H = 100.0 * (0.22 * h_temp + 0.20 * h_oxy + 0.18 * h_grav + 0.14 * h_press
                 + 0.08 * h_day + 0.10 * h_stab + 0.08 * h_toxic)

    # E: environment_type target encoded score (already on 0-100 scale)
    E = np.clip(_safe_num(env_type_score, default=float(np.nanmean(env_type_score))).values, 0.0, 100.0)

    # R: resources，裁剪到 [0,100]
    R = 100.0 * (0.45 * np.tanh(rare / 60.0) + 0.30 * energy + 0.25 * water)
    R = np.clip(R, 0.0, 100.0)

    # S: safety
    S = 100.0 * magnetic * np.exp(-0.6 * radiation) * np.exp(-0.5 * seismic) \
        * np.exp(-0.5 * storm) * np.exp(-0.4 * bio_threat)

    # Ec: economic (strategic / (1 + penalties))，裁剪到 [0,100] 与其他 comp_* 同量纲
    Ec = 100.0 * (strategic / 10.0) / (1.0 + 0.5 * (terra / 10.0) + 0.4 * (cost / 10.0))
    Ec = np.clip(Ec, 0.0, 100.0)

    # A: aesthetic
    A = 100.0 * np.exp(-((albedo - 0.35) / 0.2) ** 2) * np.exp(-((cloud - 40.0) / 30.0) ** 2)

    df["comp_H"] = H.astype(np.float32)
    df["comp_E"] = E.astype(np.float32)
    df["comp_R"] = R.astype(np.float32)
    df["comp_S"] = S.astype(np.float32)
    df["comp_Ec"] = Ec.astype(np.float32)
    df["comp_A"] = A.astype(np.float32)
    df["prior_blend"] = (0.40 * df["comp_H"] + 0.25 * df["comp_E"] + 0.15 * df["comp_R"]
                        + 0.10 * df["comp_S"] + 0.10 * df["comp_Ec"] + 0.05 * df["comp_A"]).astype(np.float32)

    # Interactions
    df["H_times_S"] = (df["comp_H"] * df["comp_S"] / 100.0).astype(np.float32)
    df["H_minus_E"] = (df["comp_H"] - df["comp_E"]).astype(np.float32)
    df["low_safety_flag"] = (df["comp_S"] < 30.0).astype(np.float32)
    df["high_hazard_flag"] = ((radiation >= 1.0) | (seismic >= 1.0) | (storm >= 1.0)).astype(np.float32)

    # Helpful legacy feature-engineering that remained predictive
    df["feat_oxygen_temp"] = (oxy * temp).astype(np.float32)
    df["feat_pressure_gravity_ratio"] = (press / (np.abs(grav) + 1e-3)).astype(np.float32)
    df["feat_magnetic_radiation_ratio"] = (magnetic / (1.0 + radiation)).astype(np.float32)
    df["feat_cost_strategy_ratio"] = (strategic / (1.0 + cost)).astype(np.float32)
    df["feat_terraform_resource_ratio"] = (np.tanh(rare / 60.0) / (1.0 + terra)).astype(np.float32)

    # Distribution logging for sanity check
    for col in ["comp_H", "comp_E", "comp_R", "comp_S", "comp_Ec", "comp_A", "prior_blend"]:
        v = df[col].values.astype(np.float64)
        logger.info(
            f"[TargetComp] {col:<12s} mean={v.mean():.3f} std={v.std():.3f} "
            f"min={v.min():.3f} max={v.max():.3f}"
        )
    return df


def orthogonalize_features(train_mod: np.ndarray, test_mod: np.ndarray, train_ref: np.ndarray, test_ref: np.ndarray, name: str, alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """从模态特征中减去可由表格/基线线性解释的部分，降低跨模态冗余。"""
    ref_scaler = StandardScaler()
    mod_scaler = StandardScaler()
    R_tr = ref_scaler.fit_transform(train_ref).astype(np.float32)
    R_te = ref_scaler.transform(test_ref).astype(np.float32)
    M_tr = mod_scaler.fit_transform(train_mod).astype(np.float32)
    M_te = mod_scaler.transform(test_mod).astype(np.float32)
    A = R_tr.T @ R_tr + alpha * np.eye(R_tr.shape[1], dtype=np.float32)
    B = R_tr.T @ M_tr
    W = np.linalg.solve(A, B).astype(np.float32)
    proj_tr = R_tr @ W
    proj_te = R_te @ W
    ortho_tr = (M_tr - proj_tr).astype(np.float32)
    ortho_te = (M_te - proj_te).astype(np.float32)
    shared = float(np.var(proj_tr) / max(np.var(M_tr), 1e-8))
    logger.info(f"{name} orthogonalization | shared_var_ratio={shared:.4f} | retained_var_ratio={1.0-shared:.4f}")
    return ortho_tr, ortho_te


def make_cosine_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.08):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.ff(self.norm(x)))

class AdapterRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int, bottleneck: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.08),
        )
        self.block1 = ResidualMLPBlock(hidden, dropout=0.08)
        self.block2 = ResidualMLPBlock(hidden, dropout=0.08)
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, bottleneck),
            nn.GELU(),
        )
        self.head = nn.Linear(bottleneck, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        feat = self.bottleneck(h)
        pred = self.head(feat).squeeze(-1)
        return pred, feat


def _train_adapter_fold(X_tr, y_tr, w_tr, X_va, y_va, X_te, device):
    scaler_np = StandardScaler()
    X_tr = scaler_np.fit_transform(X_tr).astype(np.float32)
    X_va = scaler_np.transform(X_va).astype(np.float32)
    X_te = scaler_np.transform(X_te).astype(np.float32)
    model = AdapterRegressor(X_tr.shape[1], CONFIG["ADAPTER_HIDDEN"], CONFIG["ADAPTER_BOTTLENECK"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["ADAPTER_LR"], weight_decay=CONFIG["ADAPTER_WEIGHT_DECAY"])
    amp_enabled = device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    tr_loader = DataLoader(TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
        torch.tensor(w_tr, dtype=torch.float32),
    ), batch_size=CONFIG["ADAPTER_BATCH_SIZE"], shuffle=True)
    va_loader = DataLoader(TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.float32)), batch_size=CONFIG["ADAPTER_BATCH_SIZE"], shuffle=False)
    total_steps = max(1, CONFIG["TextModel_epoch"] * len(tr_loader))
    scheduler = make_cosine_scheduler(opt, total_steps, CONFIG["ADAPTER_WARMUP_RATIO"])
    for _ in range(CONFIG["TextModel_epoch"]):
        model.train()
        for xb, yb, wb in tr_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred, feat = model(xb)
                aux = feat.pow(2).mean()
                loss = weighted_mse_tensor(pred, yb, wb) + CONFIG["ADAPTER_AUX_WEIGHT"] * aux
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()
    model.eval()
    with torch.no_grad():
        Xva_t = torch.tensor(X_va, dtype=torch.float32, device=device)
        Xte_t = torch.tensor(X_te, dtype=torch.float32, device=device)
        va_pred, va_feat = model(Xva_t)
        te_pred, te_feat = model(Xte_t)
    return va_pred.detach().float().cpu().numpy().astype(np.float32), te_pred.detach().float().cpu().numpy().astype(np.float32), va_feat.detach().float().cpu().numpy().astype(np.float32), te_feat.detach().float().cpu().numpy().astype(np.float32)


def train_adapter_cv(name, X, y, sample_weight, X_test, folds, device):
    oof_pred = np.zeros(len(X), dtype=np.float32)
    test_pred = np.zeros(len(X_test), dtype=np.float32)
    oof_feat = np.zeros((len(X), CONFIG["ADAPTER_BOTTLENECK"]), dtype=np.float32)
    test_feat = np.zeros((len(X_test), CONFIG["ADAPTER_BOTTLENECK"]), dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        start = time.time()
        va_pred, te_pred, va_feat, te_feat = _train_adapter_fold(X[tr_idx], y[tr_idx], sample_weight[tr_idx], X[va_idx], y[va_idx], X_test, device)
        oof_pred[va_idx] = va_pred
        oof_feat[va_idx] = va_feat
        test_pred += te_pred / len(folds)
        test_feat += te_feat / len(folds)
        log_stage(name, y[va_idx], va_pred, time.time() - start)
    return oof_pred, test_pred, oof_feat, test_feat


class TabMStyleRegressor(nn.Module):
    def __init__(self, n_num: int, cat_dims: List[int], embed_dim: int, hidden: int, k: int):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(dim, min(embed_dim, max(4, int(np.sqrt(dim)) + 1))) for dim in cat_dims])
        cat_out = sum(e.embedding_dim for e in self.embeds)
        in_dim = n_num + cat_out
        self.stem = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.08))
        self.block1 = ResidualMLPBlock(hidden, dropout=0.08)
        self.block2 = ResidualMLPBlock(hidden, dropout=0.08)
        self.experts = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)) for _ in range(k)])
        self.gate = nn.Linear(hidden, k)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor):
        cat_vecs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeds)] if x_cat.shape[1] > 0 else []
        x = torch.cat([x_num] + cat_vecs, dim=1) if cat_vecs else x_num
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        gate_logits = self.gate(h)
        gate = torch.softmax(gate_logits, dim=1)
        expert_outs = torch.cat([exp(h) for exp in self.experts], dim=1)
        pred = (gate * expert_outs).sum(dim=1)
        return pred, gate


def train_tabm(X_num, X_cat, y, sample_weight, X_num_test, X_cat_test, cat_dims, folds, device, seed: int = 42):
    oof = np.zeros(len(X_num), dtype=np.float32)
    test_pred = np.zeros(len(X_num_test), dtype=np.float32)
    amp_enabled = device.startswith("cuda")
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        start = time.time()
        scaler_num = StandardScaler()
        Xtr_num = scaler_num.fit_transform(X_num[tr_idx]).astype(np.float32)
        Xva_num = scaler_num.transform(X_num[va_idx]).astype(np.float32)
        Xte_num = scaler_num.transform(X_num_test).astype(np.float32)
        model = TabMStyleRegressor(X_num.shape[1], cat_dims, CONFIG["TABM_EMBED_DIM"], CONFIG["TABM_HIDDEN"], CONFIG["TABM_K"]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["TABM_LR"], weight_decay=CONFIG["TABM_WEIGHT_DECAY"])
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        tr_loader = DataLoader(TensorDataset(
            torch.tensor(Xtr_num, dtype=torch.float32),
            torch.tensor(X_cat[tr_idx], dtype=torch.long),
            torch.tensor(y[tr_idx], dtype=torch.float32),
            torch.tensor(sample_weight[tr_idx], dtype=torch.float32),
        ), batch_size=CONFIG["TABM_BATCH_SIZE"], shuffle=True)
        total_steps = max(1, CONFIG["TabM_epoch"] * len(tr_loader))
        scheduler = make_cosine_scheduler(opt, total_steps, CONFIG["TABM_WARMUP_RATIO"])
        for _ in range(CONFIG["TabM_epoch"]):
            model.train()
            for xb_num, xb_cat, yb, wb in tr_loader:
                xb_num, xb_cat, yb, wb = xb_num.to(device), xb_cat.to(device), yb.to(device), wb.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    pred, gate = model(xb_num, xb_cat)
                    gate_mean = gate.mean(dim=0)
                    aux = ((gate_mean - 1.0 / gate.shape[1]) ** 2).mean()
                    loss = weighted_mse_tensor(pred, yb, wb) + CONFIG["TABM_AUX_WEIGHT"] * aux
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                scheduler.step()
        model.eval()
        with torch.no_grad():
            va_pred, _ = model(torch.tensor(Xva_num, dtype=torch.float32, device=device), torch.tensor(X_cat[va_idx], dtype=torch.long, device=device))
            te_pred, _ = model(torch.tensor(Xte_num, dtype=torch.float32, device=device), torch.tensor(X_cat_test, dtype=torch.long, device=device))
        va_pred = va_pred.detach().float().cpu().numpy().astype(np.float32)
        te_pred = te_pred.detach().float().cpu().numpy().astype(np.float32)
        oof[va_idx] = va_pred
        test_pred += te_pred / len(folds)
        log_stage("TabM", y[va_idx], va_pred, time.time() - start)
    return oof, test_pred


def train_catboost(X_train_df, y, X_test_df, categorical_features, folds, seed: int = 42):
    oof = np.zeros(len(X_train_df), dtype=np.float32)
    test_pred = np.zeros(len(X_test_df), dtype=np.float32)
    cat_idx = [X_train_df.columns.get_loc(c) for c in categorical_features if c in X_train_df.columns]
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        start = time.time()
        model = CatBoostRegressor(
            iterations=CONFIG["CatBoost_epoch"],
            learning_rate=CONFIG["CATBOOST_LEARNING_RATE"],
            depth=CONFIG["CATBOOST_DEPTH"],
            l2_leaf_reg=CONFIG["CATBOOST_L2_LEAF_REG"],
            subsample=CONFIG["CATBOOST_SUBSAMPLE"],
            rsm=CONFIG["CATBOOST_RSM"],
            random_strength=CONFIG["CATBOOST_RANDOM_STRENGTH"],
            bagging_temperature=CONFIG["CATBOOST_BAGGING_TEMPERATURE"],
            loss_function="Lq:q=2",
            eval_metric="RMSE",
            random_state=seed + fold,
            verbose=CONFIG["CATBOOST_VERBOSE"],
        )
        with suppress_stdout_stderr():
            model.fit(X_train_df.iloc[tr_idx], y[tr_idx], cat_features=cat_idx)
        va_pred = model.predict(X_train_df.iloc[va_idx]).astype(np.float32)
        oof[va_idx] = va_pred
        test_pred += model.predict(X_test_df).astype(np.float32) / len(folds)
        log_stage("CatBoost", y[va_idx], va_pred, time.time() - start)
    return oof, test_pred


def train_lightgbm(X_train, y, X_test, folds, seed: int = 42):
    oof = np.zeros(len(X_train), dtype=np.float32)
    test_pred = np.zeros(len(X_test), dtype=np.float32)
    params = {
        "objective": CONFIG["LIGHTGBM_OBJECTIVE"], "alpha": CONFIG["LIGHTGBM_HUBER_ALPHA"],
        "metric": "l2", "boosting_type": "gbdt",
        "learning_rate": CONFIG["LIGHTGBM_LEARNING_RATE"], "num_leaves": CONFIG["LIGHTGBM_NUM_LEAVES"], "max_depth": CONFIG["LIGHTGBM_MAX_DEPTH"],
        "min_child_samples": CONFIG["LIGHTGBM_MIN_CHILD_SAMPLES"], "feature_fraction": CONFIG["LIGHTGBM_COLSAMPLE"],
        "bagging_fraction": CONFIG["LIGHTGBM_SUBSAMPLE"], "bagging_freq": 1, "lambda_l1": CONFIG["LIGHTGBM_L1"],
        "lambda_l2": CONFIG["LIGHTGBM_L2"], "verbosity": -1, "seed": seed, "num_threads": 0,
    }
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        start = time.time()
        dtr = lgb.Dataset(X_train.iloc[tr_idx], label=y[tr_idx])
        with suppress_stdout_stderr():
            model = lgb.train(params, dtr, num_boost_round=CONFIG["LightGBM_epoch"], callbacks=[lgb.log_evaluation(0)])
        va_pred = model.predict(X_train.iloc[va_idx]).astype(np.float32)
        oof[va_idx] = va_pred
        test_pred += model.predict(X_test).astype(np.float32) / len(folds)
        log_stage("LightGBM", y[va_idx], va_pred, time.time() - start)
    return oof, test_pred


def train_xgboost(X_train, y, X_test, folds, seed: int = 42):
    """XGBoost with standard squared-error loss. Provides a gradient-boosting
    variant distinct enough from CatBoost/LightGBM via different split rules
    and tree growth, while staying numerically stable."""
    oof = np.zeros(len(X_train), dtype=np.float32)
    test_pred = np.zeros(len(X_test), dtype=np.float32)
    dte = xgb.DMatrix(X_test)
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        start = time.time()
        dtr = xgb.DMatrix(X_train.iloc[tr_idx], label=y[tr_idx])
        dva = xgb.DMatrix(X_train.iloc[va_idx], label=y[va_idx])
        params = dict(
            objective="reg:squarederror",
            eta=CONFIG["XGBOOST_LEARNING_RATE"],
            max_depth=CONFIG["XGBOOST_MAX_DEPTH"],
            min_child_weight=CONFIG["XGBOOST_MIN_CHILD_WEIGHT"],
            subsample=CONFIG["XGBOOST_SUBSAMPLE"],
            colsample_bytree=CONFIG["XGBOOST_COLSAMPLE"],
            reg_alpha=CONFIG["XGBOOST_REG_ALPHA"],
            reg_lambda=CONFIG["XGBOOST_REG_LAMBDA"],
            tree_method="hist",
            eval_metric="rmse",
            seed=seed + fold,
            base_score=float(np.mean(y[tr_idx])),
            verbosity=0,
        )
        with suppress_stdout_stderr():
            model = xgb.train(
                params, dtr, num_boost_round=CONFIG["XGBoost_epoch"],
                verbose_eval=False,
            )
        va_pred = model.predict(dva).astype(np.float32)
        oof[va_idx] = va_pred
        test_pred += model.predict(dte).astype(np.float32) / len(folds)
        log_stage("XGBoost", y[va_idx], va_pred, time.time() - start)
    return oof, test_pred


def train_knn_components(comp_train: np.ndarray, y: np.ndarray, comp_test: np.ndarray,
                         folds: List, k: int = 25):
    """KNN over 6 subscores + a few key raw numerics.
       Under a synthetic-composite target KNN captures local consistency and
       typically produces residuals weakly correlated with tree models."""
    oof = np.zeros(len(y), dtype=np.float32)
    test_pred = np.zeros(len(comp_test), dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        start = time.time()
        sc = StandardScaler().fit(comp_train[tr_idx])
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
        knn.fit(sc.transform(comp_train[tr_idx]), y[tr_idx])
        va_pred = knn.predict(sc.transform(comp_train[va_idx])).astype(np.float32)
        oof[va_idx] = va_pred
        test_pred += knn.predict(sc.transform(comp_test)).astype(np.float32) / len(folds)
        log_stage("KNN", y[va_idx], va_pred, time.time() - start)
    return oof, test_pred


def convex_stack_solver(oof_matrix: np.ndarray, y: np.ndarray, test_matrix: np.ndarray):
    """Convex combination stacking: non-negative weights that sum to 1.
    Avoids the ±sign cancellation that RidgeCV/NNLS produce when base models
    are highly correlated."""
    n = oof_matrix.shape[1]
    def loss(w):
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if s <= 1e-8:
            return float("inf")
        w = w / s
        return float(np.mean((oof_matrix @ w - y) ** 2))
    w0 = np.ones(n, dtype=np.float64) / n
    res = minimize(
        loss, w0, method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"maxiter": CONFIG["STACK_CONVEX_MAXITER"], "ftol": 1e-9},
    )
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    w = w / s if s > 1e-8 else np.ones(n) / n
    pred_tr = (oof_matrix @ w).astype(np.float32)
    pred_te = (test_matrix @ w).astype(np.float32)
    return pred_tr, pred_te, w.astype(np.float32), 0.0


def quantile_align(pred_test: np.ndarray, pred_oof: np.ndarray, y_true: np.ndarray,
                   n_q: int = 200, clip_lo: float = 0.0, clip_hi: float = 100.0) -> np.ndarray:
    """Align pred_test's quantiles to the true-y distribution using pred_oof -> y mapping."""
    q = np.linspace(0.0, 1.0, n_q)
    pred_q = np.quantile(pred_oof, q)
    true_q = np.quantile(y_true, q)
    aligned = np.interp(pred_test, pred_q, true_q)
    return np.clip(aligned, clip_lo, clip_hi).astype(np.float32)


def fit_stack_solver(Xtr_s: np.ndarray, y: np.ndarray, Xte_s: np.ndarray, solver_name: str,
                     anchor_tr: Optional[np.ndarray] = None, anchor_te: Optional[np.ndarray] = None):
    if solver_name == "convex":
        pred_tr, pred_te, coef, _ = convex_stack_solver(Xtr_s, y, Xte_s)
        return {"pred_tr": pred_tr, "pred_te": pred_te, "coef": coef, "intercept": 0.0, "label": "ConvexStack"}
    if solver_name == "ridge":
        model = RidgeCV(alphas=np.logspace(-4, 3, CONFIG["STACK_RIDGE_ALPHAS_COUNT"]))
        model.fit(Xtr_s, y)
        pred_tr = model.predict(Xtr_s).astype(np.float32)
        pred_te = model.predict(Xte_s).astype(np.float32)
        coef = np.asarray(model.coef_, dtype=np.float32)
        intercept = float(model.intercept_)
        label = "RidgeCV"
    elif solver_name == "elasticnet":
        model = ElasticNetCV(l1_ratio=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8], alphas=np.logspace(-4, 1, 24), max_iter=CONFIG["STACK_ELASTICNET_MAXITER"], random_state=CONFIG["RANDOM_STATE"])
        model.fit(Xtr_s, y)
        pred_tr = model.predict(Xtr_s).astype(np.float32)
        pred_te = model.predict(Xte_s).astype(np.float32)
        coef = np.asarray(model.coef_, dtype=np.float32)
        intercept = float(model.intercept_)
        label = "ElasticNetCV"
    elif solver_name == "nnls" and nnls is not None:
        coef, _ = nnls(Xtr_s, y.astype(np.float64))
        coef = coef.astype(np.float32)
        pred_raw = Xtr_s @ coef
        intercept = float(y.mean() - pred_raw.mean())
        pred_tr = (pred_raw + intercept).astype(np.float32)
        pred_te = (Xte_s @ coef + intercept).astype(np.float32)
        label = "NNLS"
    elif solver_name == "anchored_nnls" and nnls is not None and anchor_tr is not None and anchor_te is not None:
        target = (y - anchor_tr).astype(np.float64)
        coef, _ = nnls(Xtr_s, target)
        coef = coef.astype(np.float32)
        pred_delta_tr = Xtr_s @ coef
        intercept = float(target.mean() - pred_delta_tr.mean())
        pred_tr = (anchor_tr + pred_delta_tr + intercept).astype(np.float32)
        pred_te = (anchor_te + Xte_s @ coef + intercept).astype(np.float32)
        label = "AnchoredNNLS"
    else:
        return None
    return {"pred_tr": pred_tr, "pred_te": pred_te, "coef": coef, "intercept": intercept, "label": label}


def residual_diversity_score(y: np.ndarray, pred_matrix: np.ndarray) -> float:
    resid = y[:, None] - pred_matrix
    corr = np.corrcoef(resid.T)
    iu = np.triu_indices_from(corr, k=1)
    return float(np.nanmean(np.abs(corr[iu]))) if len(iu[0]) else 0.0


def log_hard_sample_diagnostics(y: np.ndarray, pred_dict: Dict[str, np.ndarray], ref_pred: np.ndarray, top_frac: float = 0.2):
    err = np.abs(y - ref_pred)
    k = max(1, int(len(y) * top_frac))
    hard_idx = np.argsort(-err)[:k]
    easy_idx = np.argsort(err)[: max(1, len(y) - k)]
    logger.info(f"HARD SAMPLE DIAGNOSTICS | reference=CatBoost | top_frac={top_frac:.2f} | hard_n={len(hard_idx)}")
    for name, pred in pred_dict.items():
        hard_rmse = rmse(y[hard_idx], pred[hard_idx])
        easy_rmse = rmse(y[easy_idx], pred[easy_idx])
        logger.info(f"[{name:<16s}] hard_rmse={hard_rmse:.4f} | easy_rmse={easy_rmse:.4f} | hard_gain_vs_cat={rmse(y[hard_idx], ref_pred[hard_idx]) - hard_rmse:.4f}")


def log_residual_structure(y: np.ndarray, pred_dict: Dict[str, np.ndarray]):
    names = list(pred_dict.keys())
    resid = np.vstack([y - pred_dict[n] for n in names])
    corr = np.corrcoef(resid)
    logger.info("RESIDUAL CORRELATION MATRIX (rows/cols follow " + ", ".join(names) + ")")
    logger.info(np.array2string(corr, precision=4, suppress_small=False))


def train_stacking_subset_search(base_oof: np.ndarray, y: np.ndarray, base_test_preds: np.ndarray, model_names: List[str]):
    idx = {n: i for i, n in enumerate(model_names)}
    ref_idx = idx.get("CatBoost", 0)
    ref_pred = base_oof[:, ref_idx]
    err = np.abs(y - ref_pred)
    k = max(1, int(len(y) * 0.20))
    hard_idx = np.argsort(-err)[:k]
    ref_hard_rmse = rmse(y[hard_idx], ref_pred[hard_idx])
    global_gains = {name: rmse(y, ref_pred) - rmse(y, base_oof[:, i]) for i, name in enumerate(model_names)}
    hard_gains = {name: ref_hard_rmse - rmse(y[hard_idx], base_oof[:, i][hard_idx]) for i, name in enumerate(model_names)}

    force = [idx[n] for n in CONFIG["STACK_FORCE_INCLUDE"] if n in idx]
    optional = []
    for i, n in enumerate(model_names):
        if i in force:
            continue
        if global_gains.get(n, -999) >= CONFIG["STACK_GLOBAL_GAIN_FLOOR"] or hard_gains.get(n, -999) > CONFIG["STACK_HARD_GAIN_FLOOR"]:
            optional.append(i)
    best = None
    logger.info("STACKING SUBSET SEARCH")
    for r in range(max(0, CONFIG["STACK_MIN_MODELS"] - len(force)), min(len(optional), CONFIG["STACK_MAX_MODELS"] - len(force)) + 1):
        for comb in itertools.combinations(optional, r):
            sel = sorted(force + list(comb))
            names = [model_names[i] for i in sel]
            Xtr = base_oof[:, sel]
            Xte = base_test_preds[:, sel]
            subset_corr_penalty = residual_diversity_score(y, Xtr) * CONFIG["STACK_CORR_PENALTY"]
            neg_hard_penalty = 0.0
            for n in names:
                if n not in CONFIG["STACK_FORCE_INCLUDE"] and hard_gains.get(n, 0.0) < 0:
                    neg_hard_penalty += (-hard_gains[n]) * CONFIG["STACK_HARD_GAIN_PENALTY"]
            for solver_name in CONFIG["STACK_SOLVERS"]:
                if solver_name == "convex":
                    fit = fit_stack_solver(Xtr, y, Xte, solver_name)
                    coef_map = np.zeros(len(model_names), dtype=np.float32)
                    if fit is not None:
                        coef_map[np.array(sel)] = fit["coef"].astype(np.float32)
                elif solver_name == "anchored_nnls":
                    other_local = [j for j, i in enumerate(sel) if model_names[i] != "CatBoost"]
                    if len(other_local) == 0:
                        fit = {"pred_tr": ref_pred.astype(np.float32), "pred_te": base_test_preds[:, ref_idx].astype(np.float32), "coef": np.zeros(0, dtype=np.float32), "intercept": 0.0, "label": "AnchoredNNLS"}
                    else:
                        Dtr = Xtr[:, other_local] - ref_pred[:, None]
                        Dte = Xte[:, other_local] - base_test_preds[:, ref_idx][:, None]
                        scaler = StandardScaler()
                        Dtr_s = scaler.fit_transform(Dtr)
                        Dte_s = scaler.transform(Dte)
                        fit = fit_stack_solver(Dtr_s, y, Dte_s, solver_name, anchor_tr=ref_pred.astype(np.float32), anchor_te=base_test_preds[:, ref_idx].astype(np.float32))
                    coef_map = np.zeros(len(model_names), dtype=np.float32)
                    if fit is not None and len(other_local) > 0:
                        for pos, c in zip(other_local, fit["coef"]):
                            coef_map[sel[pos]] = c
                        coef_map[ref_idx] = 1.0
                else:
                    scaler = StandardScaler()
                    Xtr_s = scaler.fit_transform(Xtr)
                    Xte_s = scaler.transform(Xte)
                    fit = fit_stack_solver(Xtr_s, y, Xte_s, solver_name)
                    coef_map = np.zeros(len(model_names), dtype=np.float32)
                    if fit is not None:
                        coef_map[np.array(sel)] = fit["coef"].astype(np.float32)
                if fit is None:
                    continue
                raw_mse = mse(y, fit["pred_tr"])
                cur_obj = raw_mse + subset_corr_penalty + neg_hard_penalty
                logger.info(f"[StackSubset] solver={fit['label']:<12s} | models={names} | mse={raw_mse:.6f} | corr_penalty={subset_corr_penalty:.6f} | hard_penalty={neg_hard_penalty:.6f} | objective={cur_obj:.6f}")
                if best is None or cur_obj < best[0]:
                    best = (cur_obj, fit, names, raw_mse, subset_corr_penalty, neg_hard_penalty, coef_map)
    _, fit, names, raw_mse, cp, hp, coef_full = best
    logger.info(f"[StackBest] solver={fit['label']} | models={names} | raw_mse={raw_mse:.6f} | corr_penalty={cp:.6f} | hard_penalty={hp:.6f}")
    return fit["pred_tr"], fit["pred_te"], coef_full, names, fit["label"] + "+SubsetSearch", hard_gains, global_gains



def main(args):
    logger.info(f"[CONFIG] device={args.device} | folds={args.n_folds} | seed={CONFIG['RANDOM_STATE']}")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    train_df, test_df = read_data(args.data_dir)
    logger.info(f"Loaded train={train_df.shape} test={test_df.shape}")
    numeric_cols, categorical_cols, text_cols = identify_columns(train_df)

    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    y = train_df["settlement_index"].values.astype(np.float32)
    folds = make_folds(y, args.n_folds, CONFIG["RANDOM_STATE"])

    # K-fold target encoding for categorical columns (leak-free)
    te_cols = [c for c in ["environment_type", "atmosphere_primary_gases", "viability_class", "dominant_biome"]
               if c in train_df.columns]
    te_train: Dict[str, np.ndarray] = {}
    te_test: Dict[str, np.ndarray] = {}
    for col in te_cols:
        tr_enc, te_enc = kfold_target_encode(
            train_df[col], y, test_df[col], folds, smoothing=CONFIG["TE_SMOOTHING"]
        )
        te_train[col] = tr_enc
        te_test[col] = te_enc
        logger.info(
            f"[TargetEncode] {col:<32s} train mean={tr_enc.mean():.3f} std={tr_enc.std():.3f} | "
            f"test mean={te_enc.mean():.3f} std={te_enc.std():.3f}"
        )

    # Environment-type score (0-100) feeds into engineer_target_components comp_E
    if "environment_type" in te_cols:
        env_score_full = np.concatenate([te_train["environment_type"], te_test["environment_type"]])
    else:
        env_score_full = np.full(len(full_df), float(y.mean()), dtype=np.float32)
    env_score_series = pd.Series(env_score_full, index=full_df.index)

    # Build target-formula feature components (6 sub-scores + prior + interactions)
    target_comp_df = engineer_target_components(full_df, env_score_series)

    full_proc, encoders = preprocess_tabular(full_df, numeric_cols, categorical_cols, True)
    full_proc = pd.concat([full_proc.reset_index(drop=True), target_comp_df.reset_index(drop=True)], axis=1)
    train_proc = full_proc.iloc[:len(train_df)].copy().reset_index(drop=True)
    test_proc = full_proc.iloc[len(train_df):].copy().reset_index(drop=True)

    # Splice target-encoded numeric features into train_proc / test_proc
    for col in te_cols:
        train_proc[f"te_{col}"] = te_train[col]
        test_proc[f"te_{col}"] = te_test[col]

    train_texts = concatenate_text_fields(train_df, text_cols)
    test_texts = concatenate_text_fields(test_df, text_cols)
    text_emb_train = get_hf_text_embeddings(train_texts, args.text_model, args.batch_size, args.device, CONFIG["TEXT_MAX_LENGTH"])
    text_emb_test = get_hf_text_embeddings(test_texts, args.text_model, args.batch_size, args.device, CONFIG["TEXT_MAX_LENGTH"])
    text_low_train, text_low_test = fit_pca_features(text_emb_train, text_emb_test, CONFIG["TEXT_PCA_DIM"], "Text")

    drop_cols = [c for c in text_cols + ["settlement_index", "id"] if c in train_proc.columns]
    train_tab_df = train_proc.drop(columns=drop_cols).copy()
    test_tab_df = test_proc.drop(columns=drop_cols).copy()
    cat_cols = [c for c in categorical_cols if c in train_tab_df.columns]
    num_cols = [c for c in train_tab_df.columns if c not in cat_cols]

    # Keyword-based sentiment features from raw text fields (Phase 4)
    kw_train = text_keyword_features(train_texts)
    kw_test = text_keyword_features(test_texts)
    kw_cols = ["kw_neg", "kw_pos", "kw_net", "kw_net_ratio", "kw_len"]
    for i, c in enumerate(kw_cols):
        train_tab_df[c] = kw_train[:, i]
        test_tab_df[c] = kw_test[:, i]
    num_cols = [c for c in train_tab_df.columns if c not in cat_cols]
    logger.info(f"Keyword features shape train={kw_train.shape} test={kw_test.shape}")

    seed_base = CONFIG["RANDOM_STATE"]

    # --- CatBoost (single-seed, fixed iterations, no early stop) ---
    cat_oof, cat_test = train_catboost(
        train_tab_df, y, test_tab_df, cat_cols, folds, seed=seed_base,
    )
    log_prediction_diagnostics("CatBoost", y, cat_oof)

    hard_weights = compute_hard_weights(y - cat_oof) if CONFIG["USE_HARD_SAMPLE_WEIGHT"] else np.ones_like(y, dtype=np.float32)

    # --- TextModel (independent; predicts y directly; extra keyword features concat to PCA input) ---
    text_in_train = np.hstack([text_low_train, kw_train]).astype(np.float32)
    text_in_test = np.hstack([text_low_test, kw_test]).astype(np.float32)
    text_oof, text_test, text_adapt_train, text_adapt_test = train_adapter_cv(
        "TextModel", text_in_train, y.astype(np.float32), hard_weights, text_in_test, folds, args.device
    )

    # --- LightGBM (single-seed, uses text-adapter bottleneck features) ---
    X_lgb = pd.concat([train_tab_df.reset_index(drop=True),
                       pd.DataFrame(text_adapt_train, columns=[f"txt_adapt_{i}" for i in range(text_adapt_train.shape[1])])], axis=1)
    X_lgb_test = pd.concat([test_tab_df.reset_index(drop=True),
                            pd.DataFrame(text_adapt_test, columns=[f"txt_adapt_{i}" for i in range(text_adapt_test.shape[1])])], axis=1)
    lgb_oof, lgb_test = train_lightgbm(X_lgb, y, X_lgb_test, folds, seed=seed_base)

    # --- XGBoost (single-seed, squared-error) ---
    xgb_oof, xgb_test = train_xgboost(X_lgb, y, X_lgb_test, folds, seed=seed_base)

    # --- TabM (single-seed, independent — predicts y directly) ---
    tabm_num_cols = [c for c in num_cols if c not in set(CONFIG["TABM_DROP_NUM_COLS"])]
    text_aux_dim = min(CONFIG["TABM_TEXT_AUX_DIM"], text_low_train.shape[1])
    X_num = np.hstack([
        train_tab_df[tabm_num_cols].values.astype(np.float32),
        text_low_train[:, :text_aux_dim].astype(np.float32),
    ]).astype(np.float32)
    X_num_test = np.hstack([
        test_tab_df[tabm_num_cols].values.astype(np.float32),
        text_low_test[:, :text_aux_dim].astype(np.float32),
    ]).astype(np.float32)
    X_cat = train_tab_df[cat_cols].values.astype(np.int64) if cat_cols else np.zeros((len(train_tab_df), 0), dtype=np.int64)
    X_cat_test = test_tab_df[cat_cols].values.astype(np.int64) if cat_cols else np.zeros((len(test_tab_df), 0), dtype=np.int64)
    cat_dims = [len(encoders[c].classes_) for c in cat_cols]
    tabm_oof, tabm_test = train_tabm(
        X_num, X_cat, y, hard_weights, X_num_test, X_cat_test, cat_dims,
        folds, args.device, seed=seed_base,
    )

    # --- KNN on 6-subscore space + key raw numerics ---
    knn_cols = [c for c in ["comp_H", "comp_E", "comp_R", "comp_S", "comp_Ec", "comp_A",
                            "prior_blend", "mean_temp_c", "oxygen_percent", "gravity_g", "rare_mineral_index"]
                if c in train_tab_df.columns]
    knn_train_mat = train_tab_df[knn_cols].values.astype(np.float32)
    knn_test_mat = test_tab_df[knn_cols].values.astype(np.float32)
    knn_oof, knn_test = train_knn_components(knn_train_mat, y, knn_test_mat, folds, k=CONFIG["KNN_K"])

    base_oof = np.vstack([cat_oof, lgb_oof, xgb_oof, tabm_oof, text_oof, knn_oof]).T.astype(np.float32)
    base_test = np.vstack([cat_test, lgb_test, xgb_test, tabm_test, text_test, knn_test]).T.astype(np.float32)
    model_names = ["CatBoost", "LightGBM", "XGBoost", "TabM", "TextModel", "KNN"]
    stack_oof, stack_test, stack_coef, selected_models, stack_solver, stack_hard_gains, stack_global_gains = train_stacking_subset_search(base_oof, y, base_test, model_names)

    # --- Quantile alignment on final test predictions ---
    stack_oof_rmse_before = rmse(y, stack_oof)
    stack_oof_aligned = quantile_align(stack_oof, stack_oof, y)
    stack_oof_rmse_after = rmse(y, stack_oof_aligned)
    stack_test = quantile_align(stack_test, stack_oof, y)
    logger.info(f"[QuantileAlign] stack_oof RMSE before={stack_oof_rmse_before:.4f} | after_self_align={stack_oof_rmse_after:.4f}")

    logger.info("")
    logger.info("MODEL / MODALITY DIAGNOSTICS")
    for name, pred in zip(model_names + ["StackingFinal"], [cat_oof, lgb_oof, xgb_oof, tabm_oof, text_oof, knn_oof, stack_oof]):
        log_prediction_diagnostics(name, y, pred)
    logger.info(f"Text PCA explained dim={CONFIG['TEXT_PCA_DIM']} | approx_signal_ratio_check=see explained variance above")
    logger.info(f"Tabular feature dim           = {train_tab_df.shape[1]}")
    logger.info(f"LGB augmented feature dim     = {X_lgb.shape[1]}")
    logger.info(f"TabM numeric dim             = {X_num.shape[1]}")
    logger.info(f"TabM categorical cols        = {len(cat_cols)}")
    pred_dict = {
        "CatBoost": cat_oof,
        "LightGBM": lgb_oof,
        "XGBoost": xgb_oof,
        "TabM": tabm_oof,
        "TextModel": text_oof,
        "KNN": knn_oof,
        "StackingFinal": stack_oof,
    }
    log_hard_sample_diagnostics(y, pred_dict, cat_oof, top_frac=0.2)
    logger.info("STACK HARD-GAIN SUMMARY (vs CatBoost on hard samples)")
    for n, g in stack_hard_gains.items():
        logger.info(f"{n:<18s} hard_gain_vs_cat={g:.4f} | global_gain_vs_cat={stack_global_gains.get(n, 0.0):.4f}")
    log_residual_structure(y, {k: pred_dict[k] for k in ["CatBoost", "LightGBM", "XGBoost", "TabM", "TextModel", "KNN"]})
    try:
        full_cat_model = CatBoostRegressor(
            iterations=CONFIG["CatBoost_epoch"], learning_rate=CONFIG["CATBOOST_LEARNING_RATE"], depth=CONFIG["CATBOOST_DEPTH"],
            l2_leaf_reg=CONFIG["CATBOOST_L2_LEAF_REG"], subsample=CONFIG["CATBOOST_SUBSAMPLE"], rsm=CONFIG["CATBOOST_RSM"],
            random_strength=CONFIG["CATBOOST_RANDOM_STRENGTH"], bagging_temperature=CONFIG["CATBOOST_BAGGING_TEMPERATURE"],
            loss_function="RMSE", eval_metric="RMSE", verbose=False, random_seed=CONFIG["RANDOM_STATE"]
        )
        full_cat_model.fit(train_tab_df, y, cat_features=cat_cols, verbose=False)
        imp = full_cat_model.get_feature_importance()
        top_idx = np.argsort(-imp)[: CONFIG["DIAG_TOPK_FEATURES"]]
        logger.info("TOP CATBOOST FEATURE IMPORTANCE")
        for j in top_idx:
            logger.info(f"{train_tab_df.columns[j]:<28s} importance={float(imp[j]):.6f}")
    except Exception as e:
        logger.info(f"TOP CATBOOST FEATURE IMPORTANCE skipped due to: {e}")

    logger.info("")
    logger.info("FINAL RESULTS")
    for name, pred in zip(model_names + ["StackingFinal"], [cat_oof, lgb_oof, xgb_oof, tabm_oof, text_oof, knn_oof, stack_oof]):
        logger.info(f"{name:<18s} {rmse(y, pred):.4f}")
    logger.info("")
    logger.info("MODALITY / REGRESSOR MSE REPORT")
    base_rmse = rmse(y, cat_oof)
    for name, pred in zip(model_names, [cat_oof, lgb_oof, xgb_oof, tabm_oof, text_oof, knn_oof]):
        logger.info(f"{name:<18s} MSE={mse(y, pred):.6f} | RMSE={rmse(y, pred):.4f} | gain_vs_cat={base_rmse - rmse(y, pred):.4f}")
    logger.info(f"StackingFinal       MSE={mse(y, stack_oof):.6f} | RMSE={rmse(y, stack_oof):.4f} | gain_vs_cat={base_rmse - rmse(y, stack_oof):.4f}")
    logger.info(f"STACKING CONTRIBUTION REPORT | solver={stack_solver}")
    if "AnchoredNNLS" in stack_solver:
        logger.info("CatBoost is used as fixed anchor baseline in AnchoredNNLS; other coefficients are residual-correction weights.")
    logger.info(f"Selected stack models = {selected_models}")
    for name, coef in zip(model_names, stack_coef):
        logger.info(f"{name:<18s} coef={coef: .6f}")
    corr = np.corrcoef(base_oof.T)
    logger.info("BASE MODEL CORRELATION (rows/cols follow " + ", ".join(model_names) + ")")
    logger.info(np.array2string(corr, precision=4, suppress_small=False))
    submission = pd.DataFrame({"id": test_df["id"], "settlement_index": stack_test})
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "submission.csv")
    submission.to_csv(out_path, index=False)
    logger.info(f"Submission saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settlement index stacking: target-formula features + 6 independent base models + convex stacking + quantile alignment")
    parser.add_argument("--data_dir", type=str, default=CONFIG["DATA_DIR"])
    parser.add_argument("--output_dir", type=str, default=CONFIG["OUTPUT_DIR"])
    parser.add_argument("--n_folds", type=int, default=CONFIG["N_FOLDS"])
    parser.add_argument("--text_model", type=str, default=CONFIG["TEXT_MODEL"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["TEXT_BATCH_SIZE"])
    parser.add_argument("--device", type=str, default=CONFIG["DEVICE"])
    args = parser.parse_args()
    main(args)
