"""
Main training script for the Exoplanet Settlement Viability Prediction task.

This script implements a multi‑modal, multi‑model training pipeline that
simultaneously leverages structured tabular features and rich unstructured
textual data.  While the original challenge description also referenced
imagery associated with each planet, the provided dataset does not include
any JPEG files; instead it contains an `image_prompt` field describing the
visual appearance of the planet.  Because of this, image processing is
limited to embedding the descriptive prompt text rather than pixels.  The
architecture consists of several base learners trained in an out‑of‑fold
fashion followed by a simple stacking regressor that learns how best to
combine their predictions.

Outline of the pipeline:

1. **Data loading and preprocessing**
   - Read training and test CSV files.
   - Identify numeric and categorical columns; cast types and fill missing
     values accordingly.
   - Concatenate several free‑text fields into a single string per example.
   - Compute dense vector embeddings for the text using a pretrained
     Transformer model from the `sentence_transformers` library.  The
     default model, `all-MiniLM-L6-v2`, returns 384‑dimensional sentence
     embeddings.
   - For TabNet, convert categorical columns to integer indices and
     collect the cardinality of each categorical feature.

2. **Base model training with out‑of‑fold predictions**
   - A `CatBoostRegressor` trained solely on the tabular features.  CatBoost
     natively handles categorical inputs and missing values, making it well
     suited for the heterogeneous data at hand.
   - A `TabNetRegressor` from the `pytorch_tabnet` package trained on
     tabular features concatenated with the text embeddings.  TabNet
     combines gradient boosting ideas with neural attention mechanisms and
     can learn useful representations for high‑dimensional numeric input.
   - A gradient boosting model (`LightGBMRegressor`) trained on the same
     feature set as TabNet.  LightGBM is included as an additional strong
     learner whose performance often complements CatBoost and TabNet.

   Each base model is trained inside a K‑fold cross‑validation loop (default
   5 folds).  Out‑of‑fold (OOF) predictions on the training set and
   averaged predictions on the test set are collected for each model.

3. **Stacking meta‑model**
   - A simple linear model (Ridge regression by default) is fit on the
     aggregated OOF predictions to learn optimal blending weights.  The
     fitted meta‑model is then used to combine the base model outputs on
     the test set.

Results produced by this script include OOF predictions for each base
learner, the fitted meta‑model and its predictions on the test set, and a
`submission.csv` file suitable for upload to the competition platform.  The
code is intended for execution in a Jupyter/Colab environment with
internet access for model downloads.

Usage:

```
!pip install catboost lightgbm pytorch-tabnet sentence-transformers
python solution.py --data_dir /content/dataset --n_folds 5 --device cuda
```

"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.exceptions import NotFittedError

# Base models
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
import lightgbm as lgb

# Text embedding
from sentence_transformers import SentenceTransformer


def read_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read training and test data from a directory.

    The dataset directory is expected to contain `train.csv` and `test.csv`.

    Parameters
    ----------
    data_dir : str
        Path to the directory with the CSV files.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The loaded training and test dataframes.
    """
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Expected train.csv and test.csv in {data_dir}, found {os.listdir(data_dir)}"
        )
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def identify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Determine which columns are numeric, categorical, and textual.

    Numeric columns are those with a numeric dtype or that can be coerced to
    numeric without introducing NaNs.  Textual columns are pre‑defined
    narrative fields and any additional descriptive columns.  Categorical
    columns are all remaining non‑numeric, non‑text fields.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe from which to derive column types.

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        Lists of numeric columns, categorical columns and text columns.
    """
    # Predefined textual columns present in the dataset
    text_cols = [
        "exploration_log",
        "environmental_report",
        "incident_report",
        "image_prompt",
        "description",
    ]

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    # Identify numeric columns by attempting to convert to numeric
    for col in df.columns:
        if col in text_cols or col in ["settlement_index", "id"]:
            continue
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            numeric_cols.append(col)
        else:
            # Try to coerce to numeric; if conversion yields many NaNs, treat as categorical
            try:
                coerced = pd.to_numeric(df[col], errors="coerce")
                num_na = coerced.isna().sum()
                if num_na < 0.3 * len(df):
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            except Exception:
                categorical_cols.append(col)
    return numeric_cols, categorical_cols, text_cols


def preprocess_tabular(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    is_train: bool = True,
    label_encoders: dict = None,
) -> Tuple[pd.DataFrame, dict]:
    """Preprocess numeric and categorical columns.

    Numeric columns have missing values filled with the median.  Categorical
    columns are optionally encoded to integer labels when training TabNet
    models.  The function returns the processed dataframe and the mapping
    of column names to fitted `LabelEncoder` objects for reuse on test data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing both training and test samples.
    numeric_cols : List[str]
        Names of numeric feature columns.
    categorical_cols : List[str]
        Names of categorical feature columns.
    is_train : bool, default True
        Whether the dataframe is the training set.  When False, the
        existing label encoders will be applied instead of fitting new ones.
    label_encoders : dict, default None
        Previously fitted label encoders for categorical columns.  Only used
        when `is_train` is False.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        The processed dataframe and a dictionary mapping column names to
        `LabelEncoder` instances.
    """
    df_proc = df.copy()
    encoders = {} if label_encoders is None else label_encoders

    # Fill numeric missing values with median
    for col in numeric_cols:
        if df_proc[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
            # Convert to numeric where possible
            df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce")
        median_val = df_proc[col].median()
        df_proc[col] = df_proc[col].fillna(median_val)

    # Encode categorical columns
    for col in categorical_cols:
        if is_train:
            le = LabelEncoder()
            df_proc[col] = df_proc[col].astype(str).fillna("unknown")
            df_proc[col] = le.fit_transform(df_proc[col])
            encoders[col] = le
        else:
            if col not in encoders:
                raise NotFittedError(f"LabelEncoder for column '{col}' has not been fitted.")
            le = encoders[col]
            df_proc[col] = df_proc[col].astype(str).fillna("unknown")
            # When unseen labels appear in test data, map them to a new value
            df_proc[col] = df_proc[col].apply(lambda x: x if x in le.classes_ else "<UNK>")
            if "<UNK>" not in le.classes_:
                # Extend classes to include unknown token
                le.classes_ = np.append(le.classes_, "<UNK>")
            df_proc[col] = le.transform(df_proc[col])
    return df_proc, encoders


def concatenate_text_fields(df: pd.DataFrame, text_cols: List[str]) -> List[str]:
    """Concatenate multiple text fields into a single string per row.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing text columns.
    text_cols : List[str]
        List of column names containing text.

    Returns
    -------
    List[str]
        Concatenated text for each row.
    """
    # Replace missing values with empty strings and join with separators
    texts = (
        df[text_cols]
        .fillna("")
        .apply(lambda row: " \n ".join(row.values.astype(str)), axis=1)
        .tolist()
    )
    return texts


def compute_text_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64, device: str = "cpu") -> np.ndarray:
    """Compute sentence embeddings for a list of texts.

    This function wraps the `SentenceTransformer` model from the
    `sentence_transformers` library and applies it in batches to avoid
    excessive memory consumption.

    Parameters
    ----------
    texts : List[str]
        List of documents to embed.
    model_name : str, default 'all-MiniLM-L6-v2'
        Identifier of the pretrained sentence transformer model to load.
    batch_size : int, default 64
        Number of documents to process per batch.
    device : str, default 'cpu'
        Device on which to run the model ('cpu' or 'cuda').

    Returns
    -------
    np.ndarray
        Matrix of shape (len(texts), embedding_dim) containing sentence
        embeddings for each input document.
    """
    model = SentenceTransformer(model_name, device=device)
    embeddings_list = []
    # Process texts in batches
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        emb = model.encode(
            batch_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=batch_size,
        )
        embeddings_list.append(emb)
    embeddings = np.vstack(embeddings_list)
    return embeddings


def train_catboost(
    X: pd.DataFrame,
    y: np.ndarray,
    categorical_features: List[str],
    X_test: pd.DataFrame,
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train a CatBoostRegressor with out‑of‑fold predictions.

    Parameters
    ----------
    X : pd.DataFrame
        Training feature matrix.
    y : np.ndarray
        Target array.
    categorical_features : List[str]
        Names of categorical columns in `X`.
    X_test : pd.DataFrame
        Test feature matrix.
    n_folds : int, default 5
        Number of cross‑validation folds.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Out‑of‑fold predictions for the training set and averaged predictions
        for the test set.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    cat_features_indices = [X.columns.get_loc(col) for col in categorical_features]

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.03,
            depth=8,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_state=random_state + fold,
            verbose=200,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            cat_features=cat_features_indices,
            use_best_model=True,
            verbose=200,
        )
        oof_preds[valid_idx] = model.predict(X_valid)
        test_preds += model.predict(X_test) / n_folds
    return oof_preds, test_preds


def train_tabnet(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    categorical_dims: List[int],
    cat_idxs: List[int],
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train a TabNetRegressor with out‑of‑fold predictions.

    Parameters
    ----------
    X : np.ndarray
        Training feature matrix.  Categorical columns should be integer
        encoded and placed at the indices specified by `cat_idxs`.
    y : np.ndarray
        Target array.
    X_test : np.ndarray
        Test feature matrix with the same structure as `X`.
    categorical_dims : List[int]
        Cardinality (number of unique values) for each categorical feature.
    cat_idxs : List[int]
        Positions of categorical columns within `X`.
    n_folds : int, default 5
        Number of cross‑validation folds.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Out‑of‑fold predictions for the training set and averaged predictions
        for the test set.
    """
    oof_preds = np.zeros(X.shape[0])
    test_preds = np.zeros(X_test.shape[0])
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model = TabNetRegressor(
            cat_idxs=cat_idxs,
            cat_dims=categorical_dims,
            n_d=16,
            n_a=16,
            n_steps=5,
            gamma=1.3,
            lambda_sparse=1e-5,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-3),
            mask_type='sparsemax',
            n_shared=1,
            n_independent=1,
            seed=random_state + fold,
        )
        model.fit(
            X_train=X_train,
            y_train=y_train.reshape(-1, 1),
            eval_set=[(X_valid, y_valid.reshape(-1, 1))],
            max_epochs=200,
            patience=30,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
        )
        oof_preds[valid_idx] = model.predict(X_valid).squeeze()
        test_preds += model.predict(X_test).squeeze() / n_folds
    return oof_preds, test_preds


def train_lightgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train a LightGBMRegressor with out‑of‑fold predictions.

    Parameters
    ----------
    X : pd.DataFrame
        Training feature matrix.
    y : np.ndarray
        Target array.
    X_test : pd.DataFrame
        Test feature matrix.
    n_folds : int, default 5
        Number of cross‑validation folds.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Out‑of‑fold predictions for the training set and averaged predictions
        for the test set.
    """
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.7,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'seed': random_state,
        'verbosity': -1,
    }

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_valid = lgb.Dataset(X_valid, label=y_valid)

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=10000,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=['train', 'valid'],
            early_stopping_rounds=300,
            verbose_eval=500,
        )
        oof_preds[valid_idx] = model.predict(X_valid, num_iteration=model.best_iteration)
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_folds
    return oof_preds, test_preds


def train_stacking_model(
    base_oof: np.ndarray,
    y: np.ndarray,
    base_test_preds: np.ndarray,
    model_type: str = 'ridge',
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train a simple stacking model on OOF predictions.

    Parameters
    ----------
    base_oof : np.ndarray
        Matrix of shape (n_samples, n_models) containing OOF predictions
        from the base learners.
    y : np.ndarray
        Target array.
    base_test_preds : np.ndarray
        Matrix of shape (n_test_samples, n_models) containing test set
        predictions from the base learners.
    model_type : str, default 'ridge'
        Type of meta‑model to train ('ridge' or 'elasticnet' or 'catboost').
    alpha : float, default 1.0
        Regularisation strength for ridge or elasticnet.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        OOF predictions from the stacking model (same shape as y) and its
        predictions for the test set.
    """
    if model_type == 'ridge':
        meta_model = Ridge(alpha=alpha, random_state=42)
    else:
        raise NotImplementedError(f"Meta‑model type '{model_type}' is not implemented.")

    meta_model.fit(base_oof, y)
    meta_oof = meta_model.predict(base_oof)
    meta_test_preds = meta_model.predict(base_test_preds)
    return meta_oof, meta_test_preds


def main(args: argparse.Namespace) -> None:
    # 1. Load data
    train_df, test_df = read_data(args.data_dir)

    # 2. Identify column types
    numeric_cols, categorical_cols, text_cols = identify_columns(train_df)

    # 3. Preprocess tabular data
    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    full_df_proc, encoders = preprocess_tabular(
        full_df, numeric_cols, categorical_cols, is_train=True
    )
    # Split back into training and test processed frames
    train_proc = full_df_proc.iloc[: len(train_df)].copy()
    test_proc = full_df_proc.iloc[len(train_df) :].copy().reset_index(drop=True)

    # 4. Create text embeddings for both train and test sets
    print("Computing text embeddings...")
    train_texts = concatenate_text_fields(train_df, text_cols)
    test_texts = concatenate_text_fields(test_df, text_cols)
    text_emb_train = compute_text_embeddings(
        train_texts, model_name=args.embedding_model, batch_size=args.batch_size, device=args.device
    )
    text_emb_test = compute_text_embeddings(
        test_texts, model_name=args.embedding_model, batch_size=args.batch_size, device=args.device
    )

    # 5. Assemble feature matrices
    # Drop columns not used for tabular models (text fields and target)
    drop_cols = text_cols + ["settlement_index", "exploration_log", "environmental_report", "incident_report", "image_prompt", "description"]
    # Some of these may not exist in test
    drop_cols = [col for col in drop_cols if col in train_proc.columns]
    X_tab = train_proc.drop(columns=drop_cols + ["id"])
    X_tab_test = test_proc.drop(columns=drop_cols + ["id"])
    y = train_df["settlement_index"].values

    # Identify categorical columns positions for TabNet
    cat_feature_indices = [X_tab.columns.get_loc(col) for col in categorical_cols if col in X_tab.columns]
    cat_dims = [int(train_proc[col].nunique()) for col in categorical_cols if col in X_tab.columns]

    # Concatenate text embeddings with numeric/categorical features for TabNet and LightGBM
    X_tab_np = X_tab.values
    X_tab_test_np = X_tab_test.values
    X_tab_text = np.hstack([X_tab_np, text_emb_train])
    X_tab_text_test = np.hstack([X_tab_test_np, text_emb_test])

    # Keep DataFrame version for LightGBM (which accepts pandas)
    X_lgb = pd.DataFrame(X_tab_text, columns=[f"f{i}" for i in range(X_tab_text.shape[1])])
    X_lgb_test = pd.DataFrame(X_tab_text_test, columns=[f"f{i}" for i in range(X_tab_text.shape[1])])

    # 6. Train base models
    print("Training CatBoost...")
    cat_oof, cat_test = train_catboost(
        X_tab.reset_index(drop=True), y, [col for col in categorical_cols if col in X_tab.columns], X_tab_test.reset_index(drop=True), n_folds=args.n_folds
    )
    print(f"CatBoost OOF RMSE: {np.sqrt(mean_squared_error(y, cat_oof)):.4f}")

    import torch  # Deferred import to avoid requirement if not training TabNet
    print("Training TabNet...")
    tabnet_oof, tabnet_test = train_tabnet(
        X_tab_text.astype(np.float32),
        y.astype(np.float32),
        X_tab_text_test.astype(np.float32),
        categorical_dims=cat_dims,
        cat_idxs=cat_feature_indices,
        n_folds=args.n_folds,
    )
    print(f"TabNet OOF RMSE: {np.sqrt(mean_squared_error(y, tabnet_oof)):.4f}")

    print("Training LightGBM...")
    lgb_oof, lgb_test = train_lightgbm(
        X_lgb, y, X_lgb_test, n_folds=args.n_folds
    )
    print(f"LightGBM OOF RMSE: {np.sqrt(mean_squared_error(y, lgb_oof)):.4f}")

    # 7. Stack base model predictions
    base_oof = np.vstack([cat_oof, tabnet_oof, lgb_oof]).T
    base_test_preds = np.vstack([cat_test, tabnet_test, lgb_test]).T
    print("Training stacking model...")
    meta_oof, meta_test = train_stacking_model(
        base_oof, y, base_test_preds, model_type='ridge', alpha=args.meta_alpha
    )
    print(f"Stacking OOF RMSE: {np.sqrt(mean_squared_error(y, meta_oof)):.4f}")

    # 8. Save submission file
    submission = pd.DataFrame({
        'id': test_df['id'],
        'settlement_index': meta_test,
    })
    submission_path = os.path.join(args.output_dir, 'submission.csv')
    os.makedirs(args.output_dir, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Exoplanet settlement viability prediction pipeline")
    parser.add_argument(
        '--data_dir', type=str, default='dataset', help='Directory containing train.csv and test.csv'
    )
    parser.add_argument(
        '--output_dir', type=str, default='output', help='Directory to save the submission file'
    )
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross‑validation')
    parser.add_argument(
        '--embedding_model', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name'
    )
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for text embedding')
    parser.add_argument('--device', type=str, default='cpu', help="Device for text embedding ('cpu' or 'cuda')")
    parser.add_argument('--meta_alpha', type=float, default=1.0, help='Regularisation strength for the ridge meta‑model')
    args = parser.parse_args()
    main(args)