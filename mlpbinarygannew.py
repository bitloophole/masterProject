import random
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


# ------------------------------
# Global config
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "iot_dataset_after_undersampling.csv"
SYNTHETIC_TEST_PATH = "synthetic_test_set.csv"
PRETRAINED_MODEL_PATH = "mlp_global_model_eps_5.0.pt"

RANDOM_STATE = 42
TEST_SIZE = 0.2
EPSILON = 5.0


# ------------------------------
# Utilities
# ------------------------------
def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------
# Data preprocessing
# ------------------------------
def load_balanced_dataframe(csv_path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df.drop_duplicates(inplace=True)
    df.fillna(0, inplace=True)

    
    if "Label" in df.columns:
        df= df.drop(columns=["Label"])

    print("\n================ REAL DATA SUMMARY ================")
    print(df["Attack_Category"].value_counts())
    print("===================================================")

    return df


def split_global_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["Attack_Category"],
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def fit_scaler(train_df: pd.DataFrame, feature_columns: List[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[feature_columns].values)
    return scaler


def transform_dataframe(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_columns: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    X = scaler.transform(df[feature_columns].values)
    y = df["Attack_Category"].values.astype(int)
    return X, y


def load_synthetic_test_set(
    synthetic_csv_path: str,
    feature_columns: List[str],
) -> pd.DataFrame:
    synthetic_test_df = pd.read_csv(synthetic_csv_path)

    synthetic_test_df.drop_duplicates(inplace=True)
    synthetic_test_df.fillna(0, inplace=True)

    if "Label" in synthetic_test_df.columns:
        synthetic_test_df = synthetic_test_df.drop(columns=["Label"])

    if synthetic_test_df["Attack_Category"].dtype == object:
        synthetic_test_df["Attack_Category"] = synthetic_test_df["Attack_Category"].apply(
            lambda value: 0 if value == "BENIGN" else 1
        )

    required_columns = feature_columns + ["Attack_Category"]
    synthetic_test_df = synthetic_test_df[required_columns].copy()

    synthetic_test_df["Attack_Category"] = synthetic_test_df[
        "Attack_Category"
    ].astype(int)

    synthetic_test_df = synthetic_test_df.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    print("\n=========== SYNTHETIC TEST SET SUMMARY ===========")
    print(synthetic_test_df["Attack_Category"].value_counts())
    print("==================================================")

    return synthetic_test_df


# ------------------------------
# Model
# ------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_model(input_dim: int) -> nn.Module:
    return MLP(input_dim).to(DEVICE)


def load_trained_model(path: str, input_dim: int) -> nn.Module:
    model = make_model(input_dim)

    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()

    return model


# ------------------------------
# Evaluation
# ------------------------------
def test(model: nn.Module, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    start_time = time.perf_counter()

    with torch.no_grad():
        logits = model(X_tensor).cpu().numpy().ravel()

    inference_time = time.perf_counter() - start_time

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1_score": float(f1_score(y, preds, zero_division=0)),
        "inference_time_sec": float(inference_time),
        "inference_time_per_sample_sec": float(inference_time / max(len(y), 1)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }

    try:
        metrics["auc_roc"] = float(roc_auc_score(y, probs))
    except ValueError:
        metrics["auc_roc"] = float("nan")

    return metrics


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(f"\n================ {title} ================")
    print(f"Accuracy       : {metrics['accuracy']:.4f}")
    print(f"Precision      : {metrics['precision']:.4f}")
    print(f"Recall         : {metrics['recall']:.4f}")
    print(f"F1-score       : {metrics['f1_score']:.4f}")
    print(f"AUC-ROC        : {metrics['auc_roc']:.4f}")
    print(f"Inference Time : {metrics['inference_time_sec']:.6f} s")
    print("------------------------------------------")
    print("Confusion Matrix")
    print(f"TN: {int(metrics['tn'])}    FP: {int(metrics['fp'])}")
    print(f"FN: {int(metrics['fn'])}    TP: {int(metrics['tp'])}")
    print("==========================================")


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    set_seed(RANDOM_STATE)

    # 1. Load real dataset
    df = load_balanced_dataframe(DATA_PATH)

    # 2. Split real data exactly like training
    train_df, real_test_df = split_global_train_test(df)

    # 3. Use training data only to fit scaler
    feature_columns = [
        column for column in train_df.columns if column != "Attack_Category"
    ]

    scaler = fit_scaler(train_df, feature_columns)
    input_dim = len(feature_columns)

    print("\n================ MODEL INFO ================")
    print(f"Input dimension       : {input_dim}")
    print(f"Loaded pretrained DP model: {PRETRAINED_MODEL_PATH}")
    print(f"Privacy budget ε      : {EPSILON}")
    print("============================================")

    # 4. Load saved DP-trained model
    model = load_trained_model(
        path=PRETRAINED_MODEL_PATH,
        input_dim=input_dim,
    )

    # 5. Evaluate on real test data
    X_real, y_real = transform_dataframe(
        real_test_df,
        scaler,
        feature_columns,
    )

    real_test_metrics = test(model, X_real, y_real)

    # 6. Load and evaluate synthetic test data
    synthetic_test_df = load_synthetic_test_set(
        synthetic_csv_path=SYNTHETIC_TEST_PATH,
        feature_columns=feature_columns,
    )

    X_synthetic, y_synthetic = transform_dataframe(
        synthetic_test_df,
        scaler,
        feature_columns,
    )

    synthetic_test_metrics = test(model, X_synthetic, y_synthetic)

    # 7. Print results
    print_metrics("REAL TEST RESULTS", real_test_metrics)
    print_metrics("SYNTHETIC TEST RESULTS", synthetic_test_metrics)

    # 8. Save summary
    result = {
        "epsilon": EPSILON,

        "real_accuracy": real_test_metrics["accuracy"],
        "real_precision": real_test_metrics["precision"],
        "real_recall": real_test_metrics["recall"],
        "real_f1_score": real_test_metrics["f1_score"],
        "real_auc_roc": real_test_metrics["auc_roc"],
        "real_inference_time_sec": real_test_metrics["inference_time_sec"],
        "real_tn": real_test_metrics["tn"],
        "real_fp": real_test_metrics["fp"],
        "real_fn": real_test_metrics["fn"],
        "real_tp": real_test_metrics["tp"],

        "synthetic_accuracy": synthetic_test_metrics["accuracy"],
        "synthetic_precision": synthetic_test_metrics["precision"],
        "synthetic_recall": synthetic_test_metrics["recall"],
        "synthetic_f1_score": synthetic_test_metrics["f1_score"],
        "synthetic_auc_roc": synthetic_test_metrics["auc_roc"],
        "synthetic_inference_time_sec": synthetic_test_metrics["inference_time_sec"],
        "synthetic_tn": synthetic_test_metrics["tn"],
        "synthetic_fp": synthetic_test_metrics["fp"],
        "synthetic_fn": synthetic_test_metrics["fn"],
        "synthetic_tp": synthetic_test_metrics["tp"],
    }

    summary_df = pd.DataFrame([result])

    print("\n================ SUMMARY TABLE ================")
    print(summary_df.to_string(index=False))
    print("===============================================")

    summary_df.to_csv(
        "pretrained_dp_model_real_vs_synthetic_results.csv",
        index=False,
    )

    print("\nSaved results to: pretrained_dp_model_real_vs_synthetic_results.csv")