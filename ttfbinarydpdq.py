import os
import time
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.ao.quantization import quantize_dynamic

from opacus.validators import ModuleValidator
DEVICE = torch.device("cpu")

NUM_CLIENTS = 5
RANDOM_STATE = 42

DATA_PATH = "iot_dataset_after_undersampling.csv"

# Change this to ttf_global_model_eps_5.0.pt if you want epsilon 5
MODEL_PATH = "saved_models/ttf_global_model_eps_1.0.pt"


@dataclass
class ClientPartition:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


class TabTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.ff_dim = ff_dim

        self.feature_embedding = nn.Parameter(torch.randn(input_dim, embed_dim))
        self.feature_bias = nn.Parameter(torch.zeros(input_dim, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        tokens = x * self.feature_embedding.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        tokens = self.transformer(tokens)
        return self.classifier(tokens)


def format_bytes(size_in_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_in_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA_PATH)

    df.drop_duplicates(inplace=True)
    df.fillna(0, inplace=True)

    print("\n================ DATA SUMMARY ================")
    print(df["Attack_Category"].value_counts())
    print("================================================")

    X = df.drop(["Label", "Attack_Category"], axis=1, errors="ignore").values
    y = df["Attack_Category"].values.astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def create_clients(X, y, num_clients=NUM_CLIENTS) -> List[ClientPartition]:
    clients = []
    rng = np.random.default_rng(RANDOM_STATE)

    idx_benign = np.where(y == 0)[0]
    idx_attack = np.where(y == 1)[0]

    rng.shuffle(idx_benign)
    rng.shuffle(idx_attack)

    benign_splits = np.array_split(idx_benign, num_clients)
    attack_splits = np.array_split(idx_attack, num_clients)

    for client_idx in range(num_clients):
        if client_idx % 2 == 0:
            idx = np.concatenate([
                benign_splits[client_idx],
                attack_splits[client_idx][: len(attack_splits[client_idx]) // 3],
            ])
        else:
            idx = np.concatenate([
                benign_splits[client_idx][: len(benign_splits[client_idx]) // 3],
                attack_splits[client_idx],
            ])

        rng.shuffle(idx)

        X_client = X[idx]
        y_client = y[idx]

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_client,
            y_client,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_client,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.125,
            random_state=RANDOM_STATE,
            stratify=y_train_full,
        )

        clients.append(ClientPartition(X_train, X_val, X_test, y_train, y_val, y_test))

    return clients


def get_model_disk_size_bytes(model: nn.Module) -> int:
    model_cpu = model.to("cpu").eval()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        temp_path = tmp.name

    try:
        torch.save(model_cpu, temp_path)
        size_bytes = os.path.getsize(temp_path)
    finally:
        os.remove(temp_path)

    return int(size_bytes)

def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    torch.backends.quantized.engine = "qnnpack"

    model_cpu = model.to("cpu").eval()

    quantized_model = quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8,
    )

    return quantized_model


def test(model: nn.Module, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    start_time = time.perf_counter()
    with torch.no_grad():
        logits = model(X_tensor).cpu().numpy().ravel()
    inference_time = time.perf_counter() - start_time

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1_score": f1_score(y, preds, zero_division=0),
        "auc_roc": roc_auc_score(y, probs),
        "inference_time_sec": inference_time,
        "inference_time_per_sample_sec": inference_time / max(len(y), 1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


if __name__ == "__main__":
    X, y = load_data()
    clients = create_clients(X, y)

    input_dim = X.shape[1]

    fp32_model = TabTransformer(input_dim)

    errors = ModuleValidator.validate(fp32_model, strict=False)
    if errors:
        fp32_model = ModuleValidator.fix(fp32_model)

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    fp32_model.load_state_dict(state_dict, strict=True)
    fp32_model.eval()

    fp32_size = get_model_disk_size_bytes(fp32_model)

    quantized_model = apply_dynamic_quantization(fp32_model)
    int8_size = get_model_disk_size_bytes(quantized_model)

    total_time = 0.0
    total_samples = 0

    all_metrics = []
    total_tn = total_fp = total_fn = total_tp = 0

    for client in clients:
        metrics = test(quantized_model, client.X_test, client.y_test)
        n = len(client.y_test)

        all_metrics.append((n, metrics))
        total_time += metrics["inference_time_sec"]
        total_samples += n

        total_tn += metrics["tn"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]
        total_tp += metrics["tp"]

    def weighted_avg(key):
        return sum(n * m[key] for n, m in all_metrics) / sum(n for n, _ in all_metrics)

    print("\n================ TTF PTDQ RESULTS ================")
    print(f"Loaded Model          : {MODEL_PATH}")
    print(f"Dataset               : {DATA_PATH}")
    print(f"Accuracy              : {weighted_avg('accuracy'):.4f}")
    print(f"Precision             : {weighted_avg('precision'):.4f}")
    print(f"Recall                : {weighted_avg('recall'):.4f}")
    print(f"F1-score              : {weighted_avg('f1_score'):.4f}")
    print(f"AUC-ROC               : {weighted_avg('auc_roc'):.4f}")
    print(f"Total Inference Time  : {total_time:.6f} s")
    print(f"Avg Inference/sample  : {(total_time / total_samples) * 1000:.6f} ms")
    print(f"FP32 Model Size       : {format_bytes(fp32_size)}")
    print(f"INT8 Model Size       : {format_bytes(int8_size)}")
    print(f"Size Reduction        : {format_bytes(fp32_size - int8_size)}")
    print(f"Reduction Percentage  : {100 * (fp32_size - int8_size) / max(fp32_size, 1):.2f}%")
    print("==================================================")

    print("\n========== CONFUSION MATRIX ==========")
    print(f"TN: {int(total_tn)}    FP: {int(total_fp)}")
    print(f"FN: {int(total_fn)}    TP: {int(total_tp)}")
    print("======================================")