
import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from flwr.common import Context, Metrics, Parameters, parameters_to_ndarrays
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
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------
# Reduce Flower log noise
# ------------------------------
logging.getLogger("flwr").setLevel(logging.WARNING)
logging.getLogger("flwr.server").setLevel(logging.WARNING)
logging.getLogger("flwr.client").setLevel(logging.WARNING)


# Global config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLIENTS = 5
NUM_ROUNDS = 45
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
RANDOM_STATE = 42


def format_bytes(size_in_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_in_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size_in_bytes:.2f} B"



# DATA CONTAINER


@dataclass
class ClientPartition:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


@dataclass
class ExperimentResources:
    clients: List[ClientPartition]
    input_dim: int
    model_stats: Dict[str, float]


 
#  PREPROCESS DATA


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv("iot_dataset_after_undersampling.csv")

    
    print("\n================ DATA SUMMARY ================")
    print(df["Attack_Category"].value_counts())

    X = df.drop(["Label", "Attack_Category"], axis=1).values
    y = df["Attack_Category"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y



#  CLIENT DATA SPLIT 


def create_clients(
    X: np.ndarray, y: np.ndarray, num_clients: int = NUM_CLIENTS
) -> List[ClientPartition]:
    clients: List[ClientPartition] = []
    rng = np.random.default_rng(RANDOM_STATE)

    idx_benign = np.where(y == 0)[0]
    idx_attack = np.where(y == 1)[0]

    rng.shuffle(idx_benign)
    rng.shuffle(idx_attack)

    benign_splits = np.array_split(idx_benign, num_clients)
    attack_splits = np.array_split(idx_attack, num_clients)

    print("\n============== CLIENT DISTRIBUTION ==============")

    for client_idx in range(num_clients):
        if client_idx % 2 == 0:
            idx = np.concatenate(
                [
                    benign_splits[client_idx],
                    attack_splits[client_idx][: len(attack_splits[client_idx]) // 3],
                ]
            )
        else:
            idx = np.concatenate(
                [
                    benign_splits[client_idx][: len(benign_splits[client_idx]) // 3],
                    attack_splits[client_idx],
                ]
            )

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

        benign_count = int((y_client == 0).sum())
        attack_count = int((y_client == 1).sum())

        print(
            f"Client {client_idx}: total={len(y_client)} | "
            f"benign={benign_count} | attack={attack_count} | "
            f"train={len(y_train)} | val={len(y_val)} | test={len(y_test)}"
        )

        clients.append(
            ClientPartition(
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
            )
        )

    print("=================================================\n")
    return clients



#  MODEL

"""
class MLP(nn.Module):
    
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # 1st hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),         # 2nd hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),          # 3rd hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),   
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

"""
class MLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),   # 1st hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),          # 2nd hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1),           # output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def get_model_stats(model: nn.Module, input_dim: int) -> Dict[str, float]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    total_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )

    # Approximate FLOPs
    forward_flops = (2 * input_dim * 64) + (2 * 64 * 1) + 64 + 1
    training_flops = forward_flops * 3  # rough approximation

    return {
        "parameters": float(total_params),
        "model_size_bytes": float(total_bytes),
        "forward_flops_per_sample": float(forward_flops),
        "training_flops_per_sample": float(training_flops),
    }


def get_parameter_bytes(parameters: List[np.ndarray]) -> int:
    return int(sum(array.nbytes for array in parameters))



# TRAIN & TEST


def train(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = LOCAL_EPOCHS,
) -> Dict[str, float]:
    model.train()

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    train_losses: List[float] = []
    val_losses: List[float] = []

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    start_time = time.perf_counter()

    for _ in range(epochs):
        running_loss = 0.0
        total_samples = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            current_batch_size = batch_X.size(0)
            running_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

        train_losses.append(running_loss / max(total_samples, 1))

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = criterion(val_logits, y_val_tensor).item()
        val_losses.append(float(val_loss))
        model.train()

    training_time = time.perf_counter() - start_time
    return {
        "train_time_sec": float(training_time),
        "train_loss": float(train_losses[-1]),
        "val_loss": float(val_losses[-1]),
    }


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




def weighted_average(metrics: List[Tuple[int, Metrics]], keys: List[str]) -> Dict[str, float]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: Dict[str, float] = {}
    for key in keys:
        valid = []
        for num_examples, metric in metrics:
            value = metric.get(key)
            if value is None:
                continue
            if isinstance(value, (int, float)) and not np.isnan(value):
                valid.append((num_examples, float(value)))

        if valid:
            aggregated[key] = sum(n * v for n, v in valid) / sum(n for n, _ in valid)

    return aggregated


def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    scalar_keys = [
        "train_loss",
        "val_loss",
    ]
    aggregated = weighted_average(metrics, scalar_keys)

    aggregated["total_train_examples"] = float(sum(num for num, _ in metrics))
    aggregated["communication_bytes"] = float(
        sum(metric.get("communication_bytes", 0.0) for _, metric in metrics)
    )
    aggregated["training_flops"] = float(
        sum(metric.get("training_flops", 0.0) for _, metric in metrics)
    )

    aggregated["round_total_train_time_sec"] = float(
        sum(metric.get("train_time_sec", 0.0) for _, metric in metrics)
    )
    aggregated["round_max_train_time_sec"] = float(
        max((metric.get("train_time_sec", 0.0) for _, metric in metrics), default=0.0)
    )

    aggregated["parameter_count"] = float(metrics[0][1].get("parameter_count", 0.0))
    aggregated["model_size_bytes"] = float(metrics[0][1].get("model_size_bytes", 0.0))
    return aggregated



class FLClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        client_id: int,
        model_stats: Dict[str, float],
    ) -> None:
        self.model = model
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.client_id = client_id
        self.model_stats = model_stats

    def get_parameters(self, config):
        return [value.detach().cpu().numpy() for _, value in self.model.state_dict().items()]

    def set_parameters(self, parameters) -> None:
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict(
            {key: torch.tensor(value, device=DEVICE) for key, value in state_dict.items()},
            strict=True,
        )

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        parameter_bytes = get_parameter_bytes(parameters)

        train_stats = train(
            self.model,
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            epochs=LOCAL_EPOCHS,
        )

        metrics = {
            "train_time_sec": float(train_stats["train_time_sec"]),
            "train_loss": float(train_stats["train_loss"]),
            "val_loss": float(train_stats["val_loss"]),
            "communication_bytes": float(parameter_bytes * 2),  # receive + send
            "model_size_bytes": float(self.model_stats["model_size_bytes"]),
            "parameter_count": float(self.model_stats["parameters"]),
            "training_flops": float(
                len(self.X_train)
                * LOCAL_EPOCHS
                * self.model_stats["training_flops_per_sample"]
            ),
        }

        return self.get_parameters(config), len(self.X_train), metrics




class MetricsFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round_logs: List[Dict[str, float]] = []
        self.final_parameters: Optional[Parameters] = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters

        if aggregated_metrics:
            row = {
                "round": float(server_round),
                "train_loss": float(aggregated_metrics.get("train_loss", 0.0)),
                "val_loss": float(aggregated_metrics.get("val_loss", 0.0)),
                "round_total_train_time_sec": float(
                    aggregated_metrics.get("round_total_train_time_sec", 0.0)
                ),
                "round_max_train_time_sec": float(
                    aggregated_metrics.get("round_max_train_time_sec", 0.0)
                ),
                "communication_bytes": float(
                    aggregated_metrics.get("communication_bytes", 0.0)
                ),
            }

            self.round_logs.append(row)

            print(
                f"Round {server_round:02d} | "
                f"TrainLoss={row['train_loss']:.4f} | "
                f"ValLoss={row['val_loss']:.4f} | "
                f"RoundTrainTime(sum)={row['round_total_train_time_sec']:.4f}s | "
                f"Comm={format_bytes(row['communication_bytes'])}"
            )

        return aggregated_parameters, aggregated_metrics




def make_client_fn(resources: ExperimentResources) -> Callable[[Context], fl.client.Client]:
    def client_fn(context: Context):
        cid = int(context.node_config["partition-id"])
        client_data = resources.clients[cid]
        model = MLP(input_dim=resources.input_dim).to(DEVICE)

        client = FLClient(
            model=model,
            X_train=client_data.X_train,
            X_val=client_data.X_val,
            y_train=client_data.y_train,
            y_val=client_data.y_val,
            client_id=cid,
            model_stats=resources.model_stats,
        )
        return client.to_client()

    return client_fn


def plot_loss_curves(round_logs: List[Dict[str, float]]) -> None:
    if not round_logs:
        return

    rounds = [int(row["round"]) for row in round_logs]
    train_losses = [float(row["train_loss"]) for row in round_logs]
    val_losses = [float(row["val_loss"]) for row in round_logs]

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, train_losses, marker="o", linewidth=2, label="Training Loss")
    plt.plot(rounds, val_losses, marker="s", linewidth=2, label="Validation Loss")
    plt.xlabel("Federated Round")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Round")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()



def evaluate_final_global_model(
    parameters: Parameters,
    resources: ExperimentResources,
) -> Dict[str, float]:
    if parameters is None:
        return {}

    ndarrays = parameters_to_ndarrays(parameters)
    parameter_bytes = get_parameter_bytes(ndarrays)

    client_metrics: List[Tuple[int, Metrics]] = []
    total_inference_time_sec = 0.0
    total_test_examples = 0

    for client_data in resources.clients:
        model = MLP(resources.input_dim).to(DEVICE)

        state_dict = dict(zip(model.state_dict().keys(), ndarrays))
        model.load_state_dict(
            {key: torch.tensor(value, device=DEVICE) for key, value in state_dict.items()},
            strict=True,
        )

        metrics = test(model, client_data.X_test, client_data.y_test)
        num_examples = len(client_data.y_test)

        client_metrics.append((num_examples, metrics))
        total_inference_time_sec += float(metrics["inference_time_sec"])
        total_test_examples += num_examples

    aggregated = weighted_average(
        client_metrics,
        ["accuracy", "precision", "recall", "f1_score", "auc_roc"],
    )

    aggregated["tn"] = float(sum(metric.get("tn", 0.0) for _, metric in client_metrics))
    aggregated["fp"] = float(sum(metric.get("fp", 0.0) for _, metric in client_metrics))
    aggregated["fn"] = float(sum(metric.get("fn", 0.0) for _, metric in client_metrics))
    aggregated["tp"] = float(sum(metric.get("tp", 0.0) for _, metric in client_metrics))

    aggregated["final_eval_total_inference_time_sec"] = float(total_inference_time_sec)
    aggregated["final_eval_avg_inference_time_per_sample_sec"] = float(
        total_inference_time_sec / max(total_test_examples, 1)
    )
    aggregated["final_eval_communication_bytes"] = float(parameter_bytes * len(resources.clients))
    aggregated["total_test_examples"] = float(total_test_examples)

    return aggregated




if __name__ == "__main__":
    total_start = time.perf_counter()

    X, y = load_data()
    clients = create_clients(X, y, num_clients=NUM_CLIENTS)
    input_dim = X.shape[1]
    model_stats = get_model_stats(MLP(input_dim), input_dim)

    resources = ExperimentResources(
        clients=clients,
        input_dim=input_dim,
        model_stats=model_stats,
    )

    print("================ MODEL PROFILE ================")
    print(f"Parameters           : {int(model_stats['parameters'])}")
    print(f"Model Size           : {format_bytes(model_stats['model_size_bytes'])}")
    print(f"Forward FLOPs/sample : {int(model_stats['forward_flops_per_sample'])}")
    print(f"Training FLOPs/sample: {int(model_stats['training_flops_per_sample'])}")
    print("================================================\n")

    strategy = MetricsFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,      # no round-wise test/validation evaluation by server
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=0,
        min_available_clients=NUM_CLIENTS,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
    )

    print("============== FEDERATED TRAINING ==============")
    print(
        f"Clients={NUM_CLIENTS} | Rounds={NUM_ROUNDS} | "
        f"Local Epochs={LOCAL_EPOCHS} | Batch Size={BATCH_SIZE}"
    )
    print("================================================\n")

    fl.simulation.start_simulation(
        client_fn=make_client_fn(resources),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    total_runtime = time.perf_counter() - total_start

    final_eval = evaluate_final_global_model(strategy.final_parameters, resources)

    total_training_communication_bytes = float(
        sum(row["communication_bytes"] for row in strategy.round_logs)
    )
    final_eval_communication_bytes = float(
        final_eval.get("final_eval_communication_bytes", 0.0)
    )
    total_communication_bytes = (
        total_training_communication_bytes + final_eval_communication_bytes
    )

    total_client_train_time_sec = float(
        sum(row["round_total_train_time_sec"] for row in strategy.round_logs)
    )
    total_round_max_train_time_sec = float(
        sum(row["round_max_train_time_sec"] for row in strategy.round_logs)
    )

    last_train_loss = float(strategy.round_logs[-1]["train_loss"]) if strategy.round_logs else 0.0
    last_val_loss = float(strategy.round_logs[-1]["val_loss"]) if strategy.round_logs else 0.0

    print("\n================ FINAL RESULTS ================")
    print(f"Final Accuracy       : {final_eval.get('accuracy', 0.0):.4f}")
    print(f"Final Precision      : {final_eval.get('precision', 0.0):.4f}")
    print(f"Final Recall         : {final_eval.get('recall', 0.0):.4f}")
    print(f"Final F1-score       : {final_eval.get('f1_score', 0.0):.4f}")
    print(f"Final AUC-ROC        : {final_eval.get('auc_roc', 0.0):.4f}")
    print(f"Final Train Loss     : {last_train_loss:.4f}")
    print(f"Final Validation Loss: {last_val_loss:.4f}")
    print(f"Client Train Time(sum): {total_client_train_time_sec:.4f} s")
    print(f"Round Train Time(sum of max/client-round): {total_round_max_train_time_sec:.4f} s")
    print(
        f"Final Inference Time(total over all test clients): "
        f"{final_eval.get('final_eval_total_inference_time_sec', 0.0):.6f} s"
    )
    print(
        f"Avg Inference Time/sample: "
        f"{final_eval.get('final_eval_avg_inference_time_per_sample_sec', 0.0) * 1000:.6f} ms"
    )
    print(f"Training Communication : {format_bytes(total_training_communication_bytes)}")
    print(f"Final Eval Communication: {format_bytes(final_eval_communication_bytes)}")
    print(f"Total Communication    : {format_bytes(total_communication_bytes)}")
    print(f"Total Runtime          : {total_runtime:.2f} s")
    print("================================================")

    print("\n==========  CONFUSION MATRIX ==========")
    print(f"TN: {int(final_eval.get('tn', 0))}    FP: {int(final_eval.get('fp', 0))}")
    print(f"FN: {int(final_eval.get('fn', 0))}    TP: {int(final_eval.get('tp', 0))}")
    print("=======================================================\n")

    print("Round-wise Results (copy into thesis table):")
    print("Round\tTrainLoss\tValLoss\tRoundTrainTime(sum)\tComm")
    for row in strategy.round_logs:
        print(
            f"{int(row['round'])}\t"
            f"{row['train_loss']:.4f}\t"
            f"{row['val_loss']:.4f}\t"
            f"{row['round_total_train_time_sec']:.4f}\t"
            f"{row['communication_bytes']:.0f}"
        )

    plot_loss_curves(strategy.round_logs)