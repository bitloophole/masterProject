import logging
import random
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

from models import TabTransformer
from flwr.common import Context, Metrics, Parameters, parameters_to_ndarrays
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
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
from torch.utils.data import DataLoader, TensorDataset

logging.getLogger("flwr").setLevel(logging.WARNING)
logging.getLogger("flwr.server").setLevel(logging.WARNING)
logging.getLogger("flwr.client").setLevel(logging.WARNING)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLIENTS = 5
NUM_ROUNDS = 45
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
RANDOM_STATE = 42

EPSILON_VALUES = [1.0, 3.0, 5.0, 10.0]
MAX_GRAD_NORM = 0.5

# Use fixed noise multipliers instead of recalibrating noise every FL round.
# Smaller epsilon should use larger noise; larger epsilon should use smaller noise.
# This gives a stable, theory-consistent privacy/utility ordering.
EPSILON_TO_NOISE = {
    1.0: 1.5,
    3.0: 1.0,
    5.0: 0.7,
    10.0: 0.4,
}


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_bytes(size_in_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_in_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size_in_bytes:.2f} B"


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


@dataclass
class DPConfig:
    target_epsilon: float
    noise_multiplier: float
    max_grad_norm: float = MAX_GRAD_NORM


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv("iot_dataset_after_undersampling.csv")

    print("\n================ DATA SUMMARY ================")
    print(df["Attack_Category"].value_counts())

    X = df.drop(["Label", "Attack_Category"], axis=1).values
    y = df["Attack_Category"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


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


def make_model(input_dim: int) -> nn.Module:
    model = TabTransformer(input_dim, dropout=0.0)

    # Opacus compatibility validation. For TransformerEncoder/MultiheadAttention,
    # Opacus may warn depending on your installed version. If validation fails,
    # use the custom attention replacement note in chat.
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        model = ModuleValidator.fix(model)

    return model.to(DEVICE)


def get_model_stats(model: nn.Module, input_dim: int) -> Dict[str, float]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    total_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )

    if isinstance(model, TabTransformer):
        d = model.embed_dim
        L = model.input_dim
        ff_dim = model.ff_dim
        num_layers = model.num_layers

        embedding_flops = 2 * L * d
        per_layer_flops = (
            8 * L * d * d
            + 4 * L * L * d
            + 4 * L * d * ff_dim
        )
        transformer_flops = per_layer_flops * num_layers
        fc1_flops = 2 * (L * d) * 64
        fc2_flops = 2 * 64 * 32
        fc3_flops = 2 * 32 * 1
        forward_flops = embedding_flops + transformer_flops + fc1_flops + fc2_flops + fc3_flops
    else:
        forward_flops = 0.0

    training_flops = forward_flops * 3

    return {
        "parameters": float(total_params),
        "model_size_bytes": float(total_bytes),
        "forward_flops_per_sample": float(forward_flops),
        "training_flops_per_sample": float(training_flops),
    }


def get_parameter_bytes(parameters: List[np.ndarray]) -> int:
    return int(sum(array.nbytes for array in parameters))


def train_with_dp(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    dp_config: DPConfig,
    epochs: int = LOCAL_EPOCHS,
) -> Dict[str, float]:
    model.train()

    train_dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    target_delta = min(1e-5, 1.0 / max(len(X), 1))

    # Important fix:
    # Do NOT use make_private_with_epsilon inside every FL round.
    # That recalibrates noise each round and makes epsilon logs misleading.
    # Instead, use a fixed noise multiplier for each target epsilon.
    privacy_engine = PrivacyEngine(accountant="prv")
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=dp_config.noise_multiplier,
        max_grad_norm=dp_config.max_grad_norm,
    )

    train_losses: List[float] = []
    val_losses: List[float] = []

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    start_time = time.perf_counter()

    for _ in range(epochs):
        running_loss = 0.0
        total_samples = 0

        for batch_X, batch_y in train_loader:
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
    epsilon_spent = privacy_engine.get_epsilon(delta=target_delta)
    noise_multiplier = float(dp_config.noise_multiplier)

    return {
        "train_time_sec": float(training_time),
        "train_loss": float(train_losses[-1]),
        "val_loss": float(val_losses[-1]),
        # This epsilon is local to this client-round training call.
        # For thesis reporting, compare by target_epsilon and fixed noise_multiplier.
        "epsilon_spent": float(epsilon_spent),
        "target_epsilon": float(dp_config.target_epsilon),
        "delta": float(target_delta),
        "max_grad_norm": float(dp_config.max_grad_norm),
        "noise_multiplier": float(noise_multiplier),
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
        "epsilon_spent",
        "target_epsilon",
        "delta",
        "max_grad_norm",
        "noise_multiplier",
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
        input_dim: int,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        client_id: int,
        model_stats: Dict[str, float],
        dp_config: DPConfig,
    ) -> None:
        self.input_dim = input_dim
        self.model = make_model(input_dim)
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.client_id = client_id
        self.model_stats = model_stats
        self.dp_config = dp_config

    def get_parameters(self, config):
        return [value.detach().cpu().numpy() for _, value in self.model.state_dict().items()]

    def set_parameters(self, parameters) -> None:
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict(
            {key: torch.tensor(value, device=DEVICE) for key, value in state_dict.items()},
            strict=True,
        )

    def fit(self, parameters, config):
        # Important: rebuild fresh model every FL round, because Opacus wraps the model.
        self.model = make_model(self.input_dim)
        self.set_parameters(parameters)
        parameter_bytes = get_parameter_bytes(parameters)

        train_stats = train_with_dp(
            self.model,
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            dp_config=self.dp_config,
            epochs=LOCAL_EPOCHS,
        )

        metrics = {
            "train_time_sec": float(train_stats["train_time_sec"]),
            "train_loss": float(train_stats["train_loss"]),
            "val_loss": float(train_stats["val_loss"]),
            "epsilon_spent": float(train_stats["epsilon_spent"]),
            "target_epsilon": float(train_stats["target_epsilon"]),
            "delta": float(train_stats["delta"]),
            "max_grad_norm": float(train_stats["max_grad_norm"]),
            "noise_multiplier": float(train_stats["noise_multiplier"]),
            "communication_bytes": float(parameter_bytes * 2),
            "model_size_bytes": float(self.model_stats["model_size_bytes"]),
            "parameter_count": float(self.model_stats["parameters"]),
            "training_flops": float(
                len(self.X_train) * LOCAL_EPOCHS * self.model_stats["training_flops_per_sample"]
            ),
        }

        return self.get_parameters(config), len(self.X_train), metrics


class MetricsFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, experiment_epsilon: float, **kwargs):
        super().__init__(**kwargs)
        self.experiment_epsilon = experiment_epsilon
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
                "target_epsilon": float(aggregated_metrics.get("target_epsilon", self.experiment_epsilon)),
                "epsilon_spent": float(aggregated_metrics.get("epsilon_spent", 0.0)),
                "noise_multiplier": float(aggregated_metrics.get("noise_multiplier", 0.0)),
                "train_loss": float(aggregated_metrics.get("train_loss", 0.0)),
                "val_loss": float(aggregated_metrics.get("val_loss", 0.0)),
                "round_total_train_time_sec": float(
                    aggregated_metrics.get("round_total_train_time_sec", 0.0)
                ),
                "round_max_train_time_sec": float(
                    aggregated_metrics.get("round_max_train_time_sec", 0.0)
                ),
                "communication_bytes": float(aggregated_metrics.get("communication_bytes", 0.0)),
            }

            self.round_logs.append(row)

            print(
                f"Round {server_round:02d} | "
                f"Eps(target)={row['target_epsilon']:.2f} | "
                f"Eps(spent)={row['epsilon_spent']:.4f} | "
                f"NoiseMult={row['noise_multiplier']:.4f} | "
                f"TrainLoss={row['train_loss']:.4f} | "
                f"ValLoss={row['val_loss']:.4f} | "
                f"RoundTrainTime(sum)={row['round_total_train_time_sec']:.4f}s | "
                f"Comm={format_bytes(row['communication_bytes'])}"
            )

        return aggregated_parameters, aggregated_metrics


def make_client_fn(
    resources: ExperimentResources, dp_config: DPConfig
) -> Callable[[Context], fl.client.Client]:
    def client_fn(context: Context):
        cid = int(context.node_config["partition-id"])
        client_data = resources.clients[cid]

        client = FLClient(
            input_dim=resources.input_dim,
            X_train=client_data.X_train,
            X_val=client_data.X_val,
            y_train=client_data.y_train,
            y_val=client_data.y_val,
            client_id=cid,
            model_stats=resources.model_stats,
            dp_config=dp_config,
        )
        return client.to_client()

    return client_fn


def plot_loss_curves(round_logs: List[Dict[str, float]], epsilon: float) -> None:
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
    plt.title(f"Training and Validation Loss per Round, DP ε={epsilon}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_metric_vs_epsilon(summary_df: pd.DataFrame, metric: str, ylabel: str) -> None:
    if summary_df.empty or metric not in summary_df.columns:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(summary_df["epsilon"], summary_df[metric], marker="o", linewidth=2)
    plt.xlabel("Privacy Budget, ε")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Privacy Budget")
    plt.grid(True, linestyle="--", alpha=0.5)
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
        model = make_model(resources.input_dim)
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
    aggregated["final_eval_communication_bytes"] = float(
        parameter_bytes * len(resources.clients)
    )
    aggregated["total_test_examples"] = float(total_test_examples)

    return aggregated


def save_global_model(parameters: Parameters, input_dim: int, epsilon: float, path: str = ".") -> None:
    if parameters is None:
        print(f"No model to save for ε={epsilon}")
        return

    model = make_model(input_dim)
    ndarrays = parameters_to_ndarrays(parameters)
    state_dict = dict(zip(model.state_dict().keys(), ndarrays))

    model.load_state_dict(
        {key: torch.tensor(value, device=DEVICE) for key, value in state_dict.items()},
        strict=True,
    )

    filename = f"{path}/ttf_global_model_eps_{epsilon}.pt"
    torch.save(model.state_dict(), filename)
    print(f"Saved model for ε={epsilon} → {filename}")


def run_dp_experiment(
    epsilon: float,
    resources: ExperimentResources,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    print("\n" + "=" * 70)
    print(f"STARTING TTF DP EXPERIMENT WITH PRIVACY BUDGET ε = {epsilon}")
    print("=" * 70)

    set_seed(RANDOM_STATE)
    dp_config = DPConfig(
        target_epsilon=epsilon,
        noise_multiplier=EPSILON_TO_NOISE[float(epsilon)],
        max_grad_norm=MAX_GRAD_NORM,
    )

    strategy = MetricsFedAvg(
        experiment_epsilon=epsilon,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=0,
        min_available_clients=NUM_CLIENTS,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
    )

    experiment_start = time.perf_counter()

    fl.simulation.start_simulation(
        client_fn=make_client_fn(resources, dp_config),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    experiment_runtime = time.perf_counter() - experiment_start
    final_eval = evaluate_final_global_model(strategy.final_parameters, resources)

    total_training_communication_bytes = float(
        sum(row["communication_bytes"] for row in strategy.round_logs)
    )
    final_eval_communication_bytes = float(
        final_eval.get("final_eval_communication_bytes", 0.0)
    )
    total_communication_bytes = total_training_communication_bytes + final_eval_communication_bytes

    total_client_train_time_sec = float(
        sum(row["round_total_train_time_sec"] for row in strategy.round_logs)
    )
    total_round_max_train_time_sec = float(
        sum(row["round_max_train_time_sec"] for row in strategy.round_logs)
    )

    last_train_loss = float(strategy.round_logs[-1]["train_loss"]) if strategy.round_logs else 0.0
    last_val_loss = float(strategy.round_logs[-1]["val_loss"]) if strategy.round_logs else 0.0
    last_epsilon_spent = float(strategy.round_logs[-1]["epsilon_spent"]) if strategy.round_logs else 0.0
    last_noise_multiplier = float(strategy.round_logs[-1]["noise_multiplier"]) if strategy.round_logs else 0.0

    result = {
        "epsilon": float(epsilon),
        "epsilon_spent_last_round": float(last_epsilon_spent),
        "noise_multiplier_last_round": float(last_noise_multiplier),
        "accuracy": float(final_eval.get("accuracy", 0.0)),
        "precision": float(final_eval.get("precision", 0.0)),
        "recall": float(final_eval.get("recall", 0.0)),
        "f1_score": float(final_eval.get("f1_score", 0.0)),
        "auc_roc": float(final_eval.get("auc_roc", 0.0)),
        "final_train_loss": float(last_train_loss),
        "final_val_loss": float(last_val_loss),
        "client_train_time_sum_sec": float(total_client_train_time_sec),
        "round_train_time_sum_of_max_sec": float(total_round_max_train_time_sec),
        "final_inference_time_total_sec": float(
            final_eval.get("final_eval_total_inference_time_sec", 0.0)
        ),
        "avg_inference_time_per_sample_ms": float(
            final_eval.get("final_eval_avg_inference_time_per_sample_sec", 0.0) * 1000.0
        ),
        "training_communication_bytes": float(total_training_communication_bytes),
        "final_eval_communication_bytes": float(final_eval_communication_bytes),
        "total_communication_bytes": float(total_communication_bytes),
        "total_runtime_sec": float(experiment_runtime),
        "tn": float(final_eval.get("tn", 0.0)),
        "fp": float(final_eval.get("fp", 0.0)),
        "fn": float(final_eval.get("fn", 0.0)),
        "tp": float(final_eval.get("tp", 0.0)),
    }

    print("\n================ FINAL RESULTS ================")
    print(f"Privacy Budget (ε)    : {epsilon:.2f}")
    print(f"Epsilon Spent         : {result['epsilon_spent_last_round']:.4f}")
    print(f"Noise Multiplier      : {result['noise_multiplier_last_round']:.4f}")
    print(f"Final Accuracy        : {result['accuracy']:.4f}")
    print(f"Final Precision       : {result['precision']:.4f}")
    print(f"Final Recall          : {result['recall']:.4f}")
    print(f"Final F1-score        : {result['f1_score']:.4f}")
    print(f"Final AUC-ROC         : {result['auc_roc']:.4f}")
    print(f"Final Train Loss      : {result['final_train_loss']:.4f}")
    print(f"Final Validation Loss : {result['final_val_loss']:.4f}")
    print(f"Client Train Time(sum): {result['client_train_time_sum_sec']:.4f} s")
    print(f"Round Train Time(sum of max/client-round): {result['round_train_time_sum_of_max_sec']:.4f} s")
    print(f"Final Inference Time(total): {result['final_inference_time_total_sec']:.6f} s")
    print(f"Avg Inference Time/sample: {result['avg_inference_time_per_sample_ms']:.6f} ms")
    print(f"Training Communication : {format_bytes(result['training_communication_bytes'])}")
    print(f"Final Eval Communication: {format_bytes(result['final_eval_communication_bytes'])}")
    print(f"Total Communication    : {format_bytes(result['total_communication_bytes'])}")
    print(f"Total Runtime          : {result['total_runtime_sec']:.2f} s")
    print("================================================")

    print("\n========== CONFUSION MATRIX ==========")
    print(f"TN: {int(result['tn'])}    FP: {int(result['fp'])}")
    print(f"FN: {int(result['fn'])}    TP: {int(result['tp'])}")
    print("======================================\n")

    print(f"Round-wise Results for TTF ε={epsilon}:")
    print("Round\tEpsSpent\tNoiseMult\tTrainLoss\tValLoss\tRoundTrainTime(sum)\tComm")
    for row in strategy.round_logs:
        print(
            f"{int(row['round'])}\t"
            f"{row['epsilon_spent']:.4f}\t"
            f"{row['noise_multiplier']:.4f}\t"
            f"{row['train_loss']:.4f}\t"
            f"{row['val_loss']:.4f}\t"
            f"{row['round_total_train_time_sec']:.4f}\t"
            f"{row['communication_bytes']:.0f}"
        )

    save_global_model(strategy.final_parameters, resources.input_dim, epsilon)
    return result, strategy.round_logs


if __name__ == "__main__":
    set_seed(RANDOM_STATE)
    total_start = time.perf_counter()

    X, y = load_data()
    clients = create_clients(X, y, num_clients=NUM_CLIENTS)
    input_dim = X.shape[1]
    model_stats = get_model_stats(make_model(input_dim), input_dim)

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

    print("============== FEDERATED TTF DP TRAINING ==============")
    print(
        f"Clients={NUM_CLIENTS} | Rounds={NUM_ROUNDS} | "
        f"Local Epochs={LOCAL_EPOCHS} | Batch Size={BATCH_SIZE}"
    )
    print(f"Privacy Budgets (ε)={EPSILON_VALUES}")
    print(f"Fixed epsilon-to-noise mapping={EPSILON_TO_NOISE}")
    print("========================================================\n")

    all_results: List[Dict[str, float]] = []
    all_round_logs: Dict[float, List[Dict[str, float]]] = {}

    for epsilon in EPSILON_VALUES:
        result, round_logs = run_dp_experiment(epsilon, resources)
        all_results.append(result)
        all_round_logs[epsilon] = round_logs
        plot_loss_curves(round_logs, epsilon)

    total_runtime = time.perf_counter() - total_start
    summary_df = pd.DataFrame(all_results).sort_values("epsilon").reset_index(drop=True)

    print("\n================ TTF DP EXPERIMENT SUMMARY ================")
    print(
        summary_df[
            [
                "epsilon",
                "epsilon_spent_last_round",
                "noise_multiplier_last_round",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "auc_roc",
                "final_train_loss",
                "final_val_loss",
                "total_runtime_sec",
            ]
        ].to_string(index=False)
    )
    print("===========================================================\n")

    print("Formatted summary for thesis table:")
    print(
        "Epsilon\tEpsSpent\tNoiseMult\tAccuracy\tPrecision\tRecall\tF1\tAUC\tTrainLoss\tValLoss\tRuntime(s)"
    )
    for _, row in summary_df.iterrows():
        print(
            f"{row['epsilon']:.1f}\t"
            f"{row['epsilon_spent_last_round']:.4f}\t"
            f"{row['noise_multiplier_last_round']:.4f}\t"
            f"{row['accuracy']:.4f}\t"
            f"{row['precision']:.4f}\t"
            f"{row['recall']:.4f}\t"
            f"{row['f1_score']:.4f}\t"
            f"{row['auc_roc']:.4f}\t"
            f"{row['final_train_loss']:.4f}\t"
            f"{row['final_val_loss']:.4f}\t"
            f"{row['total_runtime_sec']:.2f}"
        )

    summary_df.to_csv("ttf_dp_federated_results_summary.csv", index=False)
    print("\nSaved summary to: ttf_dp_federated_results_summary.csv")

    plot_metric_vs_epsilon(summary_df, "accuracy", "Accuracy")
    plot_metric_vs_epsilon(summary_df, "precision", "Precision")
    plot_metric_vs_epsilon(summary_df, "recall", "Recall")
    plot_metric_vs_epsilon(summary_df, "f1_score", "F1-score")
    plot_metric_vs_epsilon(summary_df, "auc_roc", "AUC-ROC")
    plot_metric_vs_epsilon(summary_df, "final_val_loss", "Final Validation Loss")

    print(f"\nTOTAL WALL-CLOCK TIME FOR ALL TTF DP EXPERIMENTS: {total_runtime:.2f} s")
