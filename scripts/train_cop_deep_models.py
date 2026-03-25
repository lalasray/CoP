#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


RAW_DIR = Path("Data")
COP_DIR = Path("cop_exports_all")
OUTPUT_DIR = Path("model_outputs_deep")
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_CM_PER_CELL = 2.0
Y_CM_PER_CELL = 2.1
IMU_AXIS_COLUMNS = [
    "IMU_S1_acc_x",
    "IMU_S1_acc_y",
    "IMU_S1_acc_z",
    "IMU_S1_gyro_x",
    "IMU_S1_gyro_y",
    "IMU_S1_gyro_z",
    "IMU_S2_acc_x",
    "IMU_S2_acc_y",
    "IMU_S2_acc_z",
    "IMU_S2_gyro_x",
    "IMU_S2_gyro_y",
    "IMU_S2_gyro_z",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def subject_name(raw_path: Path) -> str:
    return raw_path.stem.split("_")[-1]


def cop_path_for_raw(raw_path: Path) -> Path:
    return COP_DIR / f"{raw_path.stem}_cop.csv"


def parse_pressure_matrix(raw: str) -> np.ndarray:
    matrix = np.asarray(json.loads(raw), dtype=np.float32)
    return matrix


def load_subject_arrays(raw_path: Path) -> dict[str, np.ndarray]:
    raw_df = pd.read_csv(
        raw_path,
        usecols=["timestamp_ms", "pressure_matrix_json", *IMU_AXIS_COLUMNS],
    )
    cop_df = pd.read_csv(
        cop_path_for_raw(raw_path),
        usecols=["timestamp_ms", "cop_x", "cop_y", "cop_available", "cop_interpolated"],
    )
    df = raw_df.merge(cop_df, on="timestamp_ms", how="inner")
    df = df[df["cop_available"]].reset_index(drop=True)

    pressure = np.stack([parse_pressure_matrix(raw) for raw in df["pressure_matrix_json"]]).astype(np.float32)
    pressure_max = float(np.max(pressure))
    if pressure_max > 0:
        pressure /= pressure_max

    imu = df[IMU_AXIS_COLUMNS].to_numpy(dtype=np.float32)
    imu_mean = imu.mean(axis=0, keepdims=True)
    imu_std = imu.std(axis=0, keepdims=True) + 1e-6
    imu = (imu - imu_mean) / imu_std

    target = df[["cop_x", "cop_y"]].to_numpy(dtype=np.float32)
    timestamps = df["timestamp_ms"].to_numpy(dtype=np.int64)
    interpolated = df["cop_interpolated"].to_numpy(dtype=bool)

    return {
        "subject": np.asarray([subject_name(raw_path)] * len(df)),
        "timestamp_ms": timestamps,
        "pressure": pressure,
        "imu": imu,
        "target": target,
        "cop_interpolated": interpolated,
    }


def build_windows(
    pressure: np.ndarray,
    imu: np.ndarray,
    target: np.ndarray,
    timestamps: np.ndarray,
    subject_labels: np.ndarray,
    interpolated: np.ndarray,
    seq_len: int,
) -> dict[str, np.ndarray]:
    pad = seq_len // 2
    imu_padded = np.pad(imu, ((pad, pad), (0, 0)), mode="edge")
    sequences = np.stack([imu_padded[i : i + seq_len] for i in range(len(imu))]).astype(np.float32)
    return {
        "pressure": pressure[:, None, :, :].astype(np.float32),
        "imu_seq": sequences,
        "target": target.astype(np.float32),
        "timestamp_ms": timestamps,
        "subject": subject_labels,
        "cop_interpolated": interpolated,
    }


@dataclass
class SplitData:
    pressure: np.ndarray
    imu_seq: np.ndarray
    target: np.ndarray
    timestamp_ms: np.ndarray
    subject: np.ndarray
    cop_interpolated: np.ndarray


class CoPDataset(Dataset):
    def __init__(self, split: SplitData, modality: str, target_mean: np.ndarray, target_std: np.ndarray) -> None:
        self.pressure = torch.from_numpy(split.pressure)
        self.imu_seq = torch.from_numpy(split.imu_seq)
        self.target = torch.from_numpy(((split.target - target_mean) / target_std).astype(np.float32))
        self.modality = modality

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {"target": self.target[idx]}
        if self.modality in {"pressure", "fusion"}:
            item["pressure"] = self.pressure[idx]
        if self.modality in {"imu", "fusion"}:
            item["imu_seq"] = self.imu_seq[idx]
        return item


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.skip = nn.Identity()
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.act(x + identity)
        return x


class PressureEncoder(nn.Module):
    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 96, stride=2),
            ResidualBlock(96, 128, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class IMUTransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, model_dim: int = 64, num_heads: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return x


class PressureNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = PressureEncoder(out_dim=128)
        self.head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, pressure: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(pressure))


class IMUNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = IMUTransformerEncoder(input_dim=input_dim, model_dim=64, num_heads=4, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, imu_seq: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(imu_seq))


class FusionNet(nn.Module):
    def __init__(self, imu_input_dim: int) -> None:
        super().__init__()
        self.pressure_encoder = PressureEncoder(out_dim=128)
        self.imu_encoder = IMUTransformerEncoder(input_dim=imu_input_dim, model_dim=64, num_heads=4, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, pressure: torch.Tensor, imu_seq: torch.Tensor) -> torch.Tensor:
        p = self.pressure_encoder(pressure)
        i = self.imu_encoder(imu_seq)
        return self.head(torch.cat([p, i], dim=1))


def build_model(modality: str, imu_input_dim: int) -> nn.Module:
    if modality == "pressure":
        return PressureNet()
    if modality == "imu":
        return IMUNet(input_dim=imu_input_dim)
    if modality == "fusion":
        return FusionNet(imu_input_dim=imu_input_dim)
    raise ValueError(f"Unknown modality: {modality}")


def run_model(model: nn.Module, batch: dict[str, torch.Tensor], modality: str) -> torch.Tensor:
    if modality == "pressure":
        return model(batch["pressure"])
    if modality == "imu":
        return model(batch["imu_seq"])
    if modality == "fusion":
        return model(batch["pressure"], batch["imu_seq"])
    raise ValueError(modality)


def split_train_val(split: SplitData, val_fraction: float = 0.1) -> tuple[SplitData, SplitData]:
    n = len(split.target)
    val_n = max(1, int(n * val_fraction))
    train_n = n - val_n
    train_slice = slice(0, train_n)
    val_slice = slice(train_n, n)

    def take(s: slice) -> SplitData:
        return SplitData(
            pressure=split.pressure[s],
            imu_seq=split.imu_seq[s],
            target=split.target[s],
            timestamp_ms=split.timestamp_ms[s],
            subject=split.subject[s],
            cop_interpolated=split.cop_interpolated[s],
        )

    return take(train_slice), take(val_slice)


def concat_splits(splits: list[SplitData]) -> SplitData:
    return SplitData(
        pressure=np.concatenate([s.pressure for s in splits], axis=0),
        imu_seq=np.concatenate([s.imu_seq for s in splits], axis=0),
        target=np.concatenate([s.target for s in splits], axis=0),
        timestamp_ms=np.concatenate([s.timestamp_ms for s in splits], axis=0),
        subject=np.concatenate([s.subject for s in splits], axis=0),
        cop_interpolated=np.concatenate([s.cop_interpolated for s in splits], axis=0),
    )


def make_loader(split: SplitData, modality: str, target_mean: np.ndarray, target_std: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = CoPDataset(split=split, modality=modality, target_mean=target_mean, target_std=target_std)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_one_model(
    modality: str,
    train_split: SplitData,
    val_split: SplitData,
    test_split: SplitData,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> tuple[nn.Module, np.ndarray]:
    batch_size = 256
    epochs = 12
    patience = 3
    lr = 1e-3

    model = build_model(modality=modality, imu_input_dim=train_split.imu_seq.shape[-1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    train_loader = make_loader(train_split, modality, target_mean, target_std, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_split, modality, target_mean, target_std, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(test_split, modality, target_mean, target_std, batch_size=batch_size, shuffle=False)

    best_state = None
    best_val = math.inf
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            pred = run_model(model, batch, modality)
            loss = criterion(pred, batch["target"])
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(batch["target"])

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                pred = run_model(model, batch, modality)
                loss = criterion(pred, batch["target"])
                val_loss += float(loss.item()) * len(batch["target"])

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        print(f"[{modality}] epoch {epoch:02d} train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    preds = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            pred = run_model(model, batch, modality).cpu().numpy()
            preds.append(pred)
    pred_norm = np.concatenate(preds, axis=0)
    pred = pred_norm * target_std + target_mean
    return model, pred.astype(np.float32)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    distance = np.linalg.norm(error, axis=1)
    error_cm = error.copy()
    error_cm[:, 0] *= X_CM_PER_CELL
    error_cm[:, 1] *= Y_CM_PER_CELL
    distance_cm = np.linalg.norm(error_cm, axis=1)
    return {
        "mae_x": float(mean_absolute_error(y_true[:, 0], y_pred[:, 0])),
        "mae_y": float(mean_absolute_error(y_true[:, 1], y_pred[:, 1])),
        "rmse_x": rmse(y_true[:, 0], y_pred[:, 0]),
        "rmse_y": rmse(y_true[:, 1], y_pred[:, 1]),
        "mae_2d": float(np.mean(distance)),
        "rmse_2d": float(np.sqrt(np.mean(distance**2))),
        "mae_x_cm": float(np.mean(np.abs(error_cm[:, 0]))),
        "mae_y_cm": float(np.mean(np.abs(error_cm[:, 1]))),
        "rmse_x_cm": float(np.sqrt(np.mean(error_cm[:, 0] ** 2))),
        "rmse_y_cm": float(np.sqrt(np.mean(error_cm[:, 1] ** 2))),
        "mae_2d_cm": float(np.mean(distance_cm)),
        "rmse_2d_cm": float(np.sqrt(np.mean(distance_cm**2))),
        "r2_x": float(r2_score(y_true[:, 0], y_pred[:, 0])),
        "r2_y": float(r2_score(y_true[:, 1], y_pred[:, 1])),
    }


def main() -> None:
    set_seed(SEED)
    raw_files = sorted(RAW_DIR.glob("*.csv"))
    train_files = raw_files[:-1]
    test_file = raw_files[-1]

    print("Train subjects:", [subject_name(path) for path in train_files])
    print("Held-out subject:", subject_name(test_file))

    seq_len = 9
    subject_splits: dict[Path, SplitData] = {}
    for raw_path in raw_files:
        print(f"Loading {raw_path.name} ...")
        arrays = load_subject_arrays(raw_path)
        windows = build_windows(
            pressure=arrays["pressure"],
            imu=arrays["imu"],
            target=arrays["target"],
            timestamps=arrays["timestamp_ms"],
            subject_labels=arrays["subject"],
            interpolated=arrays["cop_interpolated"],
            seq_len=seq_len,
        )
        subject_splits[raw_path] = SplitData(**windows)

    train_subject_splits = [subject_splits[path] for path in train_files]
    test_split = subject_splits[test_file]

    inner_train_parts = []
    inner_val_parts = []
    for split in train_subject_splits:
        train_part, val_part = split_train_val(split, val_fraction=0.1)
        inner_train_parts.append(train_part)
        inner_val_parts.append(val_part)

    train_split = concat_splits(inner_train_parts)
    val_split = concat_splits(inner_val_parts)
    target_mean = train_split.target.mean(axis=0, keepdims=True).astype(np.float32)
    target_std = (train_split.target.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    predictions = pd.DataFrame(
        {
            "timestamp_ms": test_split.timestamp_ms,
            "subject": test_split.subject,
            "cop_interpolated": test_split.cop_interpolated,
            "cop_x": test_split.target[:, 0],
            "cop_y": test_split.target[:, 1],
        }
    )

    for modality in ("pressure", "imu", "fusion"):
        print(f"\nTraining deep {modality} model")
        model, y_pred = train_one_model(
            modality=modality,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            target_mean=target_mean,
            target_std=target_std,
        )
        metrics = evaluate_predictions(test_split.target, y_pred)
        metrics_rows.append({"modality": modality, **metrics})

        predictions[f"{modality}_pred_x"] = y_pred[:, 0]
        predictions[f"{modality}_pred_y"] = y_pred[:, 1]
        dx_cm = (y_pred[:, 0] - test_split.target[:, 0]) * X_CM_PER_CELL
        dy_cm = (y_pred[:, 1] - test_split.target[:, 1]) * Y_CM_PER_CELL
        predictions[f"{modality}_error_2d_cm"] = np.sqrt(dx_cm**2 + dy_cm**2)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "modality": modality,
                "target_mean": target_mean,
                "target_std": target_std,
                "seq_len": seq_len,
            },
            MODELS_DIR / f"{modality}_deep_cop_model.pt",
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("mae_2d_cm").reset_index(drop=True)
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    predictions.to_csv(PREDICTIONS_DIR / f"{subject_name(test_file)}_predictions.csv", index=False)

    summary = {
        "train_subjects": [subject_name(path) for path in train_files],
        "held_out_subject": subject_name(test_file),
        "n_train_samples": int(len(train_split.target)),
        "n_val_samples": int(len(val_split.target)),
        "n_test_samples": int(len(test_split.target)),
        "seq_len": seq_len,
        "device": str(DEVICE),
        "metrics": metrics_df.to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(metrics_df.to_string(index=False))
    print()
    print(f"Saved metrics to {OUTPUT_DIR / 'metrics.csv'}")
    print(f"Saved predictions to {PREDICTIONS_DIR / f'{subject_name(test_file)}_predictions.csv'}")


if __name__ == "__main__":
    main()
