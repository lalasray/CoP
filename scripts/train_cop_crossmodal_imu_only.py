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
OUTPUT_DIR = Path("model_outputs_crossmodal")
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_CM_PER_CELL = 2.0
Y_CM_PER_CELL = 0.21
SEQ_LEN = 9
EMBED_DIM = 128
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
    return np.asarray(json.loads(raw), dtype=np.float32)


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
    pressure /= max(float(pressure.max()), 1.0)

    imu = df[IMU_AXIS_COLUMNS].to_numpy(dtype=np.float32)
    imu_mean = imu.mean(axis=0, keepdims=True)
    imu_std = imu.std(axis=0, keepdims=True) + 1e-6
    imu = (imu - imu_mean) / imu_std

    return {
        "subject": np.asarray([subject_name(raw_path)] * len(df)),
        "timestamp_ms": df["timestamp_ms"].to_numpy(dtype=np.int64),
        "pressure": pressure,
        "imu": imu,
        "target": df[["cop_x", "cop_y"]].to_numpy(dtype=np.float32),
        "cop_interpolated": df["cop_interpolated"].to_numpy(dtype=bool),
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


class CrossModalDataset(Dataset):
    def __init__(self, split: SplitData, target_mean: np.ndarray, target_std: np.ndarray) -> None:
        self.pressure = torch.from_numpy(split.pressure)
        self.imu_seq = torch.from_numpy(split.imu_seq)
        self.target = torch.from_numpy(((split.target - target_mean) / target_std).astype(np.float32))

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "pressure": self.pressure[idx],
            "imu_seq": self.imu_seq[idx],
            "target": self.target[idx],
        }


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
        return self.act(x + identity)


class PressureEncoder(nn.Module):
    def __init__(self, out_dim: int = EMBED_DIM) -> None:
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


class IMUEncoder(nn.Module):
    def __init__(self, input_dim: int, out_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 64)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.norm = nn.LayerNorm(64)
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.proj(x)


class RegressionHead(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def contrastive_loss(imu_emb: torch.Tensor, pressure_emb: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    imu_emb = nn.functional.normalize(imu_emb, dim=1)
    pressure_emb = nn.functional.normalize(pressure_emb, dim=1)
    logits = imu_emb @ pressure_emb.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (
        nn.functional.cross_entropy(logits, labels) +
        nn.functional.cross_entropy(logits.T, labels)
    )


def temporal_smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    if pred.size(0) < 2:
        return pred.new_tensor(0.0)
    delta = pred[1:] - pred[:-1]
    return delta.pow(2).mean()


def split_train_val(split: SplitData, val_fraction: float = 0.1) -> tuple[SplitData, SplitData]:
    n = len(split.target)
    val_n = max(1, int(n * val_fraction))
    train_n = n - val_n

    def take(start: int, end: int) -> SplitData:
        return SplitData(
            pressure=split.pressure[start:end],
            imu_seq=split.imu_seq[start:end],
            target=split.target[start:end],
            timestamp_ms=split.timestamp_ms[start:end],
            subject=split.subject[start:end],
            cop_interpolated=split.cop_interpolated[start:end],
        )

    return take(0, train_n), take(train_n, n)


def concat_splits(splits: list[SplitData]) -> SplitData:
    return SplitData(
        pressure=np.concatenate([s.pressure for s in splits], axis=0),
        imu_seq=np.concatenate([s.imu_seq for s in splits], axis=0),
        target=np.concatenate([s.target for s in splits], axis=0),
        timestamp_ms=np.concatenate([s.timestamp_ms for s in splits], axis=0),
        subject=np.concatenate([s.subject for s in splits], axis=0),
        cop_interpolated=np.concatenate([s.cop_interpolated for s in splits], axis=0),
    )


def make_loader(split: SplitData, target_mean: np.ndarray, target_std: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = CrossModalDataset(split=split, target_mean=target_mean, target_std=target_std)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


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
        "rmse_x": float(np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))),
        "rmse_y": float(np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))),
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


def predict_imu(
    imu_encoder: IMUEncoder,
    imu_head: RegressionHead,
    loader: DataLoader,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    imu_encoder.eval()
    imu_head.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            imu_seq = batch["imu_seq"].to(DEVICE)
            pred = imu_head(imu_encoder(imu_seq)).cpu().numpy()
            preds.append(pred)
    pred_norm = np.concatenate(preds, axis=0)
    return pred_norm * target_std + target_mean


def train_imu_student(
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> tuple[PressureEncoder, IMUEncoder, RegressionHead]:
    pressure_encoder = PressureEncoder().to(DEVICE)
    imu_encoder = IMUEncoder(input_dim=len(IMU_AXIS_COLUMNS)).to(DEVICE)
    imu_head = RegressionHead().to(DEVICE)
    optimizer = torch.optim.AdamW(
        list(pressure_encoder.parameters()) + list(imu_encoder.parameters()) + list(imu_head.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )
    regression_loss = nn.SmoothL1Loss()
    smoothness_weight = 0.05

    best_val = math.inf
    best_state = None
    patience = 4
    bad_epochs = 0

    for epoch in range(1, 13):
        imu_encoder.train()
        imu_head.train()
        pressure_encoder.train()
        for batch in train_loader:
            pressure = batch["pressure"].to(DEVICE)
            imu_seq = batch["imu_seq"].to(DEVICE)
            target = batch["target"].to(DEVICE)

            optimizer.zero_grad()
            pressure_emb = pressure_encoder(pressure)
            imu_emb = imu_encoder(imu_seq)
            imu_pred = imu_head(imu_emb)

            loss = (
                1.0 * regression_loss(imu_pred, target) +
                0.15 * contrastive_loss(imu_emb, pressure_emb) +
                smoothness_weight * temporal_smoothness_loss(imu_pred)
            )
            loss.backward()
            optimizer.step()

        imu_encoder.eval()
        imu_head.eval()
        pressure_encoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pressure = batch["pressure"].to(DEVICE)
                imu_seq = batch["imu_seq"].to(DEVICE)
                target = batch["target"].to(DEVICE)
                pressure_emb = pressure_encoder(pressure)
                imu_emb = imu_encoder(imu_seq)
                imu_pred = imu_head(imu_emb)
                loss = (
                    1.0 * regression_loss(imu_pred, target) +
                    0.15 * contrastive_loss(imu_emb, pressure_emb) +
                    smoothness_weight * temporal_smoothness_loss(imu_pred)
                )
                val_loss += float(loss.item()) * len(target)
        val_loss /= len(val_loader.dataset)
        print(f"[student] epoch {epoch:02d} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "pressure_encoder": deepcopy(pressure_encoder.state_dict()),
                "imu_encoder": deepcopy(imu_encoder.state_dict()),
                "imu_head": deepcopy(imu_head.state_dict()),
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    pressure_encoder.load_state_dict(best_state["pressure_encoder"])
    imu_encoder.load_state_dict(best_state["imu_encoder"])
    imu_head.load_state_dict(best_state["imu_head"])
    return pressure_encoder, imu_encoder, imu_head


def main() -> None:
    set_seed(SEED)
    raw_files = sorted(RAW_DIR.glob("*.csv"))
    train_files = raw_files[:-1]
    test_file = raw_files[-1]

    print("Train subjects:", [subject_name(path) for path in train_files])
    print("Held-out subject:", subject_name(test_file))
    print("Device:", DEVICE)

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
            seq_len=SEQ_LEN,
        )
        subject_splits[raw_path] = SplitData(**windows)

    train_parts = []
    val_parts = []
    for raw_path in train_files:
        train_part, val_part = split_train_val(subject_splits[raw_path], val_fraction=0.1)
        train_parts.append(train_part)
        val_parts.append(val_part)

    train_split = concat_splits(train_parts)
    val_split = concat_splits(val_parts)
    test_split = subject_splits[test_file]

    target_mean = train_split.target.mean(axis=0, keepdims=True).astype(np.float32)
    target_std = (train_split.target.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)

    train_loader = make_loader(train_split, target_mean, target_std, batch_size=256, shuffle=True)
    val_loader = make_loader(val_split, target_mean, target_std, batch_size=256, shuffle=False)
    test_loader = make_loader(test_split, target_mean, target_std, batch_size=256, shuffle=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    pressure_encoder, imu_encoder, imu_head = train_imu_student(train_loader, val_loader)
    student_pred = predict_imu(imu_encoder, imu_head, test_loader, target_mean, target_std)

    metrics_df = pd.DataFrame(
        [
            {"branch": "imu_crossmodal_eval_imu_only", **evaluate_predictions(test_split.target, student_pred)},
        ]
    ).sort_values("mae_2d_cm").reset_index(drop=True)
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    predictions = pd.DataFrame(
        {
            "timestamp_ms": test_split.timestamp_ms,
            "subject": test_split.subject,
            "cop_interpolated": test_split.cop_interpolated,
            "cop_x": test_split.target[:, 0],
            "cop_y": test_split.target[:, 1],
            "student_pred_x": student_pred[:, 0],
            "student_pred_y": student_pred[:, 1],
        }
    )
    predictions.to_csv(PREDICTIONS_DIR / f"{subject_name(test_file)}_predictions.csv", index=False)

    torch.save(
        {
            "pressure_encoder": pressure_encoder.state_dict(),
            "target_mean": target_mean,
            "target_std": target_std,
            "seq_len": SEQ_LEN,
        },
        MODELS_DIR / "pressure_encoder_for_alignment.pt",
    )
    torch.save(
        {
            "imu_encoder": imu_encoder.state_dict(),
            "imu_head": imu_head.state_dict(),
            "target_mean": target_mean,
            "target_std": target_std,
            "seq_len": SEQ_LEN,
        },
        MODELS_DIR / "imu_student.pt",
    )

    summary = {
        "train_subjects": [subject_name(path) for path in train_files],
        "held_out_subject": subject_name(test_file),
        "device": str(DEVICE),
        "architecture": "imu-only deployment model trained with analytic CoP supervision plus cross-modal pressure/imu contrastive alignment",
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
