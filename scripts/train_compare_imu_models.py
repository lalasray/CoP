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
OUTPUT_DIR = Path("model_outputs_imu_compare")
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_CM_PER_CELL = 2.0
Y_CM_PER_CELL = 0.21
SEQ_LEN = 41
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


@dataclass
class SubjectRaw:
    subject: str
    timestamp_ms: np.ndarray
    pressure: np.ndarray
    imu: np.ndarray
    target: np.ndarray
    cop_interpolated: np.ndarray


@dataclass
class SplitData:
    pressure: np.ndarray
    imu_seq: np.ndarray
    target: np.ndarray
    timestamp_ms: np.ndarray
    subject: np.ndarray
    cop_interpolated: np.ndarray


def load_subject_raw(raw_path: Path) -> SubjectRaw:
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
    imu = df[IMU_AXIS_COLUMNS].to_numpy(dtype=np.float32)

    return SubjectRaw(
        subject=subject_name(raw_path),
        timestamp_ms=df["timestamp_ms"].to_numpy(dtype=np.int64),
        pressure=pressure,
        imu=imu,
        target=df[["cop_x", "cop_y"]].to_numpy(dtype=np.float32),
        cop_interpolated=df["cop_interpolated"].to_numpy(dtype=bool),
    )


def normalize_subject(raw: SubjectRaw, imu_mean: np.ndarray, imu_std: np.ndarray, pressure_scale: float) -> SubjectRaw:
    pressure = raw.pressure / max(float(pressure_scale), 1.0)
    imu = (raw.imu - imu_mean) / imu_std
    return SubjectRaw(
        subject=raw.subject,
        timestamp_ms=raw.timestamp_ms,
        pressure=pressure.astype(np.float32),
        imu=imu.astype(np.float32),
        target=raw.target.astype(np.float32),
        cop_interpolated=raw.cop_interpolated,
    )


def build_split(raw: SubjectRaw, seq_len: int) -> SplitData:
    pad = seq_len // 2
    imu_padded = np.pad(raw.imu, ((pad, pad), (0, 0)), mode="edge")
    imu_seq = np.stack([imu_padded[i : i + seq_len] for i in range(len(raw.imu))]).astype(np.float32)
    return SplitData(
        pressure=raw.pressure[:, None, :, :].astype(np.float32),
        imu_seq=imu_seq,
        target=raw.target.astype(np.float32),
        timestamp_ms=raw.timestamp_ms,
        subject=np.asarray([raw.subject] * len(raw.target)),
        cop_interpolated=raw.cop_interpolated,
    )


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


class MultiModalDataset(Dataset):
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
        if stride != 1 or in_ch != out_ch:
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
    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(24, 32),
            ResidualBlock(32, 48, stride=2),
            ResidualBlock(48, 64, stride=2),
            ResidualBlock(64, 96, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(96, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class IMUEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )
        self.proj = nn.Linear(128, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        x = self.proj(x)
        return self.norm(x)


class RegressionHead(nn.Module):
    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def contrastive_loss(imu_emb: torch.Tensor, pressure_emb: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    imu_emb = nn.functional.normalize(imu_emb, dim=1)
    pressure_emb = nn.functional.normalize(pressure_emb, dim=1)
    logits = imu_emb @ pressure_emb.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (
        nn.functional.cross_entropy(logits, labels) +
        nn.functional.cross_entropy(logits.T, labels)
    )


def temporal_smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    if pred.size(0) < 3:
        return pred.new_tensor(0.0)
    velocity = pred[1:] - pred[:-1]
    acceleration = velocity[1:] - velocity[:-1]
    return acceleration.pow(2).mean()


def make_loader(split: SplitData, target_mean: np.ndarray, target_std: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = MultiModalDataset(split=split, target_mean=target_mean, target_std=target_std)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


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
    head: RegressionHead,
    loader: DataLoader,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    imu_encoder.eval()
    head.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            imu_seq = batch["imu_seq"].to(DEVICE)
            pred = head(imu_encoder(imu_seq)).cpu().numpy()
            preds.append(pred)
    pred_norm = np.concatenate(preds, axis=0)
    return pred_norm * target_std + target_mean


def train_model(
    name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    use_crossmodal: bool,
) -> tuple[dict[str, nn.Module], float]:
    imu_encoder = IMUEncoder(input_dim=len(IMU_AXIS_COLUMNS)).to(DEVICE)
    pressure_encoder = PressureEncoder().to(DEVICE) if use_crossmodal else None
    head = RegressionHead().to(DEVICE)

    params = list(imu_encoder.parameters()) + list(head.parameters())
    if pressure_encoder is not None:
        params += list(pressure_encoder.parameters())

    optimizer = torch.optim.AdamW(params, lr=8e-4, weight_decay=1e-4)
    reg_loss = nn.SmoothL1Loss()
    best_val = math.inf
    best_state = None
    patience = 4
    bad_epochs = 0

    for epoch in range(1, 13):
        imu_encoder.train()
        head.train()
        if pressure_encoder is not None:
            pressure_encoder.train()
        for batch in train_loader:
            imu_seq = batch["imu_seq"].to(DEVICE)
            pressure = batch["pressure"].to(DEVICE)
            target = batch["target"].to(DEVICE)

            optimizer.zero_grad()
            imu_emb = imu_encoder(imu_seq)
            pred = head(imu_emb)
            loss = reg_loss(pred, target) + 0.05 * temporal_smoothness_loss(pred)
            if pressure_encoder is not None:
                pressure_emb = pressure_encoder(pressure)
                loss = loss + 0.10 * contrastive_loss(imu_emb, pressure_emb)
            loss.backward()
            optimizer.step()

        imu_encoder.eval()
        head.eval()
        if pressure_encoder is not None:
            pressure_encoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imu_seq = batch["imu_seq"].to(DEVICE)
                pressure = batch["pressure"].to(DEVICE)
                target = batch["target"].to(DEVICE)
                imu_emb = imu_encoder(imu_seq)
                pred = head(imu_emb)
                loss = reg_loss(pred, target) + 0.05 * temporal_smoothness_loss(pred)
                if pressure_encoder is not None:
                    pressure_emb = pressure_encoder(pressure)
                    loss = loss + 0.10 * contrastive_loss(imu_emb, pressure_emb)
                val_loss += float(loss.item()) * len(target)
        val_loss /= len(val_loader.dataset)
        print(f"[{name}] epoch {epoch:02d} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "imu_encoder": deepcopy(imu_encoder.state_dict()),
                "head": deepcopy(head.state_dict()),
                "pressure_encoder": deepcopy(pressure_encoder.state_dict()) if pressure_encoder is not None else None,
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    imu_encoder.load_state_dict(best_state["imu_encoder"])
    head.load_state_dict(best_state["head"])
    if pressure_encoder is not None and best_state["pressure_encoder"] is not None:
        pressure_encoder.load_state_dict(best_state["pressure_encoder"])
    return {"imu_encoder": imu_encoder, "head": head, "pressure_encoder": pressure_encoder}, best_val


def main() -> None:
    set_seed(SEED)
    raw_files = sorted(RAW_DIR.glob("*.csv"))
    train_files = raw_files[:-1]
    test_file = raw_files[-1]

    print("Train subjects:", [subject_name(p) for p in train_files])
    print("Held-out subject:", subject_name(test_file))
    print("Device:", DEVICE)

    raw_subjects = {path: load_subject_raw(path) for path in raw_files}
    train_imu = np.concatenate([raw_subjects[p].imu for p in train_files], axis=0)
    imu_mean = train_imu.mean(axis=0, keepdims=True).astype(np.float32)
    imu_std = (train_imu.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    pressure_scale = float(max(raw_subjects[p].pressure.max() for p in train_files))

    normalized = {path: normalize_subject(raw_subjects[path], imu_mean, imu_std, pressure_scale) for path in raw_files}
    subject_splits = {path: build_split(normalized[path], seq_len=SEQ_LEN) for path in raw_files}

    train_parts = []
    val_parts = []
    for path in train_files:
        tr, va = split_train_val(subject_splits[path], val_fraction=0.1)
        train_parts.append(tr)
        val_parts.append(va)

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

    results = []
    predictions = pd.DataFrame(
        {
            "timestamp_ms": test_split.timestamp_ms,
            "subject": test_split.subject,
            "cop_interpolated": test_split.cop_interpolated,
            "cop_x": test_split.target[:, 0],
            "cop_y": test_split.target[:, 1],
        }
    )

    for name, use_crossmodal in [("imu_only", False), ("imu_crossmodal", True)]:
        models, best_val = train_model(name, train_loader, val_loader, use_crossmodal=use_crossmodal)
        pred = predict_imu(models["imu_encoder"], models["head"], test_loader, target_mean, target_std)
        metrics = evaluate_predictions(test_split.target, pred)
        results.append({"model": name, "best_val_loss": best_val, **metrics})
        predictions[f"{name}_pred_x"] = pred[:, 0]
        predictions[f"{name}_pred_y"] = pred[:, 1]

        torch.save(
            {
                "imu_encoder": models["imu_encoder"].state_dict(),
                "head": models["head"].state_dict(),
                "target_mean": target_mean,
                "target_std": target_std,
                "imu_mean": imu_mean,
                "imu_std": imu_std,
                "seq_len": SEQ_LEN,
            },
            MODELS_DIR / f"{name}.pt",
        )

    metrics_df = pd.DataFrame(results).sort_values("mae_2d_cm").reset_index(drop=True)
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    predictions.to_csv(PREDICTIONS_DIR / f"{subject_name(test_file)}_predictions.csv", index=False)

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_subjects": [subject_name(p) for p in train_files],
                "held_out_subject": subject_name(test_file),
                "device": str(DEVICE),
                "seq_len": SEQ_LEN,
                "note": "Global train-only IMU normalization; matched IMU-only and crossmodal IMU inference comparison.",
                "metrics": metrics_df.to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    print()
    print(metrics_df.to_string(index=False))
    print()
    print(f"Saved metrics to {OUTPUT_DIR / 'metrics.csv'}")


if __name__ == "__main__":
    main()
