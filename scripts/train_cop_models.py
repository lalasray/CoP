#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RAW_DIR = Path("Data")
COP_DIR = Path("cop_exports_all")
OUTPUT_DIR = Path("model_outputs")
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
X_CM_PER_CELL = 2.0
Y_CM_PER_CELL = 0.21

TARGET_COLUMNS = ["cop_x", "cop_y"]
PRESSURE_SUMMARY_COLUMNS = [
    "pressure_sum",
    "pressure_mean",
    "pressure_max",
    "pressure_nonzero_count",
]
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


def subject_name(raw_path: Path) -> str:
    return raw_path.stem.split("_")[-1]


def cop_path_for_raw(raw_path: Path) -> Path:
    return COP_DIR / f"{raw_path.stem}_cop.csv"


def pressure_features_from_json(series: pd.Series) -> pd.DataFrame:
    row_profiles = []
    col_profiles = []
    for raw in series:
        matrix = np.asarray(json.loads(raw), dtype=np.float32)
        row_profiles.append(matrix.sum(axis=1))
        col_profiles.append(matrix.sum(axis=0))

    row_array = np.stack(row_profiles)
    col_array = np.stack(col_profiles)
    row_cols = [f"pressure_row_sum_{i:02d}" for i in range(row_array.shape[1])]
    col_cols = [f"pressure_col_sum_{i:02d}" for i in range(col_array.shape[1])]
    return pd.DataFrame(
        np.concatenate([row_array, col_array], axis=1),
        columns=row_cols + col_cols,
    )


def imu_features_with_history(df: pd.DataFrame) -> pd.DataFrame:
    base = df[IMU_AXIS_COLUMNS].astype(np.float32).copy()
    base["IMU_S1_acc_mag"] = np.linalg.norm(df[["IMU_S1_acc_x", "IMU_S1_acc_y", "IMU_S1_acc_z"]], axis=1)
    base["IMU_S1_gyro_mag"] = np.linalg.norm(df[["IMU_S1_gyro_x", "IMU_S1_gyro_y", "IMU_S1_gyro_z"]], axis=1)
    base["IMU_S2_acc_mag"] = np.linalg.norm(df[["IMU_S2_acc_x", "IMU_S2_acc_y", "IMU_S2_acc_z"]], axis=1)
    base["IMU_S2_gyro_mag"] = np.linalg.norm(df[["IMU_S2_gyro_x", "IMU_S2_gyro_y", "IMU_S2_gyro_z"]], axis=1)

    features: dict[str, pd.Series] = {}
    for col in base.columns:
        series = base[col]
        features[col] = series
        for lag in (1, 2, 3):
            features[f"{col}_lag{lag}"] = series.shift(lag)
        features[f"{col}_roll_mean_5"] = series.rolling(window=5, min_periods=1).mean()
        features[f"{col}_roll_std_5"] = series.rolling(window=5, min_periods=1).std().fillna(0.0)

    return pd.DataFrame(features).bfill().fillna(0.0)


def load_subject_frame(raw_path: Path) -> pd.DataFrame:
    cop_path = cop_path_for_raw(raw_path)
    raw_df = pd.read_csv(
        raw_path,
        usecols=[
            "timestamp_ms",
            "pressure_matrix_json",
            *PRESSURE_SUMMARY_COLUMNS,
            *IMU_AXIS_COLUMNS,
        ],
    )
    cop_df = pd.read_csv(
        cop_path,
        usecols=["timestamp_ms", "cop_x", "cop_y", "cop_available", "cop_interpolated"],
    )
    df = raw_df.merge(cop_df, on="timestamp_ms", how="inner")
    df = df[df["cop_available"]].reset_index(drop=True)
    df["subject"] = subject_name(raw_path)

    pressure_features = pressure_features_from_json(df["pressure_matrix_json"])
    pressure_features = pd.concat([pressure_features, df[PRESSURE_SUMMARY_COLUMNS].reset_index(drop=True)], axis=1)
    imu_features = imu_features_with_history(df)

    frame = pd.concat(
        [
            df[["timestamp_ms", "subject", "cop_interpolated"] + TARGET_COLUMNS].reset_index(drop=True),
            pressure_features.add_prefix("P_"),
            imu_features.add_prefix("I_"),
        ],
        axis=1,
    )
    return frame


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
        "mae_2d": float(np.mean(np.abs(distance))),
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
    raw_files = sorted(RAW_DIR.glob("*.csv"))
    if len(raw_files) < 2:
        raise SystemExit("Need at least two CSV files for train/test splitting.")

    train_files = raw_files[:-1]
    test_file = raw_files[-1]

    print("Train subjects:", [subject_name(path) for path in train_files])
    print("Held-out subject:", subject_name(test_file))

    subject_frames = {}
    for raw_path in raw_files:
        print(f"Loading {raw_path.name} ...")
        subject_frames[raw_path] = load_subject_frame(raw_path)

    train_df = pd.concat([subject_frames[path] for path in train_files], ignore_index=True)
    test_df = subject_frames[test_file].reset_index(drop=True)

    pressure_columns = [col for col in train_df.columns if col.startswith("P_")]
    imu_columns = [col for col in train_df.columns if col.startswith("I_")]
    fusion_columns = pressure_columns + imu_columns

    X_train = {
        "pressure": train_df[pressure_columns].to_numpy(dtype=np.float32),
        "imu": train_df[imu_columns].to_numpy(dtype=np.float32),
        "fusion": train_df[fusion_columns].to_numpy(dtype=np.float32),
    }
    X_test = {
        "pressure": test_df[pressure_columns].to_numpy(dtype=np.float32),
        "imu": test_df[imu_columns].to_numpy(dtype=np.float32),
        "fusion": test_df[fusion_columns].to_numpy(dtype=np.float32),
    }
    y_train = train_df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COLUMNS].to_numpy(dtype=np.float32)

    models = {
        "pressure": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1.0)),
            ]
        ),
        "imu": RandomForestRegressor(
            n_estimators=120,
            max_depth=18,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=42,
        ),
        "fusion": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=3.0)),
            ]
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    predictions = test_df[["timestamp_ms", "subject", "cop_interpolated"] + TARGET_COLUMNS].copy()

    for name, model in models.items():
        print(f"Training {name} model ...")
        model.fit(X_train[name], y_train)
        y_pred = model.predict(X_test[name]).astype(np.float32)
        metrics = evaluate_predictions(y_test, y_pred)
        metrics_rows.append({"modality": name, **metrics})

        predictions[f"{name}_pred_x"] = y_pred[:, 0]
        predictions[f"{name}_pred_y"] = y_pred[:, 1]
        predictions[f"{name}_error_2d"] = np.linalg.norm(y_pred - y_test, axis=1)

        joblib.dump(model, MODELS_DIR / f"{name}_cop_model.joblib")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("mae_2d").reset_index(drop=True)
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    predictions.to_csv(PREDICTIONS_DIR / f"{subject_name(test_file)}_predictions.csv", index=False)

    summary = {
        "train_subjects": [subject_name(path) for path in train_files],
        "held_out_subject": subject_name(test_file),
        "n_train_samples": int(len(train_df)),
        "n_test_samples": int(len(test_df)),
        "n_pressure_features": int(len(pressure_columns)),
        "n_imu_features": int(len(imu_columns)),
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
