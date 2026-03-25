#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec


IMU_COLUMNS = {
    "S1": {
        "acc": ["IMU_S1_acc_x", "IMU_S1_acc_y", "IMU_S1_acc_z"],
        "gyro": ["IMU_S1_gyro_x", "IMU_S1_gyro_y", "IMU_S1_gyro_z"],
    },
    "S2": {
        "acc": ["IMU_S2_acc_x", "IMU_S2_acc_y", "IMU_S2_acc_z"],
        "gyro": ["IMU_S2_gyro_x", "IMU_S2_gyro_y", "IMU_S2_gyro_z"],
    },
}


def nullable_path(value: str) -> Path | None:
    if value.strip().lower() in {"", "none"}:
        return None
    return Path(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate pressure heatmaps with IMU traces for CoP CSV files."
    )
    parser.add_argument(
        "--input",
        default="Data/*.csv",
        help="Input CSV path or glob pattern. Default: Data/*.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="animations",
        help="Directory where animation files will be written.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Output animation frames per second.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="Use every Nth row to keep rendering manageable.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Output DPI for saved animations.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional hard cap on rendered frames per file.",
    )
    parser.add_argument(
        "--format",
        choices=("mp4", "gif", "none"),
        default="mp4",
        help="Output format. Use 'none' to export CoP CSVs without rendering video.",
    )
    parser.add_argument(
        "--cop-output-dir",
        type=nullable_path,
        default=Path("cop_exports"),
        help="Directory for per-file CoP CSV exports. Use 'none' to disable.",
    )
    parser.add_argument(
        "--cop-min-pressure-sum",
        type=float,
        default=1.0,
        help="Treat frames below this pressure sum as missing CoP.",
    )
    parser.add_argument(
        "--cop-max-gap",
        type=int,
        default=3,
        help="Only fill CoP dropouts up to this many consecutive frames.",
    )
    parser.add_argument(
        "--cop-smooth-window",
        type=int,
        default=5,
        help="Centered rolling window used to smooth CoP coordinates.",
    )
    return parser.parse_args()


def resolve_inputs(pattern: str) -> list[Path]:
    path = Path(pattern)
    if path.exists():
        return [path]
    return sorted(Path().glob(pattern))


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["pressure_rows"] = df["pressure_rows"].astype(int)
    df["pressure_cols"] = df["pressure_cols"].astype(int)
    df["timestamp_s"] = (df["timestamp_ms"] - df["timestamp_ms"].iloc[0]) / 1000.0

    for sensor, groups in IMU_COLUMNS.items():
        for group_name, cols in groups.items():
            values = df[cols].astype(float).to_numpy()
            df[f"{sensor}_{group_name}_mag"] = np.linalg.norm(values, axis=1)

    return df


def pressure_matrix_from_json(raw: str, rows: int, cols: int) -> np.ndarray:
    matrix = np.asarray(json.loads(raw), dtype=float)
    if matrix.shape != (rows, cols):
        raise ValueError(f"Expected pressure shape {(rows, cols)}, got {matrix.shape}")
    return matrix


def compute_center_of_pressure(matrix: np.ndarray) -> tuple[float, float] | tuple[None, None]:
    total = float(matrix.sum())
    if total <= 0:
        return None, None

    y_index, x_index = np.indices(matrix.shape)
    center_y = float((y_index * matrix).sum() / total)
    center_x = float((x_index * matrix).sum() / total)
    return center_x, center_y


def compute_cop_dataframe(
    df: pd.DataFrame,
    min_pressure_sum: float,
    max_gap: int,
    smooth_window: int,
) -> pd.DataFrame:
    raw_x: list[float] = []
    raw_y: list[float] = []
    valid: list[bool] = []

    for row in df.itertuples(index=False):
        rows = int(row.pressure_rows)
        cols = int(row.pressure_cols)
        matrix = pressure_matrix_from_json(row.pressure_matrix_json, rows, cols)
        pressure_sum = float(matrix.sum())
        if pressure_sum <= min_pressure_sum:
            raw_x.append(np.nan)
            raw_y.append(np.nan)
            valid.append(False)
            continue

        cop_x, cop_y = compute_center_of_pressure(matrix)
        raw_x.append(np.nan if cop_x is None else cop_x)
        raw_y.append(np.nan if cop_y is None else cop_y)
        valid.append(cop_x is not None and cop_y is not None)

    cop = pd.DataFrame(
        {
            "timestamp_ms": df["timestamp_ms"],
            "timestamp_s": df["timestamp_s"],
            "pressure_frame_index": df["pressure_frame_index"],
            "pressure_sum": df["pressure_sum"],
            "activity_label_text": df["activity_label_text"],
            "cop_x_raw": raw_x,
            "cop_y_raw": raw_y,
            "cop_valid_raw": valid,
        }
    )

    interpolate_limit = max(max_gap, 0)
    cop["cop_x_filled"] = cop["cop_x_raw"].interpolate(limit=interpolate_limit, limit_direction="both")
    cop["cop_y_filled"] = cop["cop_y_raw"].interpolate(limit=interpolate_limit, limit_direction="both")

    window = max(int(smooth_window), 1)
    cop["cop_x"] = (
        cop["cop_x_filled"].rolling(window=window, center=True, min_periods=1).mean()
    )
    cop["cop_y"] = (
        cop["cop_y_filled"].rolling(window=window, center=True, min_periods=1).mean()
    )
    cop["cop_interpolated"] = (~cop["cop_valid_raw"]) & cop["cop_x_filled"].notna() & cop["cop_y_filled"].notna()
    cop["cop_available"] = cop["cop_x"].notna() & cop["cop_y"].notna()
    return cop


def export_cop_csv(path: Path, cop_df: pd.DataFrame, output_dir: Path | None) -> Path | None:
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}_cop.csv"
    cop_df.to_csv(output_path, index=False)
    return output_path


def build_animation(
    path: Path,
    df: pd.DataFrame,
    output_dir: Path,
    fps: int,
    step: int,
    dpi: int,
    max_frames: int | None,
    output_format: str,
) -> Path:
    sampled = df.iloc[::step].reset_index(drop=True)
    if max_frames is not None:
        sampled = sampled.iloc[:max_frames].reset_index(drop=True)

    if sampled.empty:
        raise ValueError(f"No rows available after sampling for {path.name}")

    rows = int(sampled.loc[0, "pressure_rows"])
    cols = int(sampled.loc[0, "pressure_cols"])
    matrices = [pressure_matrix_from_json(raw, rows, cols) for raw in sampled["pressure_matrix_json"]]
    pressure_vmax = max(float(sampled["pressure_max"].max()), 1.0)

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = GridSpec(2, 3, figure=fig, width_ratios=[1.3, 1, 1])
    ax_pressure = fig.add_subplot(grid[:, 0])
    ax_acc = fig.add_subplot(grid[0, 1:])
    ax_gyro = fig.add_subplot(grid[1, 1:])

    heatmap = ax_pressure.imshow(
        matrices[0],
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=pressure_vmax,
    )
    cop_marker, = ax_pressure.plot([], [], marker="o", color="cyan", markersize=6)
    colorbar = fig.colorbar(heatmap, ax=ax_pressure, fraction=0.046, pad=0.04)
    colorbar.set_label("Pressure")

    times = sampled["timestamp_s"].to_numpy()
    acc_lines = {}
    gyro_lines = {}
    colors = {"S1": "#1f77b4", "S2": "#d62728"}

    for sensor in ("S1", "S2"):
        acc_values = sampled[f"{sensor}_acc_mag"].to_numpy()
        gyro_values = sampled[f"{sensor}_gyro_mag"].to_numpy()
        acc_lines[sensor], = ax_acc.plot(times, acc_values, label=f"{sensor} acc |mag|", color=colors[sensor])
        gyro_lines[sensor], = ax_gyro.plot(times, gyro_values, label=f"{sensor} gyro |mag|", color=colors[sensor])

    acc_cursor = ax_acc.axvline(times[0], color="black", linestyle="--", linewidth=1)
    gyro_cursor = ax_gyro.axvline(times[0], color="black", linestyle="--", linewidth=1)

    activity_text = ax_pressure.text(
        0.02,
        1.02,
        "",
        transform=ax_pressure.transAxes,
        fontsize=10,
        va="bottom",
    )

    ax_pressure.set_title(f"Pressure Map\n{path.name}")
    ax_pressure.set_xlabel("Columns")
    ax_pressure.set_ylabel("Rows")
    ax_acc.set_title("Acceleration Magnitude")
    ax_gyro.set_title("Gyroscope Magnitude")
    ax_acc.set_ylabel("|a|")
    ax_gyro.set_ylabel("|g|")
    ax_gyro.set_xlabel("Time (s)")

    ax_acc.legend(loc="upper right")
    ax_gyro.legend(loc="upper right")
    ax_acc.grid(alpha=0.3)
    ax_gyro.grid(alpha=0.3)
    ax_acc.set_xlim(times[0], times[-1] if times[-1] > times[0] else times[0] + 1e-6)
    ax_gyro.set_xlim(times[0], times[-1] if times[-1] > times[0] else times[0] + 1e-6)

    acc_max = max(float(sampled["S1_acc_mag"].max()), float(sampled["S2_acc_mag"].max()), 1.0)
    gyro_max = max(float(sampled["S1_gyro_mag"].max()), float(sampled["S2_gyro_mag"].max()), 1.0)
    ax_acc.set_ylim(0, acc_max * 1.1)
    ax_gyro.set_ylim(0, gyro_max * 1.1)

    def update(frame_index: int):
        matrix = matrices[frame_index]
        heatmap.set_data(matrix)

        cop_x = sampled.loc[frame_index, "cop_x"]
        cop_y = sampled.loc[frame_index, "cop_y"]
        if pd.isna(cop_x) or pd.isna(cop_y):
            cop_marker.set_data([], [])
        else:
            cop_marker.set_data([cop_x], [cop_y])

        current_time = times[frame_index]
        acc_cursor.set_xdata([current_time, current_time])
        gyro_cursor.set_xdata([current_time, current_time])

        label = sampled.loc[frame_index, "activity_label_text"]
        dropout_note = " | filled dropout" if bool(sampled.loc[frame_index, "cop_interpolated"]) else ""
        activity_text.set_text(
            f"t={current_time:0.2f}s | frame={frame_index + 1}/{len(sampled)} | activity={label}{dropout_note}"
        )
        return heatmap, cop_marker, acc_cursor, gyro_cursor, activity_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(sampled),
        interval=1000 / max(fps, 1),
        blit=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".mp4" if output_format == "mp4" else ".gif"
    output_path = output_dir / f"{path.stem}{suffix}"
    if output_format == "mp4":
        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=3000)
    else:
        writer = PillowWriter(fps=fps)
    animation.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    inputs = resolve_inputs(args.input)
    if not inputs:
        raise SystemExit(f"No input files matched: {args.input}")

    output_dir = Path(args.output_dir)
    for path in inputs:
        df = load_dataframe(path)
        cop_df = compute_cop_dataframe(
            df=df,
            min_pressure_sum=args.cop_min_pressure_sum,
            max_gap=args.cop_max_gap,
            smooth_window=args.cop_smooth_window,
        )
        cop_output_path = export_cop_csv(path, cop_df, args.cop_output_dir)
        if args.format != "none":
            output_path = build_animation(
                path=path,
                df=df.join(
                    cop_df[
                        [
                            "cop_x_raw",
                            "cop_y_raw",
                            "cop_x",
                            "cop_y",
                            "cop_valid_raw",
                            "cop_interpolated",
                            "cop_available",
                        ]
                    ]
                ),
                output_dir=output_dir,
                fps=args.fps,
                step=max(args.step, 1),
                dpi=args.dpi,
                max_frames=args.max_frames,
                output_format=args.format,
            )
            print(f"Saved {output_path}")
        if cop_output_path is not None:
            print(f"Saved {cop_output_path}")


if __name__ == "__main__":
    main()
