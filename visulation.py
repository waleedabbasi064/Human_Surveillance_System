"""
Visualization utilities for SPARTA.

This script supports two modes:
1. ``metrics``: generate training/validation plots from metrics CSV files.
2. ``overlay``: draw pose keypoints + joints on a video and color them by
   anomaly prediction from a SPARTA scores CSV.

Run from the project root:
    python visulation.py

Edit the configuration block below before running.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODE = "overlay"  # "metrics" or "overlay"

# Existing metrics visualization inputs
DATASETS: Dict[str, Path] = {
    "sparta-c": Path("results-sparta-c/metrics.csv"),
    "sparta-f": Path("results-sparta-f/metrics.csv"),
}
METRICS_OUT_ROOT = Path("visualizations")

# Video overlay inputs
VIDEO_PATH = Path("/home/waleed64/Documents/Human_Centric_Anomaly_Detection_Agent/videos/surveillance video.mp4")
POSE_JSON_PATH = Path("/home/waleed64/Documents/Human_Centric_Anomaly_Detection_Agent/pose_outputs/01_0000.json")
SCORES_CSV_PATH = Path("/home/waleed64/Documents/Human_Centric_Anomaly_Detection_Agent/pose_outputs/sparta_f_scores.csv")
OVERLAY_OUTPUT_PATH = Path("/home/waleed64/Documents/Human_Centric_Anomaly_Detection_Agent/pose_outputs/overlay_sparta_f.mp4")

SEGMENT_LENGTH = 24
KEYPOINT_CONFIDENCE_THRESHOLD = 0.05
NORMAL_COLOR = (0, 255, 0)
ANOMALY_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
LINE_THICKNESS = 1
POINT_RADIUS = 2
FONT_SCALE = 0.42
DRAW_BBOX = False
DRAW_SCORE = False
DRAW_FRAME_INDEX = False

# Match the trusted connection order from PoseEstimationModel/pose_estimation.py
SKELETON_EDGES: List[Tuple[int, int]] = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6),
]


# Use a pleasant grid style; fall back gracefully if seaborn styles are absent.
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("ggplot")


# -----------------------------------------------------------------------------
# Metrics plotting utilities (kept intact / backward compatible)
# -----------------------------------------------------------------------------
def load_metrics(csv_path: Path) -> Tuple[pd.DataFrame, Optional[Dict[str, float]]]:
    """Load metrics, skipping malformed lines and separating the final_best row."""
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    df["epoch_num"] = pd.to_numeric(df["epoch"], errors="coerce")
    metric_rows = df.dropna(subset=["epoch_num"]).copy()
    metric_rows["epoch_num"] = metric_rows["epoch_num"].astype(int)

    best_mask = df["epoch"].astype(str).str.contains("final_best", case=False, na=False)
    best_row = df.loc[best_mask].iloc[0].to_dict() if best_mask.any() else None

    if best_row:
        for key, val in list(best_row.items()):
            if key == "epoch":
                continue
            try:
                best_row[key] = float(val)
            except (TypeError, ValueError):
                best_row[key] = None

    return metric_rows, best_row


def annotate_min(ax, df: pd.DataFrame, column: str, color: str, label: str) -> int:
    idx = df[column].idxmin()
    epoch = int(df.loc[idx, "epoch_num"])
    value = df.loc[idx, column]
    ax.scatter(epoch, value, color=color, zorder=5, s=45)
    ax.annotate(
        f"{label}\nmin @ {epoch}",
        xy=(epoch, value),
        xytext=(epoch + 1, value * 1.05),
        arrowprops=dict(arrowstyle="->", color=color, lw=1),
        fontsize=8,
        color=color,
    )
    return epoch


def savefig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_loss(df: pd.DataFrame, best: Optional[Dict[str, float]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(df["epoch_num"], df["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax.plot(df["epoch_num"], df["eval_loss_mean"], label="Eval Loss", color="#d62728", lw=2)
    min_epoch = annotate_min(ax, df, "eval_loss_mean", color="#d62728", label="Eval loss")
    ax.axvspan(min_epoch, df["epoch_num"].max(), color="#d62728", alpha=0.06, label="After eval-loss min")
    ax.set_title("Training vs Evaluation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    if best and best.get("eval_loss_mean") is not None:
        ax.axhline(best["eval_loss_mean"], color="#d62728", ls=":", lw=1, label="Best eval loss")
    savefig(fig, out_dir / "loss.png")


def plot_auc(df: pd.DataFrame, best: Optional[Dict[str, float]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(df["epoch_num"], df["auc_roc"], label="AUC-ROC", color="#9467bd", lw=2)
    ax.plot(df["epoch_num"], df["auc_pr"], label="AUC-PR", color="#2ca02c", lw=2)
    if best:
        if best.get("auc_roc") is not None:
            ax.axhline(best["auc_roc"], color="#9467bd", ls=":", lw=1)
        if best.get("auc_pr") is not None:
            ax.axhline(best["auc_pr"], color="#2ca02c", ls=":", lw=1)
    ax.set_title("Discrimination Power")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.55, 1.0)
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4)
    savefig(fig, out_dir / "auc.png")


def plot_eer(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4.4))
    ax1.plot(df["epoch_num"], df["eer"], color="#ff7f0e", lw=2, label="EER")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("EER", color="#ff7f0e")
    ax1.tick_params(axis="y", labelcolor="#ff7f0e")
    ax2 = ax1.twinx()
    ax2.plot(df["epoch_num"], df["eer_th"], color="#1f77b4", lw=2, ls="--", label="EER Threshold")
    ax2.set_ylabel("Threshold", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    fig.suptitle("Equal Error Rate & Threshold")
    fig.legend(loc="upper right", bbox_to_anchor=(0.92, 0.92))
    ax1.grid(True, ls="--", alpha=0.4)
    savefig(fig, out_dir / "eer_threshold.png")


def plot_lr_and_eval_loss(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4.4))
    ax1.plot(df["epoch_num"], df["lr"], color="#17becf", lw=2, label="Learning rate")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("LR", color="#17becf")
    ax1.tick_params(axis="y", labelcolor="#17becf")
    ax2 = ax1.twinx()
    ax2.plot(df["epoch_num"], df["eval_loss_mean"], color="#d62728", lw=2, ls="--", label="Eval loss")
    ax2.set_ylabel("Eval loss", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    fig.suptitle("Learning Rate Schedule vs Eval Loss")
    fig.legend(loc="upper right", bbox_to_anchor=(0.92, 0.9))
    ax1.grid(True, ls="--", alpha=0.4)
    savefig(fig, out_dir / "lr_vs_eval_loss.png")


def plot_fpr(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(df["epoch_num"], df["fpr_at_target_fnr"], color="#8c564b", lw=2, label="FPR @ target FNR")
    ax.set_title("False Positive Rate at Target FNR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FPR")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4)
    savefig(fig, out_dir / "fpr_at_target_fnr.png")


def plot_threshold_stability(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(df["epoch_num"], df["threshold_at_target_fnr"], color="#7f7f7f", lw=2)
    ax.set_title("Decision Threshold Stability (Target FNR)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Threshold")
    ax.grid(True, ls="--", alpha=0.4)
    savefig(fig, out_dir / "threshold_stability.png")


def plot_loss_vs_auc(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    scatter = ax.scatter(
        df["eval_loss_mean"],
        df["auc_roc"],
        c=df["epoch_num"],
        cmap="viridis",
        s=40,
        edgecolor="k",
        linewidth=0.4,
    )
    ax.set_title("Eval Loss vs AUC-ROC (color = epoch)")
    ax.set_xlabel("Eval loss")
    ax.set_ylabel("AUC-ROC")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Epoch")
    ax.grid(True, ls="--", alpha=0.4)
    savefig(fig, out_dir / "loss_vs_auc.png")


def generate_plots(name: str, csv_path: Path, out_root: Path) -> None:
    df, best = load_metrics(csv_path)
    out_dir = out_root / name
    plot_loss(df, best, out_dir)
    plot_auc(df, best, out_dir)
    plot_eer(df, out_dir)
    plot_lr_and_eval_loss(df, out_dir)
    plot_fpr(df, out_dir)
    plot_threshold_stability(df, out_dir)
    plot_loss_vs_auc(df, out_dir)


# -----------------------------------------------------------------------------
# Video overlay utilities
# -----------------------------------------------------------------------------
def normalize_person_id(raw_value) -> str:
    text = str(raw_value).strip()
    if text.endswith('.0'):
        text = text[:-2]
    return text


def load_pose_json(json_path: Path) -> Dict[int, List[Tuple[str, Dict[str, object]]]]:
    with json_path.open('r', encoding='utf-8') as handle:
        raw = json.load(handle)

    frame_map: Dict[int, List[Tuple[str, Dict[str, object]]]] = {}
    if not isinstance(raw, dict):
        return frame_map

    for outer_key, outer_val in raw.items():
        if not isinstance(outer_val, dict):
            continue

        # Person-centric format: person_id -> frame_id -> payload
        if outer_val and all(isinstance(v, dict) and 'keypoints' in v for v in outer_val.values()):
            person_id = normalize_person_id(outer_key)
            for frame_key, payload in outer_val.items():
                try:
                    frame_idx = int(frame_key)
                except ValueError:
                    continue
                frame_map.setdefault(frame_idx, []).append((person_id, payload))
            continue

        # Frame-centric fallback: frame_id -> person_id -> payload
        try:
            frame_idx = int(outer_key)
        except ValueError:
            continue
        for person_id, payload in outer_val.items():
            if isinstance(payload, dict) and 'keypoints' in payload:
                frame_map.setdefault(frame_idx, []).append((normalize_person_id(person_id), payload))

    return frame_map


def load_prediction_ranges(csv_path: Path, segment_length: int) -> Dict[str, List[Dict[str, float]]]:
    df = pd.read_csv(csv_path)
    required = {"person_id", "start_frame", "predicted", "score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    ranges: Dict[str, List[Dict[str, float]]] = {}
    for _, row in df.iterrows():
        person_id = normalize_person_id(row["person_id"])
        if person_id == "" or person_id.lower() == "nan":
            continue
        try:
            start_frame = int(float(row["start_frame"]))
            score = float(row["score"])
            predicted = int(float(row["predicted"]))
        except (TypeError, ValueError):
            continue

        ranges.setdefault(person_id, []).append(
            {
                "start": start_frame,
                "end": start_frame + segment_length - 1,
                "predicted": 1 if predicted >= 1 else 0,
                "score": score,
            }
        )

    for person_id in ranges:
        ranges[person_id].sort(key=lambda item: item["start"])
    return ranges


def active_prediction_for_frame(person_id: str, frame_idx: int, prediction_ranges: Dict[str, List[Dict[str, float]]]) -> Tuple[int, Optional[float]]:
    person_ranges = prediction_ranges.get(normalize_person_id(person_id), [])
    active = [item for item in person_ranges if item["start"] <= frame_idx <= item["end"]]
    if not active:
        return 0, None

    predicted = 1 if any(item["predicted"] >= 1 for item in active) else 0
    score = max(item["score"] for item in active)
    return predicted, score


def reshape_keypoints(flat_keypoints: List[float]) -> List[Tuple[float, float, float]]:
    if len(flat_keypoints) % 3 != 0:
        return []
    return [tuple(flat_keypoints[i:i + 3]) for i in range(0, len(flat_keypoints), 3)]


def visible_points(keypoints: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    return [(x, y, conf) for x, y, conf in keypoints if conf >= KEYPOINT_CONFIDENCE_THRESHOLD]


def label_anchor_from_keypoints(keypoints: List[Tuple[float, float, float]]) -> Optional[Tuple[int, int]]:
    visible = visible_points(keypoints)
    if not visible:
        return None

    preferred_indices = [0, 1, 2, 5, 6]
    for idx in preferred_indices:
        if idx < len(keypoints):
            x, y, conf = keypoints[idx]
            if conf >= KEYPOINT_CONFIDENCE_THRESHOLD:
                return int(x), max(14, int(y) - 8)

    xs = [x for x, _, _ in visible]
    ys = [y for _, y, _ in visible]
    return int(min(xs)), max(14, int(min(ys)) - 8)


def draw_pose_on_frame(
    frame,
    keypoints: List[Tuple[float, float, float]],
    person_id: str,
    predicted: int,
    score: Optional[float] = None,
) -> None:
    color = ANOMALY_COLOR if predicted >= 1 else NORMAL_COLOR

    for start_idx, end_idx in SKELETON_EDGES:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            x1, y1, s1 = keypoints[start_idx]
            x2, y2, s2 = keypoints[end_idx]
            if s1 >= KEYPOINT_CONFIDENCE_THRESHOLD and s2 >= KEYPOINT_CONFIDENCE_THRESHOLD:
                cv2.line(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    LINE_THICKNESS,
                    lineType=cv2.LINE_AA,
                )

    for x, y, conf in keypoints:
        if conf >= KEYPOINT_CONFIDENCE_THRESHOLD:
            cv2.circle(
                frame,
                (int(x), int(y)),
                POINT_RADIUS,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )

    if DRAW_BBOX:
        visible = visible_points(keypoints)
        if visible:
            xs = [x for x, _, _ in visible]
            ys = [y for _, y, _ in visible]
            pad = 8
            x1, y1, x2, y2 = int(min(xs) - pad), int(min(ys) - pad), int(max(xs) + pad), int(max(ys) + pad)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)

    text_pos = label_anchor_from_keypoints(keypoints)
    if text_pos is not None:
        label = f"ID:{person_id}"
        if DRAW_SCORE and score is not None:
            label += f" {score:.2f}"
        cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, 1, lineType=cv2.LINE_AA)


def overlay_predictions_on_video(
    video_path: Path,
    pose_json_path: Path,
    scores_csv_path: Path,
    output_path: Path,
    segment_length: int = SEGMENT_LENGTH,
) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not pose_json_path.exists():
        raise FileNotFoundError(f"Pose JSON not found: {pose_json_path}")
    if not scores_csv_path.exists():
        raise FileNotFoundError(f"Scores CSV not found: {scores_csv_path}")

    frame_map = load_pose_json(pose_json_path)
    if not frame_map:
        raise RuntimeError("Pose JSON is empty or could not be parsed.")
    prediction_ranges = load_prediction_ranges(scores_csv_path, segment_length)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for person_id, payload in frame_map.get(frame_id, []):
            flat_keypoints = payload.get("keypoints", [])
            keypoints = reshape_keypoints(flat_keypoints)
            if not keypoints:
                continue

            predicted, score = active_prediction_for_frame(person_id, frame_id, prediction_ranges)
            draw_pose_on_frame(frame, keypoints, person_id=person_id, predicted=predicted, score=score)

        if DRAW_FRAME_INDEX:
            cv2.putText(
                frame,
                f"Frame: {frame_id}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()
    print(f"[OK] Saved anomaly overlay video to {output_path}")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def main() -> None:
    if MODE == "metrics":
        for name, csv_path in DATASETS.items():
            if not csv_path.exists():
                print(f"[WARN] {csv_path} not found; skipping {name}.")
                continue
            generate_plots(name, csv_path, METRICS_OUT_ROOT)
            print(f"[OK] Saved plots for {name} to {METRICS_OUT_ROOT / name}")
        return

    if MODE == "overlay":
        overlay_predictions_on_video(
            video_path=VIDEO_PATH,
            pose_json_path=POSE_JSON_PATH,
            scores_csv_path=SCORES_CSV_PATH,
            output_path=OVERLAY_OUTPUT_PATH,
            segment_length=SEGMENT_LENGTH,
        )
        return

    raise ValueError(f"Unsupported MODE: {MODE}. Use 'metrics' or 'overlay'.")


if __name__ == "__main__":
    main()
