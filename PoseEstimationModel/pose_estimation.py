from __future__ import annotations

import csv
import json
import colorsys
import os
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import random
import re
import sys

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", "/tmp")

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.engine.results import Boxes


# ---------- Config helpers ----------

# COCO 17 Connections: Pairs of keypoint indices to connect
SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), # Legs/Hips
    (5, 11), (6, 12), (5, 6),                        # Torso
    (5, 7), (6, 8), (7, 9), (8, 10),                 # Arms
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),          # Head
    (3, 5), (4, 6)                                   # Ears to Shoulders
]

def _bbox_area_xyxy(bbox: list[float] | tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def _resize_frame_if_needed(frame: np.ndarray, runtime_cfg: Dict[str, Any]) -> tuple[np.ndarray, float, float]:
    target_w = int(runtime_cfg.get("resize_width", 0) or 0)
    target_h = int(runtime_cfg.get("resize_height", 0) or 0)
    if target_w <= 0 or target_h <= 0:
        max_w = int(runtime_cfg.get("max_frame_width", 0) or 0)
        if max_w > 0 and frame.shape[1] > max_w:
            scale = max_w / float(frame.shape[1])
            target_w = max_w
            target_h = max(1, int(round(frame.shape[0] * scale)))
    if target_w <= 0 or target_h <= 0:
        return frame, 1.0, 1.0
    resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    sx = target_w / float(frame.shape[1])
    sy = target_h / float(frame.shape[0])
    return resized, sx, sy


def _scale_bbox_xyxy(bbox: list[float] | tuple[float, float, float, float], sx: float, sy: float) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [float(x1) * sx, float(y1) * sy, float(x2) * sx, float(y2) * sy]


def _bbox_iou_xyxy(box_a: list[float] | tuple[float, float, float, float], box_b: list[float] | tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _open_video_capture(source: Any) -> cv2.VideoCapture:
    if isinstance(source, int):
        return cv2.VideoCapture(source)
    source_str = str(source)
    if source_str.startswith("__camera__:"):
        try:
            cam_idx = int(source_str.split(":", 1)[1])
        except Exception:
            cam_idx = 0
        return cv2.VideoCapture(cam_idx)
    cap = cv2.VideoCapture(source_str, cv2.CAP_FFMPEG)
    if cap.isOpened():
        return cap
    cap.release()
    return cv2.VideoCapture(source_str)


def _video_writer_fourcc(output_video_path: Path, configured_fourcc: str) -> int:
    if output_video_path.suffix.lower() == ".mp4":
        return cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter_fourcc(*configured_fourcc)


def _random_vibrant_bgr(rng: random.Random) -> tuple[int, int, int]:
    # Use HSV to avoid dark/greyish random RGBs; OpenCV uses BGR.
    h = rng.random()
    s = 0.75 + 0.25 * rng.random()
    v = 0.75 + 0.25 * rng.random()
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(b * 255), int(g * 255), int(r * 255))

@dataclass
class ModelPaths:
    config: Optional[Path] = None
    checkpoint: Optional[Path] = None
    weights: Optional[Path] = None


class Config:
    """YAML wrapper with path resolution and dynamic attribute access."""

    def __init__(self, config_path: Optional[str] = None):
        self.base_dir = Path(__file__).parent
        self.path = Path(config_path) if config_path else self.base_dir / "config.yaml"
        with open(self.path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        
        # Initialize mutable flags
        self.pose_save_video = self.cfg.get("models", {}).get("pose", {}).get("save_video", False)
        self.det_save_video = self.cfg.get("models", {}).get("detection", {}).get("save_video", False)

    def resolve(self, p: str | Path | None) -> Optional[Path]:
        """
        Resolve a configured path. If the file is missing locally, attempt to
        download it from a remote source using the environment variables below:

        - `HF_WEIGHTS_REPO`: a Hugging Face Hub repo id (e.g. `username/pose-weights`).
          If set, the code will attempt to download a file with the same basename
          from that repo using `huggingface_hub.hf_hub_download`.
        - `WEIGHTS_BASE_URL`: a generic HTTP(S) base URL where files can be fetched
          by filename (fallback if `HF_WEIGHTS_REPO` is not set).

        Returns the Path to the local file (downloaded or existing) or None.
        """
        if p is None:
            return None
        p = Path(p)

        # If absolute, return as-is (exists check left to caller)
        if p.is_absolute():
            return p if p.exists() else p

        local = (self.base_dir / p).resolve()
        if local.exists():
            return local

        filename = p.name
        hf_repo = os.environ.get("HF_WEIGHTS_REPO")
        if hf_repo:
            try:
                # Lazy import to avoid hard dependency if not used
                from huggingface_hub import hf_hub_download
                target = hf_hub_download(repo_id=hf_repo, filename=filename, cache_dir=str(self.base_dir), force_filename=filename)
                downloaded = Path(target)
                if downloaded.exists():
                    dest = local
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        downloaded.replace(dest)
                    except Exception:
                        import shutil

                        shutil.copy2(downloaded, dest)
                    return dest
            except Exception:
                # proceed to HTTP fallback
                pass

        base_url = os.environ.get("WEIGHTS_BASE_URL")
        if base_url:
            try:
                import requests

                url = base_url.rstrip("/") + "/" + filename
                resp = requests.get(url, stream=True, timeout=30)
                if resp.status_code == 200:
                    local.parent.mkdir(parents=True, exist_ok=True)
                    tmp = local.with_suffix(local.suffix + ".downloading")
                    with open(tmp, "wb") as f:
                        for chunk in resp.iter_content(8192):
                            if chunk:
                                f.write(chunk)
                    tmp.replace(local)
                    return local
            except Exception:
                pass

        # Not found locally and no remote succeeded: return intended local path
        return local

    def detection_weight(self) -> ModelPaths:
        det = self.cfg["models"]["detection"]
        name = det.get("name", "")
        variant = det.get("variant") or "n"
        weights = det.get("weights", {})
        family = "yolo26" if name.startswith("yolo26") else "yolo"

        available = weights.get(family, {})
        w = available.get(variant)
        if w is None and available:
            # fallback to first available weight
            w = next(iter(available.values()))
        if w is None:
            # final fallback: default lightweight model
            w = "yolov8n.pt"

        resolved = self.resolve(w)
        # If resolved is None or doesn't exist, try any existing file from available
        if (resolved is None or not Path(resolved).exists()) and available:
            for candidate in available.values():
                cand_path = self.resolve(candidate)
                if cand_path and cand_path.exists():
                    resolved = cand_path
                    break
        if resolved is None or not Path(resolved).exists():
            raise FileNotFoundError(
                f"Detection weights not found for family='{family}' variant='{variant}'. "
                f"Checked: {resolved}. Available variants: {list(available.keys())}"
            )
        return ModelPaths(weights=resolved)

    def pose_paths(self) -> ModelPaths:
        pose = self.cfg["models"]["pose"]
        name = pose.get("name", "").lower()
        variant = pose.get("variant", "large")
        weights = pose.get("weights", {})
        
        if "vit" in name:
            block = weights.get("vitpose", {}).get(variant, {})
            cfg_path = self.resolve(block.get("config"))
            ckpt_path = self.resolve(block.get("checkpoint"))
            if cfg_path is None or ckpt_path is None or not cfg_path.exists() or not ckpt_path.exists():
                raise FileNotFoundError(
                    f"VitPose weights/config not found for variant='{variant}'. "
                    f"cfg={cfg_path}, ckpt={ckpt_path}"
                )
            return ModelPaths(config=cfg_path, checkpoint=ckpt_path)
        if "rtm" in name:
            block = weights.get("rtm", {}).get(variant, {})
            cfg_path = self.resolve(block.get("config"))
            ckpt_path = self.resolve(block.get("checkpoint"))
            if cfg_path is None or ckpt_path is None or not cfg_path.exists() or not ckpt_path.exists():
                raise FileNotFoundError(
                    f"RTMPose weights/config not found for variant='{variant}'. "
                    f"cfg={cfg_path}, ckpt={ckpt_path}"
                )
            return ModelPaths(config=cfg_path, checkpoint=ckpt_path)
        
        block = weights.get("yolo-pose", {})
        w = block.get(variant) or block.get(name) or name
        resolved = self.resolve(w)
        if (resolved is None or not resolved.exists()) and block:
            # fallback to any existing file in block
            for candidate in block.values():
                cand_path = self.resolve(candidate)
                if cand_path and cand_path.exists():
                    resolved = cand_path
                    break
        if resolved is None or not resolved.exists():
            raise FileNotFoundError(
                f"YOLO-Pose weights not found for variant='{variant}'. Expected at: {resolved}. "
                f"Available variants: {list(block.keys())}"
            )
        return ModelPaths(weights=resolved)

    @property
    def det_cfg(self) -> Dict[str, Any]: return self.cfg["models"]["detection"]
    
    @property
    def pose_cfg(self) -> Dict[str, Any]: return self.cfg["models"]["pose"]

    @property
    def runtime_cfg(self) -> Dict[str, Any]:
        cfg = dict(self.cfg.get("runtime", {}) or {})
        defaults = {
            "resize_width": 640,
            "resize_height": 360,
            "capture_buffer_size": 2,
            "use_fp16": True,
            "min_pose_bbox_area": 2500,
            "pose_every_n_frames": 3,
            "sparta_every_n_frames": 5,
            "max_tracks_per_frame": 8,
            "pose_match_iou": 0.05,
        }
        for key, value in defaults.items():
            if cfg.get(key) is None:
                cfg[key] = value
        return cfg

    def resolved_paths(self) -> Dict[str, Any]:
        raw_paths = self.cfg.get("paths", {})
        raw_input = raw_paths.get("input_video")
        if isinstance(raw_input, str) and raw_input.startswith("__camera__:"):
            input_video = raw_input
        else:
            input_video = self.resolve(raw_input)
        paths = {"input_video": input_video}
        pose_root = self.resolve(raw_paths.get("pose_output_dir") or "../pose_outputs")
        paths["pose_output_dir"] = pose_root
        paths["pose_video_dir"] = pose_root / "pose_vis"
        paths["sparta_output_dir"] = self.resolve(raw_paths.get("sparta_output_dir") or "../evaluation_results_sparta")
        return paths

    @property
    def pose_json_suffix(self) -> str:
        return self.cfg.get("paths", {}).get("pose_json_suffix", ".json")

    @property
    def static_prefix(self) -> str:
        return self.cfg.get("paths", {}).get("static_prefix", "01_")

    def human_centric_filename(self, video_stem: str) -> str:
        """Generate human-centric filename: 01_ + 4 digits (from video stem if possible)."""
        digits = "".join(re.findall(r"\d+", video_stem))
        if digits:
            suffix = digits[-4:].zfill(4)
        else:
            # deterministic fallback so UI can predict the filename
            suffix = "0000"
        return f"{self.static_prefix}{suffix}{self.pose_json_suffix}"


# ---------- Detection ----------

class PersonDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        det_params = cfg.det_cfg
        mp = cfg.detection_weight()
        
        self.model = YOLO(str(mp.weights))
        self.device = det_params.get("device", "cpu")
        self.model.to(self.device)

        self.conf = float(det_params.get("confidence_threshold", 0.3))
        self.iou = float(det_params.get("iou_threshold", 0.45))
        self.classes = det_params.get("classes") 
        
        trk_cfg = det_params.get("tracking", {})
        high_thresh = float(trk_cfg.get("track_high_thresh", self.conf))
        
        self.bt_args = Namespace(
            track_high_thresh=high_thresh,
            track_low_thresh=float(trk_cfg.get("track_low_thresh", 0.1)),
            new_track_thresh=float(trk_cfg.get("new_track_thresh", high_thresh)),
            match_thresh=float(trk_cfg.get("match_thresh", 0.8)),
            track_buffer=int(trk_cfg.get("track_buffer", 30)),
            min_box_area=float(trk_cfg.get("min_box_area", 10)),
            mot20=bool(trk_cfg.get("mot20", False)),
            fuse_score=bool(trk_cfg.get("fuse_score", False)),
            gmc=False,
        )

    def run(self, video_path: Path) -> List[Dict[str, Any]]:
        cap = _open_video_capture(str(video_path))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(self.cfg.runtime_cfg.get("capture_buffer_size", 2)))
        if not cap.isOpened(): raise RuntimeError(f"Cannot open: {video_path}")

        tracker = BYTETracker(self.bt_args, frame_rate=cap.get(cv2.CAP_PROP_FPS))
        records = []
        frame_id = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with tqdm(total=total_frames, desc="Detection & Tracking") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret: break

                frame_infer, sx, sy = _resize_frame_if_needed(frame, self.cfg.runtime_cfg)
                kwargs = {"conf": self.conf, "iou": self.iou, "verbose": False, "device": self.device}
                if self.classes: kwargs["classes"] = self.classes

                with torch.inference_mode():
                    results_list = self.model(frame_infer, **kwargs)
                results = results_list[0]  # Results object

                if results.boxes is not None:
                    boxes = results.boxes
                    # Ensure tracker receives CPU tensors
                    if hasattr(boxes, "cpu"):
                        boxes = boxes.cpu()
                else:
                    boxes = Boxes(torch.zeros((0, 6)), frame.shape[:2])

                # BYTETracker expects a Boxes-like object with .conf/.cls attributes
                tracks = tracker.update(boxes, frame_infer)

                for t in tracks:  # t: [x1, y1, x2, y2, track_id, score, cls, det_idx]
                    if len(t) >= 5:
                        x1, y1, x2, y2 = t[:4]
                        if sx != 1.0 or sy != 1.0:
                            x1, x2 = x1 / sx, x2 / sx
                            y1, y2 = y1 / sy, y2 / sy
                        track_id = t[4]
                        score = t[5] if len(t) > 5 else None
                        cls = t[6] if len(t) > 6 else None
                        records.append({
                            "frame_id": frame_id,
                            "person_id": int(track_id),
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "score": float(score) if score is not None else None,
                            "cls": int(cls) if cls is not None and not np.isnan(cls) else None,
                        })
                
                frame_id += 1
                pbar.update(1)

        cap.release()
        return records


# ---------- Pose Estimator Base ----------

class PoseEstimatorBase:
    """Abstract interface for pose estimators."""
    def process(self, video_path: Path, detections: List[Dict[str, Any]], output_dir: Path, json_suffix: str, output_name: Optional[str] = None) -> Path:
        raise NotImplementedError


# ---------- MMPose Concrete Estimator ----------

class MMPoseTopDownEstimator(PoseEstimatorBase):
    def __init__(self, cfg: Config, model_paths: ModelPaths):
        from mmpose.apis import inference_topdown, init_model
        self.cfg = cfg
        self.model = init_model(str(model_paths.config), str(model_paths.checkpoint), device=cfg.pose_cfg.get("device", "cpu"))
        self.inference_topdown = inference_topdown
        self.conf_threshold = float(cfg.pose_cfg.get("confidence_threshold", 0.25))
        self.expected_kp = None  # determined on first valid frame
        rng = random.Random(os.urandom(16))
        # One dynamic color per run (same across all persons for this video).
        self._vis_color = _random_vibrant_bgr(rng)
        self._vis_line_thickness = int(cfg.pose_cfg.get("vis_line_thickness", 2))
        self._vis_kpt_radius = int(cfg.pose_cfg.get("vis_kpt_radius", 2))

    def _extract_pose_from_mmpose_result(self, sample):
        try:
            inst = sample.pred_instances
            kpts = inst.keypoints
            scores = inst.keypoint_scores
            if torch.is_tensor(kpts):
                kpts = kpts.cpu()
            if torch.is_tensor(scores):
                scores = scores.cpu()
            kpts_np = kpts[0] if len(kpts.shape) == 3 else kpts
            scores_np = scores[0] if len(scores.shape) == 2 else scores
            triplets = [[float(x), float(y), float(s)] for (x, y), s in zip(kpts_np, scores_np)]
            return {"keypoints": triplets, "mean": float(np.mean(scores_np))}
        except Exception:
            return None

    def _estimate(self, frame, bbox):
        with torch.inference_mode():
            results = self.inference_topdown(self.model, frame, [bbox])
        if not results:
            return None
        return self._extract_pose_from_mmpose_result(results[0])

    def _estimate_batch(self, frame, bboxes):
        if not bboxes:
            return []
        with torch.inference_mode():
            results = self.inference_topdown(self.model, frame, bboxes)
        outputs = []
        for sample in results or []:
            outputs.append(self._extract_pose_from_mmpose_result(sample))
        return outputs

    def process(
        self,
        video_path: Path,
        detections: List[Dict[str, Any]],
        output_dir: Path,
        json_suffix: str,
        output_name: Optional[str] = None,
        output_video_path: Optional[Path] = None,
    ) -> Path:
        cap = _open_video_capture(str(video_path))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(self.cfg.runtime_cfg.get("capture_buffer_size", 2)))
        output_dir.mkdir(parents=True, exist_ok=True)
        sparta_json = defaultdict(dict)
        frame_map = defaultdict(list)
        for d in detections: frame_map[d["frame_id"]].append(d)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = None
        if output_video_path and self.cfg.pose_save_video:
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(output_video_path), _video_writer_fourcc(output_video_path, self.cfg.pose_cfg.get("fourcc", "XVID")), fps, (w, h))

        frame_id = 0
        min_bbox_area = float(self.cfg.runtime_cfg.get("min_pose_bbox_area", 0.0))
        with tqdm(total=total, desc="Pose Estimation") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_pose, sx, sy = _resize_frame_if_needed(frame, self.cfg.runtime_cfg)
                persons = frame_map.get(frame_id, [])
                valid_people = []
                batch_bboxes = []
                for p in persons:
                    bbox = p["bbox"]
                    if min_bbox_area > 0 and _bbox_area_xyxy(bbox) < min_bbox_area:
                        continue
                    valid_people.append(p)
                    batch_bboxes.append(_scale_bbox_xyxy(bbox, sx, sy))

                batch_poses = self._estimate_batch(frame_pose, batch_bboxes)
                for p, pose in zip(valid_people, batch_poses):
                    if pose and pose["mean"] >= self.conf_threshold:
                        kp_count = len(pose["keypoints"])
                        if self.expected_kp is None:
                            self.expected_kp = kp_count
                        if kp_count != self.expected_kp:
                            continue

                        if sx != 1.0 or sy != 1.0:
                            for triplet in pose["keypoints"]:
                                triplet[0] /= sx
                                triplet[1] /= sy

                        flat = [c for trip in pose["keypoints"] for c in trip]
                        sparta_json[str(p["person_id"])][str(frame_id)] = {
                            "keypoints": flat,
                            "scores": float(pose.get("mean", 0.0))
                        }

                        if writer is not None:
                            kpts = pose["keypoints"]
                            min_kpt_conf = float(self.cfg.pose_cfg.get("min_keypoint_confidence", 0.3))
                            color = self._vis_color
                            for start_idx, end_idx in SKELETON_CONNECTIONS:
                                if start_idx < kp_count and end_idx < kp_count:
                                    x1, y1, s1 = kpts[start_idx]
                                    x2, y2, s2 = kpts[end_idx]
                                    if s1 >= min_kpt_conf and s2 >= min_kpt_conf:
                                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, self._vis_line_thickness, lineType=cv2.LINE_AA)
                            for (x, y, s) in kpts:
                                if s >= min_kpt_conf:
                                    cv2.circle(frame, (int(x), int(y)), self._vis_kpt_radius, color, -1, lineType=cv2.LINE_AA)

                if writer is not None:
                    writer.write(frame)
                frame_id += 1
                pbar.update(1)

        cap.release()
        if writer is not None: writer.release()
        fname = output_name or f"{video_path.stem}{json_suffix}"
        out = output_dir / fname
        with open(out, "w") as f: json.dump(sparta_json, f, indent=2)
        return out


# ---------- YOLO Pose Estimator ----------

class YoloPoseEstimator(PoseEstimatorBase):
    def __init__(self, cfg: Config, model_paths: ModelPaths):
        self.cfg = cfg
        self.model = YOLO(str(model_paths.weights))
        self.conf = float(cfg.pose_cfg.get("confidence_threshold", 0.25))
        self.conf_threshold = self.conf  # Alias for interface consistency
        self.min_kpt_conf = float(cfg.pose_cfg.get("min_keypoint_confidence", 0.3))
        self.expected_kp = None  # determined on first valid frame
        rng = random.Random(os.urandom(16))
        # One dynamic color per run (same across all persons for this video).
        self._vis_color = _random_vibrant_bgr(rng)
        self._vis_line_thickness = int(cfg.pose_cfg.get("vis_line_thickness", 1))
        self._vis_kpt_radius = int(cfg.pose_cfg.get("vis_kpt_radius", 2))

        self.use_fp16 = str(cfg.pose_cfg.get("device", "cpu")).startswith("cuda") and bool(cfg.runtime_cfg.get("use_fp16", False))

    def _infer_full_frame(self, frame):
        with torch.inference_mode():
            results = self.model(frame, conf=self.conf, verbose=False, half=self.use_fp16)
        if isinstance(results, (list, tuple)):
            return results[0] if results else None
        return results

    def _extract_pose_candidates(self, result, sx: float = 1.0, sy: float = 1.0) -> List[Dict[str, Any]]:
        if result is None:
            return []
        kpt_obj = getattr(result, "keypoints", None)
        boxes = getattr(result, "boxes", None)
        if kpt_obj is None or getattr(kpt_obj, "xy", None) is None or getattr(kpt_obj, "conf", None) is None:
            return []
        if len(kpt_obj.xy) == 0:
            return []

        box_xyxy = None
        if boxes is not None and getattr(boxes, "xyxy", None) is not None and len(boxes.xyxy) > 0:
            box_xyxy = boxes.xyxy.cpu().numpy()

        candidates = []
        for idx in range(len(kpt_obj.xy)):
            kps = kpt_obj.xy[idx].cpu().numpy()
            confs = kpt_obj.conf[idx].cpu().numpy()
            if self.expected_kp is None:
                self.expected_kp = kps.shape
            if kps.shape != self.expected_kp:
                continue

            if sx != 1.0 or sy != 1.0:
                kps[:, 0] /= sx
                kps[:, 1] /= sy

            triplets = [[float(x), float(y), float(c)] for (x, y), c in zip(kps, confs)]
            if box_xyxy is not None and idx < len(box_xyxy):
                bbox = box_xyxy[idx].astype(float).tolist()
                if sx != 1.0 or sy != 1.0:
                    bbox = [bbox[0] / sx, bbox[1] / sy, bbox[2] / sx, bbox[3] / sy]
            else:
                xs = kps[:, 0]
                ys = kps[:, 1]
                bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

            candidates.append({
                "bbox": bbox,
                "keypoints": triplets,
                "mean": float(confs.mean()),
            })
        return candidates

    def _match_candidate_to_bbox(self, candidates: List[Dict[str, Any]], bbox: list[float]) -> Optional[Dict[str, Any]]:
        min_iou = float(self.cfg.runtime_cfg.get("pose_match_iou", 0.05))
        best = None
        best_iou = 0.0
        for cand in candidates:
            iou = _bbox_iou_xyxy(cand["bbox"], bbox)
            if iou > best_iou:
                best_iou = iou
                best = cand
        if best is None or best_iou < min_iou:
            return None
        return best

    def process(
        self,
        video_path: Path,
        detections: List[Dict[str, Any]],
        output_dir: Path,
        json_suffix: str,
        output_name: Optional[str] = None,
        output_video_path: Optional[Path] = None,
    ) -> Path:
        cap = _open_video_capture(str(video_path))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(self.cfg.runtime_cfg.get("capture_buffer_size", 2)))
        output_dir.mkdir(parents=True, exist_ok=True)
        sparta_json = defaultdict(dict)
        frame_map = defaultdict(list)
        for d in detections: frame_map[d["frame_id"]].append(d)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = None
        if output_video_path and self.cfg.pose_save_video:
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(output_video_path), _video_writer_fourcc(output_video_path, self.cfg.pose_cfg.get("fourcc", "XVID")), fps, (w, h))

        frame_id = 0
        with tqdm(total=total, desc="YOLO Pose") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_pose, sx, sy = _resize_frame_if_needed(frame, self.cfg.runtime_cfg)
                persons = frame_map.get(frame_id, [])
                min_bbox_area = float(self.cfg.runtime_cfg.get("min_pose_bbox_area", 0.0))
                result = self._infer_full_frame(frame_pose)
                candidates = self._extract_pose_candidates(result, sx=sx, sy=sy)

                for p in persons:
                    if min_bbox_area > 0 and _bbox_area_xyxy(p["bbox"]) < min_bbox_area:
                        continue
                    pose = self._match_candidate_to_bbox(candidates, p["bbox"])
                    if pose is None or pose.get("mean", 0.0) < self.conf_threshold:
                        continue

                    triplets = pose["keypoints"]
                    flat = [c for trip in triplets for c in trip]
                    sparta_json[str(p["person_id"])][str(frame_id)] = {
                        "keypoints": flat,
                        "scores": float(pose["mean"]),
                    }

                    if writer is not None:
                        kps = pose["keypoints"]
                        color = self._vis_color
                        for start_idx, end_idx in SKELETON_CONNECTIONS:
                            if start_idx < len(kps) and end_idx < len(kps):
                                x1, y1, s1 = kps[start_idx]
                                x2, y2, s2 = kps[end_idx]
                                if s1 >= self.min_kpt_conf and s2 >= self.min_kpt_conf:
                                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, self._vis_line_thickness, lineType=cv2.LINE_AA)
                        for x, y, c in kps:
                            if c >= self.min_kpt_conf:
                                cv2.circle(frame, (int(x), int(y)), self._vis_kpt_radius, color, -1, lineType=cv2.LINE_AA)

                if writer is not None:
                    writer.write(frame)
                frame_id += 1
                pbar.update(1)

        cap.release()
        if writer is not None: writer.release()
        fname = output_name or f"{video_path.stem}{json_suffix}"
        out = output_dir / fname
        with open(out, "w") as f: json.dump(sparta_json, f, indent=2)
        return out


# ---------- Pipeline ----------

# --- Update in pose_estimation.py ---

class PosePipeline:
    def __init__(self, config_path: Optional[str] = None):
        self.config = Config(config_path)
        self.paths = self.config.resolved_paths()
        self.last_anomaly_records: List[Dict[str, Any]] = []
        self.last_pose_json_path: Optional[Path] = None
        self.last_scores_csv_path: Optional[Path] = None

    # Inside PosePipeline class in pose_estimation.py

    def run_live(self):
        """
        Generator that yields processed frames with detection, tracking, pose estimation,
        and real-time SPARTA anomaly detection all happening in parallel.
        Uses rolling keypoint buffers for each person to enable live SPARTA inference.
        """
        p = self.paths
        cap = _open_video_capture(str(p["input_video"]))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {p['input_video']}")
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(self.config.runtime_cfg.get("capture_buffer_size", 2)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(self.config.runtime_cfg.get("resize_width", 1280) or 1280)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(self.config.runtime_cfg.get("resize_height", 720) or 720)
        runtime_cfg = self.config.runtime_cfg
        pose_every_n = max(1, int(runtime_cfg.get("pose_every_n_frames", 1) or 1))
        sparta_every_n = max(1, int(runtime_cfg.get("sparta_every_n_frames", 1) or 1))
        min_bbox_area = float(runtime_cfg.get("min_pose_bbox_area", 0.0) or 0.0)
        
        # Initialize detector, tracker, and pose estimator once
        detector = PersonDetector(self.config)
        tracker = BYTETracker(detector.bt_args, frame_rate=fps)
        
        pose_paths = self.config.pose_paths()
        pose_name = self.config.pose_cfg.get("name", "").lower()
        if "yolo" in pose_name:
            estimator = YoloPoseEstimator(self.config, pose_paths)
        elif "vit" in pose_name or "rtm" in pose_name:
            estimator = MMPoseTopDownEstimator(self.config, pose_paths)
        else:
            estimator = MMPoseTopDownEstimator(self.config, pose_paths)
        
        # --- SPARTA Anomaly Detection Setup ---
        sparta_branch = "SPARTA_C"
        sparta_threshold = 0.5
        try:
            from utils.tokenizer import Tokenizer
            from models import SPARTA_C, SPARTA_F, SPARTA_H
            from utils.train_utils import CostumLoss
            from utils.data_utils import normalize_pose
            
            sparta_cfg = self.config.cfg.get("models", {}).get("sparta", {})
            seg_len = int(sparta_cfg.get("seg_len", 24))
            num_kp = int(sparta_cfg.get("num_kp", 18))
            relative = bool(sparta_cfg.get("relative", True))
            device = self.config.pose_cfg.get("device", "cpu")
            
            sparta_branch = sparta_cfg.get("branch", "SPARTA_C")
            friendly_branch_map = {
                "Reconstruction Model": "SPARTA_C",
                "Future trajectory prediction model": "SPARTA_F",
                "Hybrid": "SPARTA_H",
            }
            sparta_branch = friendly_branch_map.get(sparta_branch, sparta_branch)
            
            ckpt_c = sparta_cfg.get("checkpoints", {}).get("sparta_c") or ""
            ckpt_f = sparta_cfg.get("checkpoints", {}).get("sparta_f") or ""
            ckpt_h_c = sparta_cfg.get("checkpoints", {}).get("sparta_h_c") or ckpt_c
            ckpt_h_f = sparta_cfg.get("checkpoints", {}).get("sparta_h_f") or ckpt_f
            th_c = float(sparta_cfg.get("checkpoints", {}).get("eer_threshold_c", 0.5))
            th_f = float(sparta_cfg.get("checkpoints", {}).get("eer_threshold_f", 0.5))
            th_h = float(sparta_cfg.get("checkpoints", {}).get("eer_threshold_h", max(th_c, th_f)))
            
            # Build SPARTA model. The live path must match the checkpoint d_model exactly.
            expand_ratio = 2 if relative else 1
            configured_d_model = num_kp * 2 * expand_ratio
            num_heads = int(sparta_cfg.get("model_num_heads", 2))
            latent_dim = int(sparta_cfg.get("model_latent_dim", 64))
            dropout = float(sparta_cfg.get("dropout", 0.3))
            sparta_feature_dim = configured_d_model

            def load_checkpoint_state(checkpoint_path: str) -> Dict[str, Any] | None:
                if not checkpoint_path or not Path(checkpoint_path).exists():
                    return None
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                return checkpoint.get("state_dict", checkpoint)

            def infer_d_model(state_dict: Dict[str, Any] | None, fallback: int) -> int:
                if not state_dict:
                    return fallback
                for key, value in state_dict.items():
                    if key.endswith("self_attn.in_proj_weight") and hasattr(value, "shape") and len(value.shape) == 2:
                        return int(value.shape[1])
                return fallback
            
            if sparta_branch == "SPARTA_C":
                state_c = load_checkpoint_state(ckpt_c)
                sparta_feature_dim = infer_d_model(state_c, configured_d_model)
                sparta_model = SPARTA_C(sparta_feature_dim, num_heads, latent_dim, 4, 1000, device=device, dropout=dropout)
                if state_c:
                    sparta_model.load_state_dict(state_c)
                    print(f"[SPARTA] Loaded SPARTA_C from {ckpt_c} | d_model={sparta_feature_dim}")
                sparta_model.to(device)
                sparta_model.eval()
                sparta_threshold = th_c
                sparta_tokenizer = Tokenizer(Namespace(branch="SPARTA_C", device=device, relative=relative, token_config="t", traj=False, num_kp=num_kp))
                loss_func = CostumLoss("MSE", a=1, b=1, c=1, d=1)
            elif sparta_branch == "SPARTA_F":
                state_f = load_checkpoint_state(ckpt_f)
                sparta_feature_dim = infer_d_model(state_f, configured_d_model)
                sparta_model = SPARTA_F(sparta_feature_dim, num_heads, latent_dim, 4, 1000, device=device)
                if state_f:
                    sparta_model.load_state_dict(state_f)
                    print(f"[SPARTA] Loaded SPARTA_F from {ckpt_f} | d_model={sparta_feature_dim}")
                sparta_model.to(device)
                sparta_model.eval()
                sparta_threshold = th_f
                sparta_tokenizer = Tokenizer(Namespace(branch="SPARTA_F", device=device, relative=relative, token_config="t", traj=False, num_kp=num_kp))
                loss_func = CostumLoss("MSE", a=1, b=1, c=1, d=1)
            elif sparta_branch == "SPARTA_H":
                state_c = load_checkpoint_state(ckpt_h_c)
                state_f = load_checkpoint_state(ckpt_h_f)
                sparta_feature_dim = infer_d_model(state_c, infer_d_model(state_f, configured_d_model))
                sparta_model = SPARTA_H(sparta_feature_dim, num_heads, latent_dim, 4, 1000, device=device, dropout=dropout)
                if state_c:
                    sparta_model.CTD.load_state_dict(state_c)
                    print(f"[SPARTA] Loaded SPARTA_H CTD from {ckpt_h_c} | d_model={sparta_feature_dim}")
                if state_f:
                    sparta_model.FTD.load_state_dict(state_f)
                    print(f"[SPARTA] Loaded SPARTA_H FTD from {ckpt_h_f} | d_model={sparta_feature_dim}")
                sparta_model.to(device)
                sparta_model.eval()
                sparta_threshold = th_h
                sparta_tokenizer = Tokenizer(Namespace(branch="SPARTA_H", device=device, relative=relative, token_config="t", traj=False, num_kp=num_kp))
                loss_func = CostumLoss("MSE", a=1, b=1, c=1, d=1)
            else:
                sparta_model = None
                sparta_tokenizer = None
                sparta_threshold = 0.5
                loss_func = None

            if sparta_model is not None:
                feature_multiplier = 4 if relative else 2
                inferred_num_kp = max(1, sparta_feature_dim // feature_multiplier)
                if inferred_num_kp != num_kp:
                    print(f"[SPARTA] Adjusting live keypoints from config num_kp={num_kp} to checkpoint num_kp={inferred_num_kp}")
                    num_kp = inferred_num_kp
            
            sparta_window_len = seg_len * 2 if sparta_branch in {"SPARTA_F", "SPARTA_H"} else seg_len
            sparta_enabled = sparta_model is not None
        except Exception as e:
            print(f"[WARNING] SPARTA loading failed: {e}. Proceeding without anomaly detection.")
            sparta_enabled = False
            sparta_model = None
            sparta_tokenizer = None
            sparta_window_len = int(self.config.cfg.get("models", {}).get("sparta", {}).get("seg_len", 24))
        
        frame_id = 0
        sparta_json = defaultdict(dict)
        self.last_anomaly_records = []
        self.last_pose_json_path = None
        self.last_scores_csv_path = None
        source_text = str(p["input_video"])
        source_stem = "camera0" if source_text.startswith("__camera__:") else Path(source_text).stem

        # Live score CSV: written while frames are processed, not after a second scoring pass.
        p["sparta_output_dir"].mkdir(parents=True, exist_ok=True)
        score_fieldnames = [
            "segment",
            "timeline_index",
            "scene_id",
            "clip_id",
            "person_id",
            "start_frame",
            "score",
            "predicted",
        ]
        branch_slug = str(sparta_branch).lower()
        scores_csv_path = p["sparta_output_dir"] / f"{branch_slug}_scores.csv"
        score_file = scores_csv_path.open("w", newline="", encoding="utf-8")
        score_writer = csv.DictWriter(score_file, fieldnames=score_fieldnames)
        score_writer.writeheader()
        self.last_scores_csv_path = scores_csv_path
        segment_index = 0
        scene_digits = "".join(re.findall(r"\d+", self.config.static_prefix))
        scene_id = int(scene_digits) if scene_digits else 1
        clip_digits = "".join(re.findall(r"\d+", source_stem))
        clip_id = int(clip_digits[-4:]) if clip_digits else 0
        max_tracks_per_frame = int(runtime_cfg.get("max_tracks_per_frame", 8) or 0)
        
        # Rolling buffers: per-person keypoint sequences for SPARTA
        per_person_kpts = defaultdict(list)  # {person_id: [(frame_id, keypoints_array), ...]}
        latest_anomaly_by_track = defaultdict(lambda: 0)  # Persist last SPARTA decision between skipped frames.
        latest_score_by_track = defaultdict(lambda: 0.0)
        cached_pose_by_track: Dict[int, Dict[str, Any]] = {}
        
        def normalize_keypoint_count(kpts_array: np.ndarray) -> np.ndarray:
            if kpts_array.shape[0] == num_kp:
                return kpts_array
            if kpts_array.shape[0] > num_kp:
                return kpts_array[:num_kp]
            pad = np.zeros((num_kp - kpts_array.shape[0], kpts_array.shape[1]), dtype=np.float32)
            return np.concatenate([kpts_array, pad], axis=0)

        def build_sparta_window(recent_frames: list[tuple[int, np.ndarray]]) -> np.ndarray:
            batch_kpts = np.array([kpt_arr for _, kpt_arr in recent_frames], dtype=np.float32)  # (T, V, 3)
            batch_kpts = normalize_pose(
                batch_kpts[None, ...],
                vid_res=[frame_width, frame_height],
                traj=False,
            ).squeeze(0)  # (T, V, 3), same scale as offline dataset
            data = batch_kpts.transpose(2, 0, 1)[:2]  # (C=2, T, V)
            if relative:
                relative_movement = np.zeros_like(data)
                for time_idx in range(1, data.shape[1]):
                    relative_movement[:, time_idx, :] = data[:, time_idx, :] - data[:, 0, :]
                data = np.concatenate((data, relative_movement), axis=2)  # (2, T, 2V)
            return data.transpose(1, 0, 2).reshape(data.shape[1], -1)  # token_config='t': (T, C*V)

        try:
            with tqdm(total=total_frames, desc="Processing") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # --- STEP 1: Detection & Tracking (unified) ---
                    det_kwargs = {"conf": detector.conf, "iou": detector.iou, "verbose": False, "device": detector.device}
                    if detector.classes:
                        det_kwargs["classes"] = detector.classes
                    
                    frame_runtime, sx, sy = _resize_frame_if_needed(frame, runtime_cfg)
                    if str(detector.device).startswith("cuda") and bool(runtime_cfg.get("use_fp16", False)):
                        det_kwargs["half"] = True
                    with torch.inference_mode():
                        results_list = detector.model(frame_runtime, **det_kwargs)
                    if isinstance(results_list, list) and len(results_list) > 0:
                        results = results_list[0]
                    else:
                        results = results_list
                    
                    if hasattr(results, "boxes") and results.boxes is not None:
                        boxes = results.boxes.cpu()
                    else:
                        boxes = Boxes(torch.zeros((0, 6)), frame.shape[:2])
                    
                    tracks = tracker.update(boxes, frame_runtime)
                    track_items = []
                    for t in tracks:
                        if len(t) < 5:
                            continue
                        x1, y1, x2, y2 = t[:4]
                        track_id = int(t[4])
                        if sx != 1.0 or sy != 1.0:
                            x1, x2 = x1 / sx, x2 / sx
                            y1, y2 = y1 / sy, y2 / sy
                        bbox = [float(x1), float(y1), float(x2), float(y2)]
                        area = _bbox_area_xyxy(bbox)
                        if min_bbox_area > 0 and area < min_bbox_area:
                            continue
                        track_items.append((area, track_id, bbox))
                    if max_tracks_per_frame > 0 and len(track_items) > max_tracks_per_frame:
                        track_items = sorted(track_items, key=lambda item: item[0], reverse=True)[:max_tracks_per_frame]
                    
                    # --- STEP 2: Pose Estimation ---
                    # Strict real-time rule: estimate pose only on refresh frames; otherwise reuse cache.
                    refresh_pose = frame_id % pose_every_n == 0
                    yolo_pose_candidates = None
                    refreshed_pose_by_track: Dict[int, Dict[str, Any]] = {}

                    if refresh_pose and isinstance(estimator, YoloPoseEstimator):
                        yolo_result = estimator._infer_full_frame(frame_runtime)
                        yolo_pose_candidates = estimator._extract_pose_candidates(yolo_result, sx=sx, sy=sy)
                    elif refresh_pose and isinstance(estimator, MMPoseTopDownEstimator) and track_items:
                        batch_frame = frame_runtime if (sx != 1.0 or sy != 1.0) else frame
                        batch_bboxes = [
                            _scale_bbox_xyxy(bbox, sx, sy) if (sx != 1.0 or sy != 1.0) else bbox
                            for _, _, bbox in track_items
                        ]
                        batch_poses = estimator._estimate_batch(batch_frame, batch_bboxes)
                        for (_, track_id, _), pose in zip(track_items, batch_poses):
                            if pose and (sx != 1.0 or sy != 1.0):
                                for triplet in pose["keypoints"]:
                                    triplet[0] /= sx
                                    triplet[1] /= sy
                            if pose and pose.get("mean", 0) >= estimator.conf_threshold:
                                refreshed_pose_by_track[track_id] = pose
                                cached_pose_by_track[track_id] = pose

                    draw_items = []
                    pending_sparta_batch = []
                    for _, track_id, bbox in track_items:
                        pose = None
                        if refresh_pose:
                            if isinstance(estimator, YoloPoseEstimator):
                                pose = estimator._match_candidate_to_bbox(yolo_pose_candidates or [], bbox)
                                if pose and pose.get("mean", 0) >= estimator.conf_threshold:
                                    cached_pose_by_track[track_id] = pose
                            elif isinstance(estimator, MMPoseTopDownEstimator):
                                pose = refreshed_pose_by_track.get(track_id)
                            else:
                                pose = cached_pose_by_track.get(track_id)
                        else:
                            pose = cached_pose_by_track.get(track_id)

                        if not pose or pose.get("mean", 0) < estimator.conf_threshold:
                            continue

                        flat = [c for trip in pose["keypoints"] for c in trip]
                        sparta_json[str(track_id)][str(frame_id)] = {
                            "keypoints": flat,
                            "scores": float(pose["mean"]),
                        }
                        draw_items.append((pose, bbox, track_id))

                        if sparta_enabled and len(pose["keypoints"]) > 0:
                            kpts_array = np.array(
                                [[k[0], k[1], k[2] if len(k) > 2 else 1.0] for k in pose["keypoints"]],
                                dtype=np.float32,
                            )
                            kpts_array = normalize_keypoint_count(kpts_array)
                            per_person_kpts[track_id].append((frame_id, kpts_array))
                            if len(per_person_kpts[track_id]) > sparta_window_len + 10:
                                per_person_kpts[track_id] = per_person_kpts[track_id][-(sparta_window_len + 5):]
                            if len(per_person_kpts[track_id]) >= sparta_window_len and frame_id % sparta_every_n == 0:
                                recent_frames = per_person_kpts[track_id][-sparta_window_len:]
                                pending_sparta_batch.append(
                                    {
                                        "track_id": track_id,
                                        "start_frame": int(recent_frames[0][0]),
                                        "model_input": build_sparta_window(recent_frames),
                                    }
                                )

                    # --- STEP 3: Batched SPARTA Anomaly Detection for all eligible people in this frame ---
                    if pending_sparta_batch and sparta_enabled:
                        try:
                            batch_np = np.stack([item["model_input"] for item in pending_sparta_batch], axis=0)
                            kpts_tensor = torch.from_numpy(batch_np).to(device, dtype=torch.float32)
                            with torch.inference_mode():
                                if sparta_branch == "SPARTA_C":
                                    recon = sparta_model.forward(kpts_tensor, kpts_tensor)
                                    loss_values = torch.mean((kpts_tensor - recon) ** 2, dim=tuple(range(1, kpts_tensor.ndim)))
                                elif sparta_branch == "SPARTA_F":
                                    input_tensor = kpts_tensor[:, :seg_len, :]
                                    target_tensor = kpts_tensor[:, seg_len:seg_len * 2, :]
                                    pred_tensor = sparta_model.forward(input_tensor, target_tensor)
                                    loss_values = torch.mean((target_tensor - pred_tensor) ** 2, dim=tuple(range(1, target_tensor.ndim)))
                                else:  # SPARTA_H
                                    input_tensor = kpts_tensor[:, :seg_len, :]
                                    target_tensor = kpts_tensor[:, seg_len:seg_len * 2, :]
                                    ctd_out, ftd_out = sparta_model.forward(input_tensor, target_tensor)
                                    loss_c = torch.mean((input_tensor - ctd_out) ** 2, dim=tuple(range(1, input_tensor.ndim)))
                                    loss_f = torch.mean((target_tensor - ftd_out) ** 2, dim=tuple(range(1, target_tensor.ndim)))
                                    loss_values = (loss_c + loss_f) * 0.5
                            for item, score_value in zip(pending_sparta_batch, loss_values.detach().cpu().numpy().reshape(-1)):
                                score = float(score_value)
                                pred = int(score > sparta_threshold)
                                track_id = int(item["track_id"])
                                latest_score_by_track[track_id] = score
                                latest_anomaly_by_track[track_id] = pred
                                score_writer.writerow(
                                    {
                                        "segment": segment_index,
                                        "timeline_index": segment_index,
                                        "scene_id": scene_id,
                                        "clip_id": clip_id,
                                        "person_id": track_id,
                                        "start_frame": int(item["start_frame"]),
                                        "score": score,
                                        "predicted": pred,
                                    }
                                )
                                segment_index += 1
                            score_file.flush()
                        except Exception as e:
                            print(f"[SPARTA] Batched inference error at frame {frame_id}: {e}")

                    # --- STEP 4: Lightweight visualization with latest live score state ---
                    for pose, bbox, track_id in draw_items:
                        current_anomaly_pred = int(latest_anomaly_by_track.get(track_id, 0))
                        current_score = float(latest_score_by_track.get(track_id, 0.0))
                        self.last_anomaly_records.append(
                            {
                                "frame_id": frame_id,
                                "person_id": int(track_id),
                                "anomaly_score": current_score,
                                "anomaly_pred": current_anomaly_pred,
                            }
                        )
                        frame = self._draw_pose_on_frame(
                            frame,
                            pose,
                            estimator,
                            bbox=bbox,
                            track_id=track_id,
                            is_anomaly=(current_anomaly_pred == 1),
                        )
                    
                    yield frame, frame_id, total_frames
                    frame_id += 1
                    pbar.update(1)
        finally:
            score_file.close()
            cap.release()
        
        # Save the final JSON after all frames processed
        p["pose_output_dir"].mkdir(parents=True, exist_ok=True)
        out_name = self.config.human_centric_filename(source_stem)
        final_path = p["pose_output_dir"] / out_name
        with open(final_path, "w") as f:
            json.dump(sparta_json, f, indent=2)
        
        self.last_pose_json_path = final_path
        return final_path

    def _estimate_yolo_pose(self, estimator, frame, bbox):
        """Extract pose from YOLO model within a bbox."""
        x1, y1, x2, y2 = map(int, bbox)
        frame_h, frame_w = frame.shape[:2]
        
        # Clamp bbox to frame
        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w, x2))
        y2 = max(0, min(frame_h, y2))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # Run YOLO pose on ROI
        results = estimator.model(roi, conf=estimator.conf, verbose=False)
        if isinstance(results, list):
            if not results:
                return None
            result = results[0]
        else:
            result = results
        
        kpt_obj = getattr(result, "keypoints", None)
        if kpt_obj is None or not hasattr(kpt_obj, 'xy') or kpt_obj.xy is None or len(kpt_obj.xy) == 0:
            return None
        
        # Get best pose (most confident)
        best_i = 0
        boxes = getattr(result, "boxes", None)
        if boxes is not None and hasattr(boxes, "conf") and len(boxes.conf) > 0:
            try:
                best_i = int(boxes.conf.argmax().item())
            except Exception:
                best_i = 0
        
        kps = kpt_obj.xy[best_i].cpu().numpy()
        confs = kpt_obj.conf[best_i].cpu().numpy()
        
        # Offset keypoints back to original frame coordinates
        kps[:, 0] += x1
        kps[:, 1] += y1
        
        triplets = [[float(x), float(y), float(c)] for (x, y), c in zip(kps, confs)]
        return {"keypoints": triplets, "mean": float(confs.mean())}

    def _draw_pose_on_frame(self, frame, pose, estimator, bbox: Optional[list[float]] = None, track_id: Optional[int] = None, is_anomaly: bool = False):
        """Draw lightweight real-time skeleton overlay. Red means anomaly, green means normal."""
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)  # BGR: red anomaly, green normal
        
        kpts = pose["keypoints"]
        min_kpt_conf = estimator.min_kpt_conf if hasattr(estimator, "min_kpt_conf") else 0.3
        line_thickness = max(1, min(2, estimator._vis_line_thickness if hasattr(estimator, "_vis_line_thickness") else 1))
        kpt_radius = max(1, min(3, estimator._vis_kpt_radius if hasattr(estimator, "_vis_kpt_radius") else 2))
        runtime_cfg = getattr(getattr(estimator, "cfg", None), "runtime_cfg", {}) or {}
        draw_bbox = bool(runtime_cfg.get("draw_bbox", False))
        draw_labels = bool(runtime_cfg.get("draw_labels", False))
        
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if start_idx < len(kpts) and end_idx < len(kpts):
                x1, y1, s1 = kpts[start_idx]
                x2, y2, s2 = kpts[end_idx]
                if s1 >= min_kpt_conf and s2 >= min_kpt_conf:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness, lineType=cv2.LINE_AA)
        
        for (x, y, s) in kpts:
            if s >= min_kpt_conf:
                cv2.circle(frame, (int(x), int(y)), kpt_radius, color, -1, lineType=cv2.LINE_AA)

        if bbox is not None and (draw_bbox or draw_labels):
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            if draw_bbox:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness, lineType=cv2.LINE_AA)
            if draw_labels and track_id is not None:
                label = f"ID:{track_id}"
                text_pos = (x1, max(y1 - 6, 12))
                cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, lineType=cv2.LINE_AA)
                cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, lineType=cv2.LINE_AA)
        
        return frame

# ---------- Main ----------

if __name__ == "__main__":
    config = Config("config.yaml")
    p = config.resolved_paths()

    print("\n[STEP 1] Running person detection...")
    detector = PersonDetector(config)
    dets = detector.run(p["input_video"])

    if dets:
        print("\n[STEP 2] Running pose estimation...")
        paths_pose = config.pose_paths()
        estimator = MMPoseTopDownEstimator(config, paths_pose)
        
        # Disable video saving manually if needed
        config.pose_save_video = False 

        hc_name = config.human_centric_filename(Path(p["input_video"]).stem)
        final_json = estimator.process(p["input_video"], dets, p["pose_output_dir"], config.pose_json_suffix, output_name=hc_name)
        print(f"\n[SUCCESS] Saved to: {final_json}")
    else:
        print("\n[SKIP] No persons found in detection phase.")
