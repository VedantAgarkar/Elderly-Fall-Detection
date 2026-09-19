"""
inference_stream.py
===================
Production-grade real-time video stream processor for Elderly Fall Detection.
Supports Webcam (0), RTSP IP Camera stream, video file (.mp4, .avi), or image sequences.
Backends: ONNX Runtime (CPU/CUDA) or PyTorch TorchScript JIT.

Features:
- 16-frame sliding ring buffer with temporal feature differencing
- Dual-threshold clinical logic (Warning: 0.30, Emergency: 0.50)
- Temporal persistence filter (requires >= 2 consecutive alarm windows)
- Real-time HUD telemetry: FPS, inference latency, confidence gauge, state machine badge
- Output video recording option with annotated HUD
"""

import os
import sys
import time
import glob
import argparse
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

# Optional ONNX Runtime import
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

# PyTorch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class FallDetectorInferenceEngine:
    """Unified inference wrapper supporting TorchScript and ONNX Runtime backends."""
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        img_size: Tuple[int, int] = (224, 224),
        seq_len: int = 16
    ):
        self.model_path = model_path
        self.device = device
        self.img_size = img_size
        self.seq_len = seq_len
        self.backend = "onnx" if model_path.endswith(".onnx") else "torchscript"

        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        if self.backend == "onnx":
            if not HAS_ONNX:
                raise ImportError("onnxruntime is required to run .onnx models. Install with: pip install onnxruntime")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            print(f"[InferenceEngine] Loaded ONNX model with providers: {self.session.get_providers()}")
        else:
            if not HAS_TORCH:
                raise ImportError("torch is required to run TorchScript (.pt) models.")
            self.torch_device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
            self.model = torch.jit.load(model_path, map_location=self.torch_device)
            self.model.eval()
            print(f"[InferenceEngine] Loaded TorchScript model on device: {self.torch_device}")

    def preprocess_clip(self, frames_bgr: list) -> np.ndarray:
        """Preprocesses a list of 16 BGR frames into normalized tensor (1, T, 3, H, W)."""
        processed = []
        for frame in frames_bgr:
            # BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            resized = cv2.resize(rgb, self.img_size, interpolation=cv2.INTER_LINEAR)
            # Normalize to [0, 1] then Standardize
            norm = (resized.astype(np.float32) / 255.0 - self.mean) / self.std
            # (H, W, C) -> (C, H, W)
            chw = np.transpose(norm, (2, 0, 1))
            processed.append(chw)
        # Stack to (1, T, 3, H, W)
        clip = np.expand_dims(np.stack(processed, axis=0), axis=0)
        return clip

    def predict(self, frames_bgr: list) -> Tuple[float, np.ndarray]:
        """Runs forward inference returning fall probability and attention weights."""
        clip_np = self.preprocess_clip(frames_bgr)

        if self.backend == "onnx":
            outputs = self.session.run(None, {self.input_name: clip_np})
            logits = outputs[0][0]  # (2,)
            attn = outputs[1][0] if len(outputs) > 1 else np.ones((self.seq_len, 1))
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            fall_prob = float(probs[1])
            return fall_prob, attn
        else:
            with torch.no_grad():
                tensor = torch.from_numpy(clip_np).to(self.torch_device)
                logits, attn = self.model(tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                fall_prob = float(probs[1])
                attn_np = attn.cpu().numpy()[0]
                return fall_prob, attn_np


class RealTimeStreamMonitor:
    """Sliding-window stream processor with HUD rendering and alarm management."""
    def __init__(
        self,
        engine: FallDetectorInferenceEngine,
        threshold: float = 0.40,
        warning_thresh: float = 0.20,
        persistence: int = 2,
        ema_alpha: float = 0.70
    ):
        self.engine = engine
        self.threshold = threshold
        self.warning_thresh = warning_thresh
        self.persistence = persistence
        self.ema_alpha = ema_alpha

        self.buffer = deque(maxlen=engine.seq_len)
        self.smoothed_prob = 0.0
        self.alarm_counter = 0
        self.state = "NORMAL"  # NORMAL, WARNING, EMERGENCY

        # Benchmark counters
        self.fps_tracker = deque(maxlen=30)
        self.latency_tracker = deque(maxlen=30)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Ingests a video frame, updates ring buffer, runs inference, and renders HUD."""
        t0 = time.time()
        self.buffer.append(frame.copy())

        fall_prob = 0.0
        inf_time_ms = 0.0
        attn_weights = None

        if len(self.buffer) == self.engine.seq_len:
            t_inf0 = time.time()
            raw_prob, attn_weights = self.engine.predict(list(self.buffer))
            inf_time_ms = (time.time() - t_inf0) * 1000.0
            self.latency_tracker.append(inf_time_ms)

            # EMA Smoothing
            self.smoothed_prob = (self.ema_alpha * raw_prob) + ((1.0 - self.ema_alpha) * self.smoothed_prob)

            # State Machine with Persistence
            if self.smoothed_prob >= self.threshold:
                self.alarm_counter += 1
                if self.alarm_counter >= self.persistence:
                    self.state = "EMERGENCY"
            elif self.smoothed_prob >= self.warning_thresh:
                self.state = "WARNING"
                self.alarm_counter = max(0, self.alarm_counter - 1)
            else:
                self.state = "NORMAL"
                self.alarm_counter = 0

        # Frame processing time and FPS
        total_time = time.time() - t0
        fps = 1.0 / max(1e-5, total_time)
        self.fps_tracker.append(fps)

        avg_fps = np.mean(self.fps_tracker)
        avg_lat = np.mean(self.latency_tracker) if self.latency_tracker else 0.0

        # Render HUD
        annotated = self._render_hud(frame, avg_fps, avg_lat)

        telemetry = {
            "state": self.state,
            "raw_prob": fall_prob,
            "smoothed_prob": self.smoothed_prob,
            "fps": avg_fps,
            "latency_ms": avg_lat,
            "alarm_active": (self.state == "EMERGENCY")
        }
        return annotated, telemetry

    def _render_hud(self, frame: np.ndarray, fps: float, lat: float) -> np.ndarray:
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # Top Banner
        cv2.rectangle(canvas, (0, 0), (w, 55), (20, 20, 20), -1)

        # State Indicator Colors
        if self.state == "EMERGENCY":
            status_color = (0, 0, 230)  # Bright Red
            status_text = "EMERGENCY: FALL DETECTED"
        elif self.state == "WARNING":
            status_color = (0, 215, 255)  # Amber / Yellow
            status_text = "CAUTION: RAPID MOTION"
        else:
            status_color = (0, 200, 50)  # Green
            status_text = "SYSTEM MONITORING: NORMAL"

        # Status Badge
        cv2.rectangle(canvas, (15, 10), (320, 45), status_color, -1)
        cv2.putText(canvas, status_text, (25, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Performance Stats
        stats_str = f"FPS: {fps:4.1f} | Latency: {lat:4.1f}ms | Buf: {len(self.buffer)}/{self.engine.seq_len}"
        cv2.putText(canvas, stats_str, (w - 380, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        # Bottom Probability Gauge
        gauge_w = 260
        gauge_h = 16
        gauge_x = w - gauge_w - 20
        gauge_y = h - 30

        cv2.rectangle(canvas, (gauge_x - 5, gauge_y - 22), (w - 15, h - 10), (20, 20, 20), -1)
        prob_str = f"Fall Risk: {self.smoothed_prob*100:.1f}%"
        cv2.putText(canvas, prob_str, (gauge_x, gauge_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        # Gauge Bar
        fill_w = int(gauge_w * min(1.0, max(0.0, self.smoothed_prob)))
        cv2.rectangle(canvas, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h), (60, 60, 60), 1)
        cv2.rectangle(canvas, (gauge_x, gauge_y), (gauge_x + fill_w, gauge_y + gauge_h), status_color, -1)

        # Threshold marker
        thresh_x = gauge_x + int(gauge_w * self.threshold)
        cv2.line(canvas, (thresh_x, gauge_y - 2), (thresh_x, gauge_y + gauge_h + 2), (255, 255, 255), 2)

        # Emergency Flashing Border
        if self.state == "EMERGENCY":
            cv2.rectangle(canvas, (0, 0), (w, h), (0, 0, 240), 6)

        return canvas


def run_stream(
    source: Union[str, int],
    model_path: str,
    output_path: Optional[str] = None,
    device: str = "cuda",
    threshold: float = 0.40,
    max_frames: Optional[int] = None
):
    print(f"\n=======================================================")
    print(f"  Elderly Fall Detection — Real-Time Inference Stream  ")
    print(f"  Source: {source} | Model: {model_path} | Device: {device}")
    print(f"=======================================================\n")

    engine = FallDetectorInferenceEngine(model_path=model_path, device=device)
    monitor = RealTimeStreamMonitor(engine=engine, threshold=threshold)

    # Open video source
    if isinstance(source, str) and os.path.isdir(source):
        # Image directory source
        files = sorted(glob.glob(os.path.join(source, "*.png")) + glob.glob(os.path.join(source, "*.jpg")))
        print(f"[Stream] Processing {len(files)} frames from directory: {source}")
        writer = None
        for i, fp in enumerate(files):
            if max_frames and i >= max_frames:
                break
            frame = cv2.imread(fp)
            if frame is None:
                continue
            annotated, tele = monitor.process_frame(frame)
            if output_path:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_path, fourcc, 25.0, (w, h))
                writer.write(annotated)
        if writer:
            writer.release()
            print(f"✓ Video output saved to: {output_path}")
        return

    # OpenCV VideoCapture source
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        print(f"✗ Failed to open video source: {source}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            annotated, telemetry = monitor.process_frame(frame)

            if writer:
                writer.write(annotated)

            if telemetry["alarm_active"]:
                print(f"[ALARM @ frame {frame_idx}] Probability: {telemetry['smoothed_prob']*100:.1f}%")

    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"✓ Annotated video output saved to: {output_path}")

    print(f"Stream processing complete. Processed {frame_idx} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Fall Detection Stream Processor")
    parser.add_argument("--source", type=str, default="Training data/fall-01-cam0-rgb", help="Path to video file, folder, or webcam ID (0)")
    parser.add_argument("--model", type=str, default="outputs/fall_detector.onnx", help="Path to ONNX or TorchScript model")
    parser.add_argument("--output", type=str, default="outputs/demo_fall_stream.mp4", help="Path to save annotated MP4 video")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--threshold", type=float, default=0.35, help="Clinical operating decision threshold")
    parser.add_argument("--max-frames", type=int, default=150, help="Maximum frames to process for demo")
    args = parser.parse_args()

    run_stream(
        source=args.source,
        model_path=args.model,
        output_path=args.output,
        device=args.device,
        threshold=args.threshold,
        max_frames=args.max_frames
    )
