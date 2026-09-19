"""
run_training.py
===============
Standalone GPU training pipeline for the Elderly Fall Detection project.
Extracts all logic from fall_detection_training.ipynb Cells 3-20 into a
single, runnable script. Tuned for GTX 1650 (4 GB VRAM):
  - batch_size = 4
  - AMP enabled
  - num_workers = 2

Outputs:
  ./checkpoints/best_fall_detector.pth  — best checkpoint (Val F1-based)
  ./checkpoints/history.json            — full loss/metric history per epoch
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import os
import glob
import json
import math
import random
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms


# ─── Configuration ──────────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    # Paths
    data_dir: str = "./Training data"
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./outputs"
    best_model_path: str = "./checkpoints/best_fall_detector.pth"

    # Dataset & Sequence
    sequence_length: int = 16
    frame_stride: int = 2
    image_size: Tuple[int, int] = (224, 224)
    num_classes: int = 2
    val_split_ratio: float = 0.20
    train_window_step: int = 4
    val_window_step: int = 6
    train_strides: Tuple[int, ...] = (1, 2)

    # Model Architecture
    backbone_pretrained: bool = True
    freeze_backbone_epochs: int = 2
    hidden_dim: int = 256
    num_lstm_layers: int = 2
    bidirectional: bool = True
    dropout_rate: float = 0.40
    use_motion_diff: bool = True

    # Training Hyperparameters — tuned for 4 GB VRAM
    batch_size: int = 4
    num_epochs: int = 20
    learning_rate: float = 1e-4
    backbone_lr: float = 2e-5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    use_amp: bool = True

    # Hardware
    num_workers: int = 2
    pin_memory: bool = True
    seed: int = 42
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

# Alias for backwards compatibility with checkpoints serialized with __main__.TrainingConfig
import __main__
__main__.TrainingConfig = TrainingConfig


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Dataset Utilities ──────────────────────────────────────────────────────
def scan_dataset_sequences(data_dir: str) -> pd.DataFrame:
    sequence_dirs = sorted([
        d for d in glob.glob(os.path.join(data_dir, "*"))
        if os.path.isdir(d)
    ])
    records = []
    for seq_dir in sequence_dirs:
        seq_name = os.path.basename(seq_dir)
        frame_files = sorted(
            glob.glob(os.path.join(seq_dir, "*.png")) +
            glob.glob(os.path.join(seq_dir, "*.jpg"))
        )
        if not frame_files:
            continue
        records.append({
            "sequence_name": seq_name,
            "sequence_path": seq_dir,
            "frame_count": len(frame_files),
        })
    return pd.DataFrame(records)


def load_urfd_ground_truth(meta_dir: str = "metadata") -> Dict[Tuple[str, int], int]:
    falls_csv = os.path.join(meta_dir, "urfall-cam0-falls.csv")
    adls_csv = os.path.join(meta_dir, "urfall-cam0-adls.csv")
    gt_map: Dict[Tuple[str, int], int] = {}
    if os.path.exists(falls_csv):
        df_falls = pd.read_csv(falls_csv, header=None)
        for _, r in df_falls.iterrows():
            gt_map[(str(r[0]), int(r[1]))] = int(r[2])
    if os.path.exists(adls_csv):
        df_adls = pd.read_csv(adls_csv, header=None)
        for _, r in df_adls.iterrows():
            gt_map[(str(r[0]), int(r[1]))] = int(r[2])
    return gt_map


def parse_frame_num(filename: str) -> int:
    stem = Path(filename).stem
    parts = stem.split('-')
    try:
        return int(parts[-1])
    except ValueError:
        return 0


def get_sequence_base(dir_name: str) -> str:
    # e.g. fall-01-cam0-rgb -> fall-01, adl-04-cam0-rgb -> adl-04
    parts = dir_name.split('-')
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return dir_name


def build_clip_annotations(
    df_seqs: pd.DataFrame,
    seq_len: int = 16,
    stride: int = 2,
    window_step: int = 8,
    strides: Optional[List[int]] = None
) -> pd.DataFrame:
    if strides is None:
        strides = [stride]

    gt_map = load_urfd_ground_truth()
    clip_records = []
    for _, row in df_seqs.iterrows():
        seq_name = row["sequence_name"]
        seq_path = row["sequence_path"]
        frame_files = sorted(
            glob.glob(os.path.join(seq_path, "*.png")) +
            glob.glob(os.path.join(seq_path, "*.jpg"))
        )
        total_frames = len(frame_files)
        if total_frames < seq_len:
            continue

        base_name = get_sequence_base(seq_name)
        is_adl_seq = base_name.startswith("adl")

        # Map each frame file to its ground-truth posture label (-1: ADL, 0: falling, 1: lying)
        if gt_map:
            frame_labels = [gt_map.get((base_name, parse_frame_num(f)), -1) for f in frame_files]
        else:
            fall_start = int(total_frames * 0.30)
            fall_end = int(total_frames * 0.75)
            frame_labels = [1 if fall_start <= i <= fall_end else -1 for i in range(total_frames)]

        for s in strides:
            span = (seq_len - 1) * s + 1
            if total_frames < span:
                continue

            for start_idx in range(0, total_frames - span + 1, window_step):
                clip_indices = [start_idx + i * s for i in range(seq_len)]
                clip_frames = [frame_files[i] for i in clip_indices]
                clip_lbls = [frame_labels[i] for i in clip_indices]

                if is_adl_seq:
                    # In ADL sequences, every activity is Non-Fall (Class 0)
                    is_fall = 0
                else:
                    # In Fall sequences, label as Fall (1) if it contains falling transition or fallen posture
                    has_transition = any(l == 0 for l in clip_lbls)
                    fallen_count = sum(1 for l in clip_lbls if l == 1)
                    is_fall = 1 if (has_transition or fallen_count >= 4) else 0

                clip_records.append({
                    "sequence_name": seq_name,
                    "sequence_path": seq_path,
                    "start_idx": start_idx,
                    "stride": s,
                    "frame_paths": clip_frames,
                    "label": is_fall,
                })

    return pd.DataFrame(clip_records)


# ─── Augmentation & Dataset ──────────────────────────────────────────────────
class VideoSequenceAugmentor:
    def __init__(self, is_train: bool = True, image_size: Tuple[int, int] = (224, 224)):
        self.is_train = is_train
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]

        if self.is_train:
            # Synchronized spatial transforms across all frames in the clip
            do_hflip   = random.random() > 0.5
            brightness = random.uniform(0.8, 1.2)
            contrast   = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            rot_angle  = random.uniform(-8.0, 8.0) if random.random() > 0.5 else 0.0

            # Random crop parameters
            crop_scale = random.uniform(0.88, 1.0)
            target_h = int(self.image_size[0] * crop_scale)
            target_w = int(self.image_size[1] * crop_scale)
            i, j, h, w = transforms.RandomCrop.get_params(
                pil_frames[0].resize(self.image_size, Image.BILINEAR),
                output_size=(target_h, target_w)
            )

            # Random cutout parameters
            use_cutout = random.random() < 0.35
            cut_h = int(self.image_size[0] * random.uniform(0.12, 0.20))
            cut_w = int(self.image_size[1] * random.uniform(0.12, 0.20))
            cut_top  = random.randint(0, self.image_size[0] - cut_h)
            cut_left = random.randint(0, self.image_size[1] - cut_w)

            processed = []
            for img in pil_frames:
                img = img.resize(self.image_size, Image.BILINEAR)
                if do_hflip:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if rot_angle != 0.0:
                    img = img.rotate(rot_angle, resample=Image.BILINEAR)
                img = transforms.functional.adjust_brightness(img, brightness)
                img = transforms.functional.adjust_contrast(img, contrast)
                img = transforms.functional.adjust_saturation(img, saturation)
                img = transforms.functional.crop(img, i, j, h, w)
                img = img.resize(self.image_size, Image.BILINEAR)

                t = self.to_tensor(img)
                if use_cutout:
                    t[:, cut_top:cut_top+cut_h, cut_left:cut_left+cut_w] = 0.0
                t = self.normalize(t)
                processed.append(t)
        else:
            processed = []
            for img in pil_frames:
                img = img.resize(self.image_size, Image.BILINEAR)
                t = self.to_tensor(img)
                t = self.normalize(t)
                processed.append(t)

        return torch.stack(processed, dim=0)  # (T, C, H, W)


class FallSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, is_train: bool, image_size: Tuple[int, int]):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.augmentor = VideoSequenceAugmentor(is_train=is_train, image_size=image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        label = int(row["label"])
        frame_paths = row["frame_paths"]

        # Temporal reversal augmentation for ADL clips (doubles normal motion diversity)
        if self.is_train and label == 0 and random.random() > 0.5:
            frame_paths = list(reversed(frame_paths))

        frames = []
        for p in frame_paths:
            img = cv2.imread(p)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            frames.append(img)
        tensor_clip = self.augmentor(frames)
        return tensor_clip, torch.tensor(label, dtype=torch.long)


def create_data_loaders(
    data_input: pd.DataFrame,
    config: TrainingConfig
) -> Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
    # Check if data_input represents raw sequences or already-annotated clips
    if "frame_count" in data_input.columns:
        df_seqs = data_input.copy()
        df_seqs["is_adl"] = df_seqs["sequence_name"].apply(lambda x: x.startswith("adl"))
        if df_seqs["is_adl"].nunique() > 1:
            train_seqs, val_seqs = train_test_split(
                df_seqs, test_size=config.val_split_ratio, random_state=config.seed, stratify=df_seqs["is_adl"]
            )
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=config.val_split_ratio, random_state=config.seed)
            train_idx, val_idx = next(gss.split(df_seqs, groups=df_seqs["sequence_name"]))
            train_seqs = df_seqs.iloc[train_idx]
            val_seqs   = df_seqs.iloc[val_idx]

        train_df = build_clip_annotations(
            train_seqs,
            seq_len=config.sequence_length,
            window_step=config.train_window_step,
            strides=list(config.train_strides)
        )
        val_df = build_clip_annotations(
            val_seqs,
            seq_len=config.sequence_length,
            window_step=config.val_window_step,
            strides=[config.frame_stride]
        )
    else:
        # Standard clip-level DataFrame split by sequence
        gss = GroupShuffleSplit(n_splits=1, test_size=config.val_split_ratio, random_state=config.seed)
        train_idx, val_idx = next(gss.split(data_input, groups=data_input["sequence_name"]))
        train_df = data_input.iloc[train_idx].copy()
        val_df   = data_input.iloc[val_idx].copy()

    print(f"  Training  : {len(train_df)} clips across {train_df['sequence_name'].nunique()} sequences")
    print(f"  Validation: {len(val_df)} clips across {val_df['sequence_name'].nunique()} sequences")
    print(f"  Class dist (train): {train_df['label'].value_counts().to_dict()}")
    print(f"  Class dist (val)  : {val_df['label'].value_counts().to_dict()}")

    train_loader = DataLoader(
        FallSequenceDataset(train_df, is_train=True,  image_size=config.image_size),
        batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=True
    )
    val_loader = DataLoader(
        FallSequenceDataset(val_df,   is_train=False, image_size=config.image_size),
        batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=False
    )
    return train_loader, val_loader, train_df, val_df


# ─── Model ──────────────────────────────────────────────────────────────────
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention_weights(lstm_outputs)
        weights = F.softmax(scores, dim=1)
        context_vector = torch.sum(weights * lstm_outputs, dim=1)
        return context_vector, weights


class FallDetectorResNetLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout_rate: float = 0.4,
        pretrained_backbone: bool = True,
        use_motion_diff: bool = True,
    ):
        super().__init__()
        self.use_motion_diff = use_motion_diff
        weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
        base_resnet = models.resnet18(weights=weights)
        self.feature_dim = base_resnet.fc.in_features  # 512
        self.backbone = nn.Sequential(*list(base_resnet.children())[:-1])

        lstm_in_dim = self.feature_dim * 2 if use_motion_diff else self.feature_dim
        self.lstm = nn.LSTM(
            input_size=lstm_in_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_lstm_layers > 1 else 0.0
        )
        effective_hidden = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = TemporalAttention(effective_hidden)
        self.classifier = nn.Sequential(
            nn.LayerNorm(effective_hidden),
            nn.Dropout(dropout_rate),
            nn.Linear(effective_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, num_classes)
        )

    def set_backbone_trainable(self, trainable: bool):
        for param in self.backbone.parameters():
            param.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C, H, W = x.shape
        frames_flat = x.view(B * T, C, H, W)
        spatial_features = self.backbone(frames_flat).view(B, T, self.feature_dim)

        if self.use_motion_diff:
            velocity = torch.zeros_like(spatial_features)
            velocity[:, 1:] = spatial_features[:, 1:] - spatial_features[:, :-1]
            temporal_features = torch.cat([spatial_features, velocity], dim=-1)
        else:
            temporal_features = spatial_features

        lstm_out, _ = self.lstm(temporal_features)
        context_vector, attn_weights = self.attention(lstm_out)
        logits = self.classifier(context_vector)
        return logits, attn_weights


# ─── Loss ────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs, targets, reduction="none", weight=self.alpha,
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


# ─── Metrics ─────────────────────────────────────────────────────────────────
class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_sum = 0.0
        self.count = 0
        self.all_targets: List[int] = []
        self.all_preds:   List[int] = []
        self.all_probs:   List[float] = []

    def update(self, loss: float, targets: torch.Tensor, logits: torch.Tensor):
        bs = targets.size(0)
        self.loss_sum += loss * bs
        self.count    += bs
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        self.all_targets.extend(targets.detach().cpu().numpy())
        self.all_preds.extend(preds)
        self.all_probs.extend(probs)

    def compute(self) -> Dict[str, float]:
        tgts  = np.array(self.all_targets)
        preds = np.array(self.all_preds)
        probs = np.array(self.all_probs)
        avg_loss = self.loss_sum / max(1, self.count)
        acc  = accuracy_score(tgts, preds)  if len(tgts) > 0 else 0.0
        prec = precision_score(tgts, preds, zero_division=0) if len(tgts) > 0 else 0.0
        rec  = recall_score(tgts, preds,    zero_division=0) if len(tgts) > 0 else 0.0
        f1   = f1_score(tgts, preds,        zero_division=0) if len(tgts) > 0 else 0.0
        try:
            auc = roc_auc_score(tgts, probs) if len(np.unique(tgts)) > 1 else 0.5
        except ValueError:
            auc = 0.5

        if len(tgts) > 0:
            tn = np.sum((tgts == 0) & (preds == 0))
            fp = np.sum((tgts == 0) & (preds == 1))
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        else:
            spec = 0.0

        return {"loss": avg_loss, "accuracy": acc, "precision": prec,
                "recall": rec, "f1": f1, "auc": auc, "specificity": spec}


# ─── Train / Validate ────────────────────────────────────────────────────────
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, config) -> Dict[str, float]:
    model.train()
    tracker = MetricTracker()
    pbar = tqdm(dataloader, desc="  [Train]", leave=False, ncols=90)
    for clips, labels in pbar:
        clips  = clips.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=config.use_amp and torch.cuda.is_available()):
            logits, _ = model(clips)
            loss = criterion(logits, labels)
        if config.use_amp and torch.cuda.is_available():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
        tracker.update(loss.item(), labels, logits)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return tracker.compute()


def validate_one_epoch(model, dataloader, criterion, config):
    model.eval()
    tracker = MetricTracker()
    with torch.no_grad():
        for clips, labels in tqdm(dataloader, desc="  [Val]  ", leave=False, ncols=90):
            clips  = clips.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=config.use_amp and torch.cuda.is_available()):
                logits, _ = model(clips)
                loss = criterion(logits, labels)
            tracker.update(loss.item(), labels, logits)
    metrics = tracker.compute()
    return metrics, np.array(tracker.all_targets), np.array(tracker.all_preds), np.array(tracker.all_probs)


# ─── Main Training Loop ──────────────────────────────────────────────────────
def run_training(
    model, train_loader, val_loader, criterion, optimizer, scheduler, config, train_df,
    start_epoch: int = 1,
    history: Optional[Dict[str, List[float]]] = None,
    best_val_f1: float = 0.0,
    patience_counter: int = 0
) -> Dict:
    scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp and torch.cuda.is_available())

    if history is None:
        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [],  "val_acc": [],
            "train_f1": [],   "val_f1": [],
            "train_auc": [],  "val_auc": [],
            "lr": []
        }

    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"  Starting Training  |  Device: {config.device}  |  Epochs: {config.num_epochs}")
    print(f"  Start Epoch: {start_epoch}  |  Batch Size: {config.batch_size}  |  AMP: {config.use_amp}  |  Workers: {config.num_workers}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, config.num_epochs + 1):
        epoch_start = time.time()

        # Backbone freeze / unfreeze
        if epoch == config.freeze_backbone_epochs + 1:
            print(f"\n[Epoch {epoch:02d}] >> Unfreezing CNN backbone for end-to-end fine-tuning.")
            model.set_backbone_trainable(True)
        elif epoch <= config.freeze_backbone_epochs:
            model.set_backbone_trainable(False)

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, config)
        val_metrics, tgts, preds, probs = validate_one_epoch(model, val_loader, criterion, config)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        history["train_auc"].append(train_metrics["auc"])
        history["val_auc"].append(val_metrics["auc"])
        history["lr"].append(current_lr)

        # Always persist history.json after every epoch
        history_path = os.path.join(config.checkpoint_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{config.num_epochs:02d}]  "
            f"Tr Loss:{train_metrics['loss']:.4f}  Tr F1:{train_metrics['f1']:.4f}  "
            f"|| "
            f"Val Loss:{val_metrics['loss']:.4f}  Val F1:{val_metrics['f1']:.4f}  "
            f"Val AUC:{val_metrics['auc']:.4f}  "
            f"| LR:{current_lr:.2e}  Time:{epoch_time:.1f}s"
        )

        # Checkpoint
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "val_metrics": val_metrics,
                "config": asdict(config),
                "history": history,
            }, config.best_model_path)
            print(f"  ✓ Best model saved → {config.best_model_path}  (Val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  · No improvement ({patience_counter}/{config.early_stopping_patience})")
            if patience_counter >= config.early_stopping_patience:
                print(f"\n[Early Stopping] No improvement for {config.early_stopping_patience} epochs. Stopping.")
                break

        # Save last checkpoint for resume
        last_ckpt_path = os.path.join(config.checkpoint_dir, "checkpoint_last.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "best_val_f1": best_val_f1,
            "patience_counter": patience_counter,
            "val_metrics": val_metrics,
            "config": asdict(config),
            "history": history,
        }, last_ckpt_path)

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  Training Complete  |  Total time: {total_time/60:.1f} min")
    print(f"  Best Val F1: {best_val_f1:.4f}")
    print(f"  Checkpoint:  {config.best_model_path}")
    print(f"{'='*70}\n")
    return history


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elderly Fall Detection Training")
    parser.add_argument("--resume", action="store_true", help="Resume training from previous checkpoint")
    args = parser.parse_args()

    config = TrainingConfig()
    seed_everything(config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"Device : {config.device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("\n[1/4] Scanning dataset …")
    df_sequences = scan_dataset_sequences(config.data_dir)
    print(f"      {len(df_sequences)} sequences  |  "
          f"{df_sequences['frame_count'].sum()} total frames")

    print("\n[2/4] Generating train & validation splits with multi-stride dense sampling …")
    train_loader, val_loader, train_df, val_df = create_data_loaders(df_sequences, config)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[3/4] Initialising model …")
    model = FallDetectorResNetLSTM(
        num_classes=config.num_classes,
        hidden_dim=config.hidden_dim,
        num_lstm_layers=config.num_lstm_layers,
        bidirectional=config.bidirectional,
        dropout_rate=config.dropout_rate,
        pretrained_backbone=config.backbone_pretrained,
        use_motion_diff=config.use_motion_diff
    ).to(config.device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Total params: {total_params:,}  |  Trainable: {trainable_params:,}")

    # ── Loss / Optimiser / Scheduler ─────────────────────────────────────────
    print("\n[4/4] Setting up loss, optimizer, and cosine scheduler …")
    class_counts = train_df["label"].value_counts().sort_index().values
    total_s = sum(class_counts)
    weights = [total_s / (len(class_counts) * c) for c in class_counts]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(config.device)
    print(f"      Class weights: {[f'{w:.3f}' for w in weights]}")
    print(f"      Label smoothing: {config.label_smoothing}  |  Dropout: {config.dropout_rate}  |  Weight decay: {config.weight_decay}")

    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0, label_smoothing=config.label_smoothing)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(),    "lr": config.backbone_lr},
        {"params": model.lstm.parameters(),        "lr": config.learning_rate},
        {"params": model.attention.parameters(),   "lr": config.learning_rate},
        {"params": model.classifier.parameters(),  "lr": config.learning_rate},
    ], weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-6
    )

    # ── Resume logic ──────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_f1 = 0.0
    patience_counter = 0
    history = None

    if args.resume:
        last_ckpt_path = os.path.join(config.checkpoint_dir, "checkpoint_last.pth")
        resume_path = last_ckpt_path if os.path.exists(last_ckpt_path) else config.best_model_path
        if os.path.exists(resume_path):
            print(f"\n[Resume] Loading state from {resume_path} ...")
            ckpt = torch.load(resume_path, map_location=config.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            else:
                for _ in range(ckpt.get("epoch", 0)):
                    scheduler.step()
            saved_epoch = ckpt.get("epoch", 0)
            start_epoch = saved_epoch + 1
            best_val_f1 = ckpt.get("val_f1", 0.0)
            patience_counter = ckpt.get("patience_counter", 0)
            history = ckpt.get("history", None)
            if start_epoch > config.freeze_backbone_epochs:
                model.set_backbone_trainable(True)
            print(f"[Resume] Resuming training from Epoch {start_epoch}/{config.num_epochs} (Previous Best Val F1: {best_val_f1:.4f})")

    # ── Train ─────────────────────────────────────────────────────────────────
    history = run_training(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        config, train_df,
        start_epoch=start_epoch,
        history=history,
        best_val_f1=best_val_f1,
        patience_counter=patience_counter
    )

    # ── Final history JSON ────────────────────────────────────────────────────
    history_path = os.path.join(config.checkpoint_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved → {history_path}")

    # ── Edge Model Export (TorchScript & ONNX) ─────────────────────────────────
    export_edge_models(model, config)


def export_edge_models(model: nn.Module, config: TrainingConfig):
    os.makedirs(config.output_dir, exist_ok=True)
    if os.path.exists(config.best_model_path):
        print(f"\n[Export] Loading best weights from {config.best_model_path} for edge deployment …")
        ckpt = torch.load(config.best_model_path, map_location=config.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    dummy_input = torch.randn(
        1, config.sequence_length, 3, config.image_size[0], config.image_size[1],
        device=config.device
    )

    # 1. TorchScript Trace
    traced_path = os.path.join(config.output_dir, "fall_detector_traced.pt")
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(traced_path)
            print(f"  ✓ TorchScript model exported → {traced_path} ({os.path.getsize(traced_path)/(1024*1024):.1f} MB)")
    except Exception as e:
        print(f"  ✗ TorchScript export failed: {e}")

    # 2. ONNX Export
    onnx_path = os.path.join(config.output_dir, "fall_detector.onnx")
    try:
        with torch.no_grad():
            torch.onnx.export(
                model, dummy_input, onnx_path,
                input_names=["video_clip"],
                output_names=["logits", "attention_weights"],
                dynamic_axes={"video_clip": {0: "batch_size"}, "logits": {0: "batch_size"}},
                opset_version=14
            )
            print(f"  ✓ ONNX model exported → {onnx_path} ({os.path.getsize(onnx_path)/(1024*1024):.1f} MB)")
    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}")
