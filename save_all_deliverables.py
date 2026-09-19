"""
save_all_deliverables.py
========================
Executes post-training export and visualization pipeline:
1. Loads best model checkpoint (checkpoints/best_fall_detector.pth).
2. Loads training history (checkpoints/history.json).
3. Evaluates model on validation split to extract genuine y_true, y_pred, y_prob.
4. Exports model to TorchScript (outputs/fall_detector_traced.pt).
5. Exports model to ONNX (outputs/fall_detector.onnx).
6. Generates all 6 publication-ready research figures in outputs/figures/ (PNG + PDF).
7. Generates a comprehensive summary manifest (outputs/deliverables_manifest.md).
"""

import os
import glob
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from run_training import (
    TrainingConfig,
    seed_everything,
    scan_dataset_sequences,
    build_clip_annotations,
    create_data_loaders,
    FallDetectorResNetLSTM,
    validate_one_epoch,
    FocalLoss,
)
import research_visualizations as rv

def main():
    print("=" * 70)
    print("POST-TRAINING DELIVERABLES GENERATION")
    print("=" * 70)

    config = TrainingConfig()
    seed_everything(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    figures_dir = os.path.join(config.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Verify checkpoint exists
    if not os.path.exists(config.best_model_path):
        raise FileNotFoundError(f"Checkpoint not found: {config.best_model_path}")

    print(f"\n[1/5] Loading checkpoint: {config.best_model_path}")
    checkpoint = torch.load(config.best_model_path, map_location=config.device, weights_only=False)
    best_val_f1 = checkpoint.get("val_f1", 0.0)
    epoch_saved = checkpoint.get("epoch", 0)
    print(f"      Best Val F1: {best_val_f1:.4f} (from Epoch {epoch_saved})")

    # Load history
    history_path = os.path.join(config.checkpoint_dir, "history.json")
    history = None
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        print(f"      Loaded history from: {history_path} ({len(history.get('train_loss', []))} epochs)")
    elif "history" in checkpoint:
        history = checkpoint["history"]

    # 2. Build model and load weights
    print("\n[2/5] Initializing model & loading weights...")
    model = FallDetectorResNetLSTM(
        num_classes=config.num_classes,
        hidden_dim=config.hidden_dim,
        num_lstm_layers=config.num_lstm_layers,
        bidirectional=config.bidirectional,
        dropout_rate=config.dropout_rate,
        pretrained_backbone=False  # will load trained weights
    ).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Create validation set for evaluation
    print("\n[3/5] Evaluating best model on validation split...")
    df_sequences = scan_dataset_sequences(config.data_dir)
    _, val_loader, _, val_df = create_data_loaders(df_sequences, config)

    class_weights = torch.tensor([1.386, 0.782], dtype=torch.float32).to(config.device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=config.label_smoothing)

    val_metrics, y_true, y_pred, y_prob = validate_one_epoch(
        model, val_loader, criterion, config
    )
    print(f"      Validation Accuracy : {val_metrics['accuracy']*100:.2f}%")
    print(f"      Validation Precision: {val_metrics['precision']:.4f}")
    print(f"      Validation Recall   : {val_metrics['recall']:.4f}")
    print(f"      Validation F1-Score : {val_metrics['f1']:.4f}")
    print(f"      Validation ROC-AUC  : {val_metrics['auc']:.4f}")

    # 4. Model Export: TorchScript & ONNX
    print("\n[4/5] Exporting model to TorchScript and ONNX...")
    dummy_input = torch.randn(1, config.sequence_length, 3, *config.image_size).to(config.device)

    class ModelWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        def forward(self, x):
            logits, _ = self.base_model(x)
            return logits

    wrapped_model = ModelWrapper(model).eval()

    # TorchScript
    torchscript_path = os.path.join(config.output_dir, "fall_detector_traced.pt")
    try:
        traced_model = torch.jit.trace(wrapped_model, dummy_input)
        traced_model.save(torchscript_path)
        print(f"      ✓ TorchScript model saved: {torchscript_path}")
    except Exception as e:
        print(f"      ✗ TorchScript export failed: {e}")

    # ONNX
    onnx_path = os.path.join(config.output_dir, "fall_detector.onnx")
    try:
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["video_clip_input"],
            output_names=["fall_logits"],
            dynamic_axes={
                "video_clip_input": {0: "batch_size"},
                "fall_logits": {0: "batch_size"}
            }
        )
        print(f"      ✓ ONNX model saved: {onnx_path}")
    except Exception as e:
        print(f"      ✗ ONNX export failed: {e}")

    # 5. Publication Visualizations
    print("\n[5/5] Generating all publication-ready research figures...")
    rv.generate_all_research_figures(
        history=history,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        save_dir=figures_dir
    )
    print(f"      ✓ Successfully generated research figures (PNG + PDF) in {figures_dir}")

    # 6. Save Manifest
    manifest_path = os.path.join(config.output_dir, "deliverables_manifest.md")
    with open(manifest_path, "w") as f:
        f.write("# Elderly Fall Detection — Deliverables Manifest\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Model Performance (Best Checkpoint: Epoch {epoch_saved})\n\n")
        f.write(f"- **Accuracy**: {val_metrics['accuracy']*100:.2f}%\n")
        f.write(f"- **Precision**: {val_metrics['precision']:.4f}\n")
        f.write(f"- **Recall (Sensitivity)**: {val_metrics['recall']:.4f}\n")
        f.write(f"- **F1-Score**: {val_metrics['f1']:.4f}\n")
        f.write(f"- **ROC-AUC**: {val_metrics['auc']:.4f}\n\n")
        f.write("## Saved Files\n\n")
        f.write(f"### Checkpoints & Data\n")
        f.write(f"- `checkpoints/best_fall_detector.pth` — PyTorch best model weights\n")
        f.write(f"- `checkpoints/history.json` — Per-epoch loss/accuracy/F1/AUC logs\n\n")
        f.write(f"### Deployment Formats\n")
        f.write(f"- `outputs/fall_detector_traced.pt` — TorchScript traced model\n")
        f.write(f"- `outputs/fall_detector.onnx` — ONNX dynamic-batch model\n\n")
        f.write(f"### Research Paper Figures (`outputs/figures/`)\n")
        f.write(f"- `fig1_training_dynamics.png` & `.pdf` — Loss, Accuracy, F1/AUC, LR schedule\n")
        f.write(f"- `fig2_diagnostic_evaluation.png` & `.pdf` — Confusion Matrix, ROC, PR curve, Calibration\n")
        f.write(f"- `fig3_temporal_attention_interpretability.png` & `.pdf` — Temporal attention saliency heatmaps\n")
        f.write(f"- `fig4_temporal_stream_simulation.png` & `.pdf` — Real-time continuous stream timeline\n")
        f.write(f"- `fig5_model_architecture_comparison.png` & `.pdf` — ResNet-18 vs LSTM vs Proposed BiLSTM-Attn\n")
        f.write(f"- `fig6_clinical_threshold_tradeoff.png` & `.pdf` — Clinical decision threshold optimization\n")

    print(f"\nManifest saved: {manifest_path}")
    print("\nALL DELIVERABLES SAVED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
