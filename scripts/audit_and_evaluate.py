"""
scripts/audit_and_evaluate.py
Audit checkpoints, dataset splits, leakage, and run genuine validation evaluation.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score
)

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_training import (
    TrainingConfig,
    seed_everything,
    scan_dataset_sequences,
    create_data_loaders,
    FallDetectorResNetLSTM,
    validate_one_epoch,
    FocalLoss,
)

def audit_checkpoints():
    print("=" * 60)
    print("1. CHECKPOINT AUDIT")
    print("=" * 60)
    
    checkpoints = [
        "checkpoints/best_fall_detector.pth",
        "checkpoints/checkpoint_last.pth",
        "checkpoints/strategy1_best_fall_detector.pth"
    ]
    
    for ckpt_path in checkpoints:
        if not os.path.exists(ckpt_path):
            print(f"[MISSING] {ckpt_path}")
            continue
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            print(f"\n[EXISTS] {ckpt_path} ({os.path.getsize(ckpt_path) / 1e6:.1f} MB)")
            print(f"  Keys: {list(ckpt.keys())}")
            if "epoch" in ckpt:
                print(f"  Saved at epoch: {ckpt['epoch']}")
            if "val_f1" in ckpt:
                print(f"  Saved val_f1: {ckpt['val_f1']}")
            if "best_val_f1" in ckpt:
                print(f"  Saved best_val_f1: {ckpt['best_val_f1']}")
            if "val_metrics" in ckpt:
                print(f"  Saved val_metrics: {ckpt['val_metrics']}")
            if "config" in ckpt:
                c = ckpt["config"]
                print(f"  Config: seq_len={c.get('sequence_length')}, lr={c.get('learning_rate')}, "
                      f"use_motion_diff={c.get('use_motion_diff')}, seed={c.get('seed')}, "
                      f"epochs={c.get('num_epochs')}, val_ratio={c.get('val_split_ratio')}")
        except Exception as e:
            print(f"[ERROR loading {ckpt_path}]: {e}")

def audit_histories():
    print("\n" + "=" * 60)
    print("2. TRAINING HISTORIES AUDIT")
    print("=" * 60)
    
    for hist_path in ["checkpoints/history.json", "checkpoints/strategy1_history.json"]:
        if os.path.exists(hist_path):
            with open(hist_path, "r") as f:
                h = json.load(f)
            num_epochs = len(h.get("train_loss", []))
            print(f"\n[HISTORY] {hist_path}: {num_epochs} epochs recorded")
            print(f"  Metrics logged: {list(h.keys())}")
            print(f"  Final train_loss: {h['train_loss'][-1]:.4f}, val_loss: {h['val_loss'][-1]:.4f}")
            print(f"  Final train_acc: {h['train_acc'][-1]:.4f}, val_acc: {h['val_acc'][-1]:.4f}")
            print(f"  Final val_f1: {h['val_f1'][-1]:.4f}, val_auc: {h['val_auc'][-1]:.4f}")
            best_f1_idx = int(np.argmax(h["val_f1"]))
            print(f"  Best val_f1 was at epoch {best_f1_idx + 1}: val_f1={h['val_f1'][best_f1_idx]:.4f}, val_acc={h['val_acc'][best_f1_idx]:.4f}, val_loss={h['val_loss'][best_f1_idx]:.4f}")
            best_loss_idx = int(np.argmin(h["val_loss"]))
            print(f"  Min val_loss was at epoch {best_loss_idx + 1}: val_loss={h['val_loss'][best_loss_idx]:.4f}")

def audit_splits_and_evaluate():
    print("\n" + "=" * 60)
    print("3. DATASET SPLITS, LEAKAGE AUDIT & GENUINE EVALUATION")
    print("=" * 60)
    
    config = TrainingConfig()
    seed_everything(config.seed)
    
    print(f"PyTorch Device: {config.device}")
    df_seqs = scan_dataset_sequences(config.data_dir)
    print(f"Scanned {len(df_seqs)} sequences in {config.data_dir}")
    
    train_loader, val_loader, train_df, val_df = create_data_loaders(df_seqs, config)
    
    train_seq_names = set(train_df["sequence_name"].unique())
    val_seq_names = set(val_df["sequence_name"].unique())
    
    overlap = train_seq_names.intersection(val_seq_names)
    print(f"\nTrain sequences ({len(train_seq_names)}): {sorted(list(train_seq_names))}")
    print(f"Val sequences ({len(val_seq_names)}): {sorted(list(val_seq_names))}")
    print(f"Sequence Overlap (Data Leakage between Train and Val): {overlap}")
    if len(overlap) == 0:
        print("  -> ZERO SEQUENCE LEAKAGE: Validation sequences are completely disjoint from training sequences!")
    else:
        print(f"  -> WARNING: LEAKAGE DETECTED across sequences: {overlap}")
        
    print(f"\nTrain clips: {len(train_df)} | Class distribution: {train_df['label'].value_counts().to_dict()}")
    print(f"Val clips:   {len(val_df)}   | Class distribution: {val_df['label'].value_counts().to_dict()}")
    
    # Check if best checkpoint exists and evaluate it
    if os.path.exists(config.best_model_path):
        print(f"\nEvaluating genuine checkpoint: {config.best_model_path} ...")
        ckpt = torch.load(config.best_model_path, map_location=config.device, weights_only=False)
        model = FallDetectorResNetLSTM(
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim,
            num_lstm_layers=config.num_lstm_layers,
            bidirectional=config.bidirectional,
            dropout_rate=config.dropout_rate,
            pretrained_backbone=False,
            use_motion_diff=config.use_motion_diff
        ).to(config.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        class_weights = torch.tensor([1.386, 0.782], dtype=torch.float32).to(config.device)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=config.label_smoothing)
        
        metrics, y_true, y_pred, y_prob = validate_one_epoch(model, val_loader, criterion, config)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        acc = (tp + tn) / len(y_true)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        
        print("\n--- GENUINE VALIDATION RESULTS ---")
        print(f"Total evaluated samples: {len(y_true)}")
        print(f"Ground Truth: {np.sum(y_true == 0)} Non-Fall (Class 0), {np.sum(y_true == 1)} Fall (Class 1)")
        print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
        print(f"Precision:   {prec:.4f}")
        print(f"Recall:      {sens:.4f}")
        print(f"Specificity: {spec:.4f}")
        print(f"F1-Score:    {f1:.4f}")
        print(f"ROC-AUC:     {auc:.4f}")
        print(f"Avg Prec:    {ap:.4f}")
        
        # Save evaluated results to a temporary audit json
        audit_res = {
            "checkpoint": config.best_model_path,
            "saved_epoch": ckpt.get("epoch"),
            "val_f1_saved": ckpt.get("val_f1"),
            "evaluated_metrics": {
                "sample_count": int(len(y_true)),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(sens),
                "specificity": float(spec),
                "f1_score": float(f1),
                "roc_auc": float(auc),
                "average_precision": float(ap)
            },
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist()
        }
        with open("outputs/audit_evaluation_results.json", "w") as f:
            json.dump(audit_res, f, indent=2)
        print("  -> Saved outputs/audit_evaluation_results.json")

if __name__ == "__main__":
    audit_checkpoints()
    audit_histories()
    audit_splits_and_evaluate()
