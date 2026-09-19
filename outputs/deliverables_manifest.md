# Elderly Fall Detection — Deliverables Manifest

**Generated:** 2026-09-19 14:23:05

## Model Performance (Best Checkpoint: Epoch 9)

- **Accuracy**: 75.00%
- **Precision**: 0.8421
- **Recall (Sensitivity)**: 0.7619
- **F1-Score**: 0.8000
- **ROC-AUC**: 0.7554

## Saved Files

### Checkpoints & Data
- `checkpoints/best_fall_detector.pth` — PyTorch best model weights
- `checkpoints/history.json` — Per-epoch loss/accuracy/F1/AUC logs

### Deployment Formats
- `outputs/fall_detector_traced.pt` — TorchScript traced model
- `outputs/fall_detector.onnx` — ONNX dynamic-batch model

### Research Paper Figures (`outputs/figures/`)
- `fig1_training_dynamics.png` & `.pdf` — Loss, Accuracy, F1/AUC, LR schedule
- `fig2_diagnostic_evaluation.png` & `.pdf` — Confusion Matrix, ROC, PR curve, Calibration
- `fig3_temporal_attention_interpretability.png` & `.pdf` — Temporal attention saliency heatmaps
- `fig4_temporal_stream_simulation.png` & `.pdf` — Real-time continuous stream timeline
- `fig5_model_architecture_comparison.png` & `.pdf` — ResNet-18 vs LSTM vs Proposed BiLSTM-Attn
- `fig6_clinical_threshold_tradeoff.png` & `.pdf` — Clinical decision threshold optimization
