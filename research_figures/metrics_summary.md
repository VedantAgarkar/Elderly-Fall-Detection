# IEEE Fall Detection Paper — Verified Experimental Metrics

- **Evaluated Checkpoint**: `./checkpoints/best_fall_detector.pth` (Epoch 4)
- **Dataset Split**: URFD Sequence-Stratified Disjoint Validation Set
- **Total Evaluated Samples**: 129 clips across 8 unseen video sequences
- **Class Breakdown**: 66 Non-Fall (ADL) clips, 63 Fall clips

## 1. Classification Metrics

| Metric | Computed Value | Description |
| :--- | :--- | :--- |
| **Accuracy** | **96.12%** | Overall classification accuracy (124/129) |
| **Precision (PPV)** | **100.00%** | Positive predictive value (0 false alarms) |
| **Recall (Sensitivity)** | **92.06%** | True positive detection rate (58/63) |
| **Specificity (TNR)** | **100.00%** | True negative rate on ADLs (66/66) |
| **F1-Score** | **0.9587** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | **0.9817** | Area under Receiver Operating Characteristic |
| **Average Precision** | **0.9871** | Area under Precision-Recall curve |

## 2. Confusion Matrix Counts

| | Predicted Non-Fall (ADL) | Predicted Fall |
| :--- | :---: | :---: |
| **True Non-Fall (ADL)** | **TN = 66** (100.0%) | **FP = 0** (0.0%) |
| **True Fall** | **FN = 5** (7.94%) | **TP = 58** (92.06%) |

## 3. Hardware Latency Profile

- **Benchmark Hardware**: NVIDIA GeForce GTX 1650 with Max-Q Design
- **PyTorch FP32 Latency**: 30.65 ± 0.23 ms (32.6 clips/s)
- **PyTorch AMP (FP16)**: 124.14 ± 4.54 ms (8.1 clips/s)
- **End-to-End Pipeline**: 50.63 ± 1.43 ms (19.7 clips/s)
