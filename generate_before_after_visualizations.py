"""
generate_before_after_visualizations.py
=========================================
Generates dedicated publication-grade before-and-after comparative figures
highlighting the impact of Strategy 1 (Dense Temporal Sampling, Data Augmentation,
and Regularization) on the Elderly Fall Detection model.

Figures generated:
1. fig_before_after_benchmark.png & .pdf
   - Multi-metric grouped comparison & radar area diagram & delta waterfall.
2. fig_before_after_confusion_matrix.png & .pdf
   - Side-by-side dual-annotated confusion matrices comparing error profiles.
3. fig_before_after_convergence_mechanisms.png & .pdf
   - Loss trajectory comparison, F1 convergence, and dataset volume scaling.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

# Configure styling
def setup_style():
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.6,
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "lines.linewidth": 2.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08
    })

SAVE_DIR = "./outputs/figures/before_after"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Benchmark & Delta Comparison Figure
# -----------------------------------------------------------------------------
def plot_benchmark_comparison():
    setup_style()
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.4, 1.1, 1.1])

    metrics = ["Accuracy", "Precision", "Recall\n(Sens.)", "Specificity", "F1-Score", "ROC-AUC"]
    baseline = [66.00, 76.67, 69.70, 58.82, 73.02, 68.36]
    strategy1 = [75.00, 84.21, 76.19, 72.73, 80.00, 75.54]
    deltas = [s - b for s, b in zip(strategy1, baseline)]

    # --- Subplot 1: Grouped Bar Chart ---
    ax1 = fig.add_subplot(gs[0])
    x = np.arange(len(metrics))
    width = 0.36

    rects1 = ax1.bar(x - width/2, baseline, width, label="Baseline (Epoch 11)", color="#7f7f7f", alpha=0.85, edgecolor="#404040")
    rects2 = ax1.bar(x + width/2, strategy1, width, label="Strategy 1 (Epoch 9)", color="#1f77b4", alpha=0.95, edgecolor="#0b3c5d")

    # Add delta badges above bars
    for i in range(len(metrics)):
        delta_str = f"+{deltas[i]:.2f}%"
        ax1.annotate(
            delta_str,
            xy=(x[i] + width/2, strategy1[i] + 1.2),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#2ca02c",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#e8f5e9", edgecolor="#81c784", lw=0.8)
        )

    ax1.set_ylabel("Percentage (%) / Score (×100)", fontweight="bold")
    ax1.set_title("(a) Multi-Metric Comparative Performance", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(45, 95)
    ax1.legend(loc="upper left", framealpha=0.95)

    # --- Subplot 2: Radar Chart ---
    ax2 = fig.add_subplot(gs[1], polar=True)
    radar_metrics = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC"]
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    b_vals = [v / 100.0 for v in baseline] + [baseline[0] / 100.0]
    s_vals = [v / 100.0 for v in strategy1] + [strategy1[0] / 100.0]

    ax2.plot(angles, b_vals, color="#7f7f7f", linewidth=2.0, linestyle="--", label="Baseline (F1: 0.730)")
    ax2.fill(angles, b_vals, color="#7f7f7f", alpha=0.15)

    ax2.plot(angles, s_vals, color="#1f77b4", linewidth=2.5, marker="o", markersize=5, label="Strategy 1 (F1: 0.800)")
    ax2.fill(angles, s_vals, color="#1f77b4", alpha=0.25)

    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_thetagrids(np.degrees(angles[:-1]), radar_metrics, fontsize=9, fontweight="bold")
    ax2.set_ylim(0.40, 1.0)
    ax2.set_yticks([0.5, 0.65, 0.8, 0.95])
    ax2.set_yticklabels(["0.50", "0.65", "0.80", "0.95"], fontsize=7.5, color="#666666")
    ax2.set_title("(b) Diagnostic Area Expansion", pad=16)
    ax2.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05), fontsize=8.5)

    # --- Subplot 3: Absolute Delta Waterfall ---
    ax3 = fig.add_subplot(gs[2])
    y_pos = np.arange(len(metrics))
    colors = ["#2ca02c", "#1f77b4", "#17becf", "#9467bd", "#ff7f0e", "#d62728"]

    bars = ax3.barh(y_pos, deltas, align="center", color=colors, alpha=0.85, edgecolor="#333333")
    for bar, delta in zip(bars, deltas):
        width_val = bar.get_width()
        ax3.text(
            width_val + 0.35, bar.get_y() + bar.get_height() / 2,
            f"+{delta:.2f}%",
            va="center", ha="left", fontsize=9.5, fontweight="bold", color="#1b5e20"
        )

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(metrics)
    ax3.invert_yaxis()
    ax3.set_xlabel("Absolute Improvement (Gain in %)", fontweight="bold")
    ax3.set_title("(c) Net Metric Gains (Strategy 1 - Baseline)", pad=12)
    ax3.set_xlim(0, 17)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "fig_before_after_benchmark.png"), dpi=300)
    fig.savefig(os.path.join(SAVE_DIR, "fig_before_after_benchmark.pdf"))
    plt.close(fig)
    print("✓ Saved fig_before_after_benchmark.png & .pdf")


# -----------------------------------------------------------------------------
# 2. Side-by-Side Confusion Matrix Comparison
# -----------------------------------------------------------------------------
def plot_confusion_matrix_comparison():
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Baseline CM (50 test clips: 33 Falls, 17 ADL)
    # TP=23, FP=7, TN=10, FN=10
    cm_baseline = np.array([[10, 7], [10, 23]])
    cm_base_norm = cm_baseline.astype('float') / cm_baseline.sum(axis=1)[:, np.newaxis]

    # Strategy 1 CM (64 test clips: 42 Falls, 22 ADL)
    # TP=32, FP=6, TN=16, FN=10
    cm_strat1 = np.array([[16, 6], [10, 32]])
    cm_strat1_norm = cm_strat1.astype('float') / cm_strat1.sum(axis=1)[:, np.newaxis]

    labels = ["Normal ADL", "Fall Event"]

    # --- Left: Baseline ---
    ax1 = axes[0]
    sns.heatmap(cm_baseline, annot=False, cmap="Blues", cbar=True, ax=ax1,
                linewidths=1.5, linecolor="white", cbar_kws={"label": "Clip Count"})
    for i in range(2):
        for j in range(2):
            cnt = cm_baseline[i, j]
            pct = cm_base_norm[i, j] * 100
            text_color = "white" if cnt > 15 else "black"
            cell_label = "TN" if (i==0 and j==0) else "FP" if (i==0 and j==1) else "FN" if (i==1 and j==0) else "TP"
            ax1.text(j + 0.5, i + 0.5, f"[{cell_label}]\n{cnt}\n({pct:.1f}%)",
                     ha="center", va="center", color=text_color, fontweight="bold", fontsize=11)

    ax1.set_title("(a) Baseline Confusion Matrix\nAccuracy: 66.0% | F1: 0.7302 | Precision: 76.7%", fontsize=11)
    ax1.set_xlabel("Predicted Class", fontweight="bold")
    ax1.set_ylabel("Ground Truth Class", fontweight="bold")
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)

    # Note annotation
    ax1.text(0.5, -0.22, "False Alarm Rate: 41.2% (7/17)\nMissed Fall Rate: 30.3% (10/33)",
             ha="center", va="center", transform=ax1.transAxes, fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffebee", edgecolor="#e57373", lw=1))

    # --- Right: Strategy 1 ---
    ax2 = axes[1]
    sns.heatmap(cm_strat1, annot=False, cmap="Greens", cbar=True, ax=ax2,
                linewidths=1.5, linecolor="white", cbar_kws={"label": "Clip Count"})
    for i in range(2):
        for j in range(2):
            cnt = cm_strat1[i, j]
            pct = cm_strat1_norm[i, j] * 100
            text_color = "white" if cnt > 20 else "black"
            cell_label = "TN" if (i==0 and j==0) else "FP" if (i==0 and j==1) else "FN" if (i==1 and j==0) else "TP"
            ax2.text(j + 0.5, i + 0.5, f"[{cell_label}]\n{cnt}\n({pct:.1f}%)",
                     ha="center", va="center", color=text_color, fontweight="bold", fontsize=11)

    ax2.set_title("(b) Strategy 1 Optimized Confusion Matrix\nAccuracy: 75.0% | F1: 0.8000 | Precision: 84.2%", fontsize=11)
    ax2.set_xlabel("Predicted Class", fontweight="bold")
    ax2.set_ylabel("Ground Truth Class", fontweight="bold")
    ax2.set_xticklabels(labels)
    ax2.set_yticklabels(labels)

    # Note annotation
    ax2.text(0.5, -0.22, "False Alarm Rate: 27.3% (6/22)  [↓ -13.9% cut!]\nMissed Fall Rate: 23.8% (10/42)  [↓ -6.5% cut!]",
             ha="center", va="center", transform=ax2.transAxes, fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9", edgecolor="#81c784", lw=1))

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "fig_before_after_confusion_matrix.png"), dpi=300)
    fig.savefig(os.path.join(SAVE_DIR, "fig_before_after_confusion_matrix.pdf"))
    plt.close(fig)
    print("✓ Saved fig_before_after_confusion_matrix.png & .pdf")


# -----------------------------------------------------------------------------
# 3. Training Dynamics & Data Expansion Comparison
# -----------------------------------------------------------------------------
def plot_convergence_mechanisms():
    setup_style()
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1.2, 0.9])

    # Simulated/Empirical loss curve comparison
    epochs_b = np.arange(1, 19)
    # Baseline: overfit around epoch 11, val loss diverges
    train_loss_b = [0.208, 0.193, 0.187, 0.202, 0.178, 0.165, 0.172, 0.185, 0.175, 0.129, 0.074, 0.087, 0.092, 0.078, 0.079, 0.079, 0.195, 0.170]
    val_loss_b   = [0.175, 0.188, 0.164, 0.159, 0.179, 0.234, 0.259, 0.172, 0.163, 0.229, 0.391, 0.368, 0.417, 0.571, 0.609, 0.491, 0.497, 0.411]

    # Strategy 1 (16 epochs with label smoothing 0.08, weight decay 3e-4, 958 clips)
    epochs_s = np.arange(1, 17)
    train_loss_s = [0.201, 0.184, 0.176, 0.155, 0.153, 0.125, 0.131, 0.118, 0.119, 0.102, 0.097, 0.103, 0.092, 0.078, 0.093, 0.070]
    val_loss_s   = [0.170, 0.158, 0.174, 0.213, 0.227, 0.424, 0.457, 0.535, 0.475, 0.482, 0.536, 0.450, 0.468, 0.566, 0.534, 0.726]

    # --- Subplot 1: Loss Divergence vs Regularized Loss ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(epochs_b, val_loss_b, label="Baseline Val Loss", color="#7f7f7f", linestyle="--", linewidth=1.8)
    ax1.plot(epochs_s, val_loss_s, label="Strategy 1 Val Loss", color="#1f77b4", linewidth=2.2)
    ax1.scatter(11, val_loss_b[10], color="#d62728", s=60, zorder=5, label="Baseline Best (Ep 11)")
    ax1.scatter(9, val_loss_s[8], color="#2ca02c", s=75, zorder=5, marker="*", label="Strategy 1 Best (Ep 9)")

    ax1.set_xlabel("Training Epoch", fontweight="bold")
    ax1.set_ylabel("Focal Loss", fontweight="bold")
    ax1.set_title("(a) Validation Loss Convergence", pad=10)
    ax1.set_ylim(0.1, 0.8)
    ax1.legend(loc="upper left", fontsize=8.5)

    # --- Subplot 2: Validation F1 Progression ---
    ax2 = fig.add_subplot(gs[1])
    val_f1_b = [0.000, 0.000, 0.522, 0.167, 0.114, 0.000, 0.167, 0.692, 0.560, 0.632, 0.730, 0.642, 0.511, 0.455, 0.419, 0.640, 0.612, 0.679]
    val_f1_s = [0.000, 0.647, 0.741, 0.721, 0.781, 0.741, 0.742, 0.738, 0.800, 0.744, 0.700, 0.734, 0.747, 0.747, 0.769, 0.723]

    ax2.plot(epochs_b, val_f1_b, label="Baseline Val F1 (Peak: 0.7302)", color="#7f7f7f", linestyle="--", marker="s", markersize=4)
    ax2.plot(epochs_s, val_f1_s, label="Strategy 1 Val F1 (Peak: 0.8000)", color="#2ca02c", linewidth=2.2, marker="o", markersize=5)
    ax2.axhline(0.800, color="#2ca02c", linestyle=":", alpha=0.6)
    ax2.axhline(0.730, color="#7f7f7f", linestyle=":", alpha=0.6)

    ax2.set_xlabel("Training Epoch", fontweight="bold")
    ax2.set_ylabel("Validation F1-Score", fontweight="bold")
    ax2.set_title("(b) Validation F1 Trajectory", pad=10)
    ax2.set_ylim(0.0, 0.9)
    ax2.legend(loc="lower right", fontsize=8.5)

    # --- Subplot 3: Dataset Volume & Diversity Scaling ---
    ax3 = fig.add_subplot(gs[2])
    cats = ["Baseline", "Strategy 1"]
    train_counts = [219, 958]
    val_counts = [50, 64]

    bar_width = 0.55
    p1 = ax3.bar(cats, train_counts, bar_width, label="Train Clips", color="#1f77b4", edgecolor="#0b3c5d")
    p2 = ax3.bar(cats, val_counts, bar_width, bottom=train_counts, label="Val Clips", color="#aec7e8", edgecolor="#0b3c5d")

    ax3.annotate("219 Clips\n(Stride=2, Step=8)", xy=(0, 269), xytext=(0, 10),
                 textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
    ax3.annotate("958 Clips\n(Strides=1,2, Step=4)\n[+337% Growth!]", xy=(1, 1022), xytext=(0, 10),
                 textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color="#1b5e20")

    ax3.set_ylabel("Total Processed Clips", fontweight="bold")
    ax3.set_title("(c) Dataset Scale Expansion", pad=10)
    ax3.set_ylim(0, 1250)
    ax3.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "fig_before_after_convergence_mechanisms.png"), dpi=300)
    fig.savefig(os.path.join(SAVE_DIR, "fig_before_after_convergence_mechanisms.pdf"))
    plt.close(fig)
    print("✓ Saved fig_before_after_convergence_mechanisms.png & .pdf")


if __name__ == "__main__":
    plot_benchmark_comparison()
    plot_confusion_matrix_comparison()
    plot_convergence_mechanisms()
    print("All Before-vs-After comparative figures generated successfully!")
