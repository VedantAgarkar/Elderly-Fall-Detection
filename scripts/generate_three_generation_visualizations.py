"""
Three-Generation Visual Benchmark Generator
Compares Baseline (Gen 1) vs Strategy 1 (Gen 2) vs Strategy 2 (Gen 3).
Generates publication-quality charts in PNG and PDF.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path

FIGURES_DIR = Path("outputs/figures/three_generation")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
})

def plot_three_generation_benchmark(metrics_gen1, metrics_gen2, metrics_gen3):
    """Grouped bar chart comparing all 6 metrics across three model generations."""
    fig, ax = plt.subplots(figsize=(12, 6.5))

    metric_keys = ["accuracy", "f1", "precision", "recall", "specificity", "auc"]
    metric_labels = ["Accuracy", "F1-Score", "Precision", "Recall (Sensitivity)", "Specificity", "ROC-AUC"]

    vals_gen1 = [metrics_gen1[k] * 100 for k in metric_keys]
    vals_gen2 = [metrics_gen2[k] * 100 for k in metric_keys]
    vals_gen3 = [metrics_gen3[k] * 100 for k in metric_keys]

    x = np.arange(len(metric_labels))
    width = 0.26

    rects1 = ax.bar(x - width, vals_gen1, width, label="Gen 1: Baseline (219 clips, Heuristic)", color="#94A3B8", edgecolor="#475569", linewidth=1.2)
    rects2 = ax.bar(x, vals_gen2, width, label="Gen 2: Strategy 1 (+Augmentations, 958 clips)", color="#3B82F6", edgecolor="#1D4ED8", linewidth=1.2)
    rects3 = ax.bar(x + width, vals_gen3, width, label="Gen 3: Strategy 2 (+Genuine ADLs + Motion Diff, 1,448 clips)", color="#10B981", edgecolor="#047857", linewidth=1.2)

    # Value labels
    for rects, color in [(rects1, "#334155"), (rects2, "#1E3A8A"), (rects3, "#064E3B")]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color)

    ax.set_ylabel("Performance Metric Score (%)", fontweight="bold")
    ax.set_title("Evolution of Elderly Fall Detection Performance Across Three Model Iterations", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", framealpha=0.95, facecolor="#F8FAFC", edgecolor="#CBD5E1")

    # Add gain annotation banner
    f1_gain_total = (metrics_gen3["f1"] - metrics_gen1["f1"]) * 100
    acc_gain_total = (metrics_gen3["accuracy"] - metrics_gen1["accuracy"]) * 100
    spec_gain_total = (metrics_gen3["specificity"] - metrics_gen1["specificity"]) * 100
    banner = f"Cumulative Net Improvements: F1: +{f1_gain_total:.1f}% | Accuracy: +{acc_gain_total:.1f}% | Specificity: +{spec_gain_total:.1f}%"
    ax.text(0.98, 0.04, banner, transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ECFDF5", edgecolor="#059669", linewidth=1.2),
            fontsize=10, fontweight="bold", color="#065F46")

    plt.tight_layout()
    out_png = FIGURES_DIR / "fig_three_generation_benchmark.png"
    out_pdf = FIGURES_DIR / "fig_three_generation_benchmark.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    print(f"Saved: {out_png}")

def plot_confusion_matrix_progression(cm_gen1, cm_gen2, cm_gen3):
    """Side-by-side progression of confusion matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    cms = [cm_gen1, cm_gen2, cm_gen3]
    titles = [
        "Gen 1: Baseline\n(Accuracy: 66.0% | Specificity: 58.8%)",
        "Gen 2: Strategy 1\n(Accuracy: 75.0% | Specificity: 72.7%)",
        "Gen 3: Strategy 2 (+Motion Diff + ADLs)\n(Accuracy: 95.35% | Specificity: 100.0%)"
    ]
    cmaps = ["Blues", "Blues", "Greens"]

    for ax, cm, title, cmap in zip(axes, cms, titles, cmaps):
        # Normalize by row
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        annot = np.empty_like(cm, dtype=object)
        for i in range(2):
            for j in range(2):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

        sns.heatmap(cm_norm, annot=annot, fmt="", cmap=cmap, ax=ax, cbar=False,
                    vmin=0, vmax=1, linewidths=1.5, linecolor="white",
                    annot_kws={"fontsize": 11, "fontweight": "bold"})

        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Class", fontweight="bold")
        ax.set_ylabel("True Ground Truth", fontweight="bold")
        ax.set_xticklabels(["Non-Fall (ADL)", "Fall"], fontweight="bold")
        ax.set_yticklabels(["Non-Fall (ADL)", "Fall"], fontweight="bold")

    plt.suptitle("Confusion Matrix Progression Across Architectural & Data Iterations", fontweight="bold", y=1.02)
    plt.tight_layout()
    out_png = FIGURES_DIR / "fig_three_generation_confusion_matrices.png"
    out_pdf = FIGURES_DIR / "fig_three_generation_confusion_matrices.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    print(f"Saved: {out_png}")

def plot_motion_differential_concept():
    """Conceptual diagram illustrating spatial features vs frame-differencing velocity."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")

    # Draw boxes
    box_style = dict(boxstyle="round,pad=0.6", linewidth=1.5)

    # Input clip
    ax.text(0.12, 0.70, "Input Video Clip\n$X \\in \\mathbb{R}^{T \\times 3 \\times 224 \\times 224}$\n($T=16$ frames)",
            ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#EFF6FF", edgecolor="#3B82F6", linewidth=1.5),
            fontsize=10, fontweight="bold")

    # ResNet backbone
    ax.text(0.38, 0.70, "ResNet-18 Backbone\n(Frozen 2 Epochs $\\to$ Fine-Tuned)\n$\\mathbf{f}_t \\in \\mathbb{R}^{512}$",
            ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3E8FF", edgecolor="#8B5CF6", linewidth=1.5),
            fontsize=10, fontweight="bold")

    # Split into appearance and velocity
    ax.text(0.68, 0.85, "Spatial Posture Features\n$\\mathbf{f}_t$ (Body angle, ground proximity)",
            ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF3C7", edgecolor="#D97706", linewidth=1.2),
            fontsize=9.5, fontweight="bold")

    ax.text(0.68, 0.55, "Temporal Feature Velocity\n$\\vec{v}_t = \\mathbf{f}_t - \\mathbf{f}_{t-1}$ (Impact acceleration)",
            ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", facecolor="#DCFCE7", edgecolor="#10B981", linewidth=1.2),
            fontsize=9.5, fontweight="bold")

    # Concatenation
    ax.text(0.68, 0.20, "Concatenated Spatio-Temporal Representation\n$[\\mathbf{f}_t, \\vec{v}_t] \\in \\mathbb{R}^{1024}$",
            ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#E0E7FF", edgecolor="#4F46E5", linewidth=1.5),
            fontsize=10, fontweight="bold")

    # Bi-LSTM + Attention
    ax.text(0.92, 0.50, "Bi-LSTM (2-Layer)\n+\nTemporal Attention\n+\nLayerNorm Classifier",
            ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEE2E2", edgecolor="#EF4444", linewidth=1.5),
            fontsize=9.5, fontweight="bold")

    # Arrows
    arrow = dict(arrowstyle="->", lw=2, color="#334155")
    ax.annotate("", xy=(0.24, 0.70), xytext=(0.20, 0.70), arrowprops=arrow)
    ax.annotate("", xy=(0.54, 0.80), xytext=(0.49, 0.73), arrowprops=arrow)
    ax.annotate("", xy=(0.54, 0.60), xytext=(0.49, 0.67), arrowprops=arrow)
    ax.annotate("", xy=(0.68, 0.32), xytext=(0.68, 0.45), arrowprops=arrow)
    ax.annotate("", xy=(0.82, 0.50), xytext=(0.76, 0.30), arrowprops=arrow)
    ax.annotate("", xy=(0.82, 0.55), xytext=(0.76, 0.75), arrowprops=arrow)

    ax.set_title("Strategy 2 Architectural Paradigm: Spatio-Temporal Feature Velocity Modeling", fontweight="bold", pad=15)
    plt.tight_layout()
    out_png = FIGURES_DIR / "fig_motion_differential_concept.png"
    out_pdf = FIGURES_DIR / "fig_motion_differential_concept.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    print(f"Saved: {out_png}")

def plot_clinical_operating_curve():
    """Plots precision, recall, F1, and accuracy across decision thresholds to identify optimal clinical point."""
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    f1_scores  = [0.9449, 0.9516, 0.9672, 0.9672, 0.9587, 0.9500, 0.9500, 0.9500, 0.9500, 0.9500, 0.9322, 0.9322, 0.9322, 0.9322, 0.9043, 0.9043]
    recalls    = [0.9524, 0.9365, 0.9365, 0.9365, 0.9206, 0.9048, 0.9048, 0.9048, 0.9048, 0.9048, 0.8730, 0.8730, 0.8730, 0.8730, 0.8254, 0.8254]
    precisions = [0.9375, 0.9672, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
    accuracies = [0.9457, 0.9535, 0.9690, 0.9690, 0.9612, 0.9535, 0.9535, 0.9535, 0.9535, 0.9535, 0.9380, 0.9380, 0.9380, 0.9380, 0.9147, 0.9147]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(thresholds, [f*100 for f in f1_scores], label="F1-Score", color="#2563EB", lw=2.5, marker="o")
    ax.plot(thresholds, [r*100 for r in recalls], label="Recall (Sensitivity)", color="#DC2626", lw=2.2, linestyle="--", marker="s")
    ax.plot(thresholds, [p*100 for p in precisions], label="Precision (PPV)", color="#059669", lw=2.2, linestyle="-.", marker="^")
    ax.plot(thresholds, [a*100 for a in accuracies], label="Accuracy", color="#7C3AED", lw=2.0, linestyle=":", marker="d")

    # Optimal operating point highlight
    opt_thresh = 0.25
    ax.axvline(opt_thresh, color="#D97706", linestyle="--", lw=1.8, label=r"Optimal Operating Point ($\tau^* = 0.25$)")
    ax.annotate(r"$\tau^* = 0.25$" + "\n" + r"$\mathbf{Acc: 96.9\%}$" + "\n" + r"$\mathbf{F1: 0.967}$" + "\n" + r"$\mathbf{Prec: 100\%}$",
                xy=(opt_thresh, 96.9), xytext=(opt_thresh + 0.05, 87),
                arrowprops=dict(facecolor="#D97706", shrink=0.08, width=1.5, headwidth=6),
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF3C7", edgecolor="#D97706", linewidth=1.2),
                fontsize=9.5, fontweight="bold")

    ax.set_title("Clinical Decision Threshold Tradeoff Curve (Validation Set)", fontweight="bold", pad=15)
    ax.set_xlabel("Decision Threshold $(\\tau)$", fontweight="bold")
    ax.set_ylabel("Metric Score (%)", fontweight="bold")
    ax.set_ylim(80, 103)
    ax.set_xlim(0.08, 0.88)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", framealpha=0.95, facecolor="#F8FAFC", edgecolor="#CBD5E1")

    plt.tight_layout()
    out_png = FIGURES_DIR / "fig_clinical_threshold_curve.png"
    out_pdf = FIGURES_DIR / "fig_clinical_threshold_curve.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    # Real benchmark empirical values
    m1 = {"accuracy": 0.6600, "f1": 0.7302, "precision": 0.7667, "recall": 0.6970, "specificity": 0.5882, "auc": 0.6836}
    m2 = {"accuracy": 0.7500, "f1": 0.8000, "precision": 0.8421, "recall": 0.7619, "specificity": 0.7273, "auc": 0.7554}
    m3 = {"accuracy": 0.9535, "f1": 0.9500, "precision": 1.0000, "recall": 0.9048, "specificity": 1.0000, "auc": 0.9923}

    cm1 = np.array([[10, 7], [10, 23]])
    cm2 = np.array([[16, 6], [10, 32]])
    cm3 = np.array([[66, 0], [6, 57]])

    plot_three_generation_benchmark(m1, m2, m3)
    plot_confusion_matrix_progression(cm1, cm2, cm3)
    plot_motion_differential_concept()
    plot_clinical_operating_curve()
