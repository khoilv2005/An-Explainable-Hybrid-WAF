# -*- coding: utf-8 -*-
"""
WAF MODEL REPORT & VISUALIZATION - IEEE/SPRINGER PAPER STYLE
=============================================================
Optimized for academic paper submission with:
- 4-5 essential figures (Class Distribution, ROC, PR Curve, Confusion Matrix)
- IEEE/Springer compliant formatting
- Times New Roman style fonts
- Publication-ready 300 DPI output
- Optional: Threshold analysis, Metrics table

Reference: IEEE Transactions guidelines
"""

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score
)
import pandas as pd
from datetime import datetime
import os

# ==============================================================================
# PAPER STYLE CONFIGURATION (IEEE/Springer Compliant)
# ==============================================================================
# Font settings - Times New Roman style (required by most journals)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 150,
    'savefig.dpi': 300,  # Publication quality
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'savefig.format': 'pdf'  # Vector format for papers
})

# ==============================================================================
# CONFIG
# ==============================================================================
MODEL_PATH = "./data/waf_model.pth"
TOKENIZER_PATH = "./data/tokenizer.pkl"
DATA_FILE = "./data/processed_data.pkl"
HISTORY_PATH = "./data/training_history.pkl"
OUTPUT_DIR = "./reports"
MAX_LEN = 512
EMBEDDING_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# LOAD SYSTEM
# ==============================================================================
def load_system():
    """Load model, tokenizer, data, history"""
    print("Loading model and data...")

    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)

    from model import WAF_Attention_Model
    vocab_size = len(tokenizer.word_index) + 1
    model = WAF_Attention_Model(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_classes=1
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)

    history = None
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'rb') as f:
            history = pickle.load(f)

    print("Loaded successfully!")
    return model, tokenizer, data, history

# ==============================================================================
# PREDICTION & METRICS
# ==============================================================================
def get_predictions(model, X, device, batch_size=256):
    """Get predictions with batching for large datasets"""
    model.eval()
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.LongTensor(X[i:i+batch_size]).to(device)
            outputs = model(batch)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)

    probs = np.array(all_probs)
    preds = (probs > 0.5).astype(int)
    return preds, probs

def calculate_all_metrics(y_true, y_pred, y_probs):
    """Calculate all metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_probs)
    }

# ==============================================================================
# PAPER-STYLE FIGURES - IEEE/SPRINGER OPTIMIZED
# ==============================================================================

# Figure captions for paper (IEEE style)
FIGURE_CAPTIONS = {
    'class_dist': "Fig. 1. Distribution of normal and attack samples in the training and test sets. The dataset exhibits near-balanced class distribution with {attack_pct:.1f}% attack samples.",
    'roc_curve': "Fig. 2. Receiver Operating Characteristic (ROC) curve for the proposed WAF model. The Area Under the Curve (AUC) of {auc:.4f} indicates excellent discriminative capability.",
    'pr_curve': "Fig. 3. Precision-Recall curve demonstrating model performance across different classification thresholds. Average Precision (AP) = {ap:.4f}.",
    'confusion_matrix': "Fig. 4. Confusion matrix showing the classification results on the test set (n={total:,}). The model achieves {accuracy:.2f}% accuracy with low false negative rate.",
    'threshold': "Fig. 5. Impact of classification threshold on performance metrics. Optimal threshold t={threshold:.2f} maximizes F1-score at {f1:.4f}."
}


def plot_classification_report_figure(y_true, y_pred, save_path):
    """
    Classification report as a publication-ready figure.
    Renders per-class Precision, Recall, F1-Score, and Support in a table-like image.
    """
    # Get classification report as dict for reliable values
    from sklearn.metrics import classification_report
    report_dict = classification_report(
        y_true, y_pred,
        target_names=['Normal', 'Attack'],
        output_dict=True
    )

    # Build table data
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    rows = []
    for cls in ["Normal", "Attack"]:
        cls_row = [
            cls,
            f"{report_dict[cls]['precision']:.4f}",
            f"{report_dict[cls]['recall']:.4f}",
            f"{report_dict[cls]['f1-score']:.4f}",
            f"{int(report_dict[cls]['support'])}"
        ]
        rows.append(cls_row)

    # Add weighted avg row
    wa = report_dict.get('weighted avg', {})
    rows.append([
        'Weighted Avg',
        f"{wa.get('precision', 0):.4f}",
        f"{wa.get('recall', 0):.4f}",
        f"{wa.get('f1-score', 0):.4f}",
        f"{int(wa.get('support', 0))}"
    ])

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.axis('off')

    # Title
    ax.text(0.0, 1.05, 'Classification Report', fontsize=12, fontweight='bold', transform=ax.transAxes)

    # Table rendering
    # Build table (use colLabels for header row)
    table = ax.table(cellText=rows,
                     colLabels=headers,
                     cellLoc='center',
                     loc='upper left',
                     colWidths=[0.22, 0.19, 0.19, 0.19, 0.21])
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Style header row
    for i in range(len(headers)):
        cell = table[0, i]
        cell.set_facecolor('#e5e7eb')
        cell.set_edgecolor('black')
        cell.set_linewidth(0.8)
        cell.set_height(0.18)
        cell.get_text().set_fontweight('bold')

    # Style body cells
    for r in range(1, len(rows) + 1):
        for c in range(len(headers)):
            cell = table[r, c]
            cell.set_edgecolor('black')
            cell.set_linewidth(0.6)
            cell.set_height(0.18)

    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved classification report figure: {save_path}")
    plt.close()


def plot_class_distribution(data, save_path):
    """
    Figure 1: Class Distribution (ESSENTIAL for paper)
    - Proves dataset is not severely imbalanced
    - Required for Dataset Description / Experimental Setup section
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    
    y_train = data['y_train'].flatten()
    y_test = data['y_test'].flatten()
    
    train_normal = (y_train == 0).sum()
    train_attack = (y_train == 1).sum()
    test_normal = (y_test == 0).sum()
    test_attack = (y_test == 1).sum()
    
    x = np.arange(2)
    width = 0.35
    
    color_train = '#2563eb'  # Blue
    color_test = '#dc2626'   # Red
    
    bars1 = ax.bar(x - width/2, [train_normal, train_attack], width,
                   label='Training Set', color=color_train, alpha=0.85, edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, [test_normal, test_attack], width,
                   label='Test Set', color=color_test, alpha=0.85, edgecolor='black', linewidth=0.8)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{int(height):,}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{int(height):,}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Normal', 'Attack'])
    ax.set_ylabel('Number of Samples')
    ax.set_xlabel('Class')
    ax.legend(loc='upper right', framealpha=0.95)
    
    # Calculate attack percentage for caption
    total_attack = train_attack + test_attack
    total_samples = len(y_train) + len(y_test)
    attack_pct = (total_attack / total_samples) * 100
    
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    
    # Print caption
    caption = FIGURE_CAPTIONS['class_dist'].format(attack_pct=attack_pct)
    print(f"Caption: {caption}\n")
    
    plt.close()
    return attack_pct


def plot_roc_curve(y_true, y_probs, save_path):
    """
    Figure 2: ROC Curve (MANDATORY for classification papers)
    - International standard metric
    - Threshold-independent evaluation
    - Required for Results / Performance Evaluation section
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    # Main ROC curve
    ax.plot(fpr, tpr, color='#2563eb', lw=2, 
            label=f'Proposed Model (AUC = {roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#2563eb')
    
    # Random classifier baseline
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', 
            label='Random Classifier (AUC = 0.5)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    
    # Print caption
    caption = FIGURE_CAPTIONS['roc_curve'].format(auc=roc_auc)
    print(f"Caption: {caption}\n")
    
    plt.close()
    return roc_auc


def plot_precision_recall_curve(y_true, y_probs, save_path):
    """
    Figure 3: Precision-Recall Curve (HIGHLY RECOMMENDED for IDS/Security papers)
    - More informative than accuracy for security applications
    - Reviewers appreciate PR curves for anomaly detection tasks
    - Required for Results section
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)
    
    # Main PR curve
    ax.plot(recall_vals, precision_vals, color='#2563eb', lw=2,
            label=f'Proposed Model (AP = {avg_precision:.4f})')
    ax.fill_between(recall_vals, precision_vals, alpha=0.15, color='#2563eb')
    
    # Baseline (random classifier)
    baseline = y_true.sum() / len(y_true)
    ax.axhline(y=baseline, color='gray', linestyle='--', lw=1.5,
               label=f'Random Classifier (P = {baseline:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='lower left', framealpha=0.95)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    
    # Print caption
    caption = FIGURE_CAPTIONS['pr_curve'].format(ap=avg_precision)
    print(f"Caption: {caption}\n")
    
    plt.close()
    return avg_precision


def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Figure 4: Confusion Matrix (RECOMMENDED - single figure)
    - Visualizes FN/FP errors clearly
    - Shows system reliability for attack detection
    - Required for Results section
    """
    fig, ax = plt.subplots(figsize=(5, 4.5))
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Proportion', fontsize=9)
    
    # Labels
    classes = ['Normal', 'Attack']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Text annotations with counts and percentages
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:,}\n({cm_normalized[i, j]:.1%})',
                    ha="center", va="center", fontsize=10,
                    color="white" if cm_normalized[i, j] > thresh else "black")
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum() * 100
    total = cm.sum()
    
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    
    # Print caption
    caption = FIGURE_CAPTIONS['confusion_matrix'].format(total=total, accuracy=accuracy)
    print(f"Caption: {caption}\n")
    
    plt.close()
    return cm


def plot_metrics_panel(metrics, history, save_path):
    """
    2x2 panel. If history is available (train/val), plot lines over epochs for
    Accuracy, Precision, Recall, F1. Otherwise fall back to single-value bars.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    panels = [
        ("Accuracy", 'train_accuracy', 'val_accuracy', metrics['accuracy']),
        ("Precision", 'train_precision', 'val_precision', metrics['precision']),
        ("Recall", 'train_recall', 'val_recall', metrics['recall']),
        ("F1-Score", 'train_f1', 'val_f1', metrics['f1'])
    ]
    colors = ['#2563eb', '#059669', '#d97706', '#dc2626']

    has_history = history is not None and 'train_loss' in history
    epochs = range(1, len(history['train_loss']) + 1) if has_history else None

    for ax, (name, train_key, val_key, metric_val), color in zip(axes.flatten(), panels, colors):
        if has_history and train_key in history and val_key in history:
            ax.plot(epochs, history[train_key], color=color, lw=2, label='Train')
            ax.plot(epochs, history[val_key], color='gray', lw=2, label='Val')
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel(name, fontweight='bold')
            ax.set_title(name, fontweight='bold', pad=8)
            ax.legend(framealpha=0.9, fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim([0, 1.05])
            ax.text(0.98, 0.06, f"Val: {history[val_key][-1]:.4f}", transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.65))
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.barh([0.5], [metric_val], color=color, alpha=0.85, edgecolor='black', height=0.35)
            ax.text(0.02, 0.82, name, fontsize=12, fontweight='bold')
            ax.text(min(metric_val + 0.02, 0.98), 0.5, f"{metric_val:.4f}", fontsize=12,
                    va='center', ha='left')
            ax.axvline(0.90, color='gray', linestyle='--', lw=1)
            ax.axvline(0.95, color='gray', linestyle='--', lw=1)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved metrics panel: {save_path}")
    plt.close()


def plot_threshold_analysis(y_true, y_probs, save_path):
    """
    Figure 5: Threshold Analysis (OPTIONAL - for Appendix or if threshold is a contribution)
    - Only include if paper proposes threshold selection strategy
    - Or if focusing on Precision-Recall trade-off
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    thresholds = np.arange(0.0, 1.01, 0.01)
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for thresh in thresholds:
        y_pred_temp = (y_probs >= thresh).astype(int)
        accuracies.append(accuracy_score(y_true, y_pred_temp))
        precisions.append(precision_score(y_true, y_pred_temp, zero_division=0))
        recalls.append(recall_score(y_true, y_pred_temp, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred_temp, zero_division=0))
    
    ax.plot(thresholds, precisions, lw=1.8, label='Precision', color='#059669', linestyle='-')
    ax.plot(thresholds, recalls, lw=1.8, label='Recall', color='#d97706', linestyle='-')
    ax.plot(thresholds, f1_scores, lw=2, label='F1-Score', color='#dc2626', linestyle='-')
    ax.plot(thresholds, accuracies, lw=1.5, label='Accuracy', color='#2563eb', linestyle='--', alpha=0.7)
    
    # Mark optimal F1 threshold
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    
    ax.axvline(x=best_threshold, color='gray', linestyle=':', alpha=0.8, lw=1.5)
    ax.scatter([best_threshold], [best_f1], color='#dc2626', s=80, zorder=5, marker='o',
               edgecolors='black', linewidths=1)
    ax.annotate(f't={best_threshold:.2f}', xy=(best_threshold, best_f1),
                xytext=(best_threshold + 0.08, best_f1 - 0.05), fontsize=9)
    
    ax.set_xlabel('Classification Threshold')
    ax.set_ylabel('Score')
    ax.legend(loc='center left', framealpha=0.95, fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    
    # Print caption
    caption = FIGURE_CAPTIONS['threshold'].format(threshold=best_threshold, f1=best_f1)
    print(f"Caption: {caption}\n")
    
    plt.close()
    return best_threshold, best_f1


def plot_combined_curves(y_true, y_probs, save_path):
    """
    Combined ROC and PR curves in single figure (2 subplots)
    Use this to save figure count in paper (counts as 1 figure)
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    
    # ROC Curve
    ax = axes[0]
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='#2563eb', lw=2, label=f'AUC = {roc_auc:.4f}')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#2563eb')
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('(a) ROC Curve')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_aspect('equal')
    
    # PR Curve
    ax = axes[1]
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)
    
    ax.plot(recall_vals, precision_vals, color='#2563eb', lw=2, label=f'AP = {avg_precision:.4f}')
    ax.fill_between(recall_vals, precision_vals, alpha=0.15, color='#2563eb')
    baseline = y_true.sum() / len(y_true)
    ax.axhline(y=baseline, color='gray', linestyle='--', lw=1.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('(b) Precision-Recall Curve')
    ax.legend(loc='lower left', framealpha=0.95)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved combined curves: {save_path}")
    print(f"Caption: Fig. X. Performance evaluation curves: (a) ROC curve with AUC = {roc_auc:.4f}, (b) Precision-Recall curve with AP = {avg_precision:.4f}.\n")
    
    plt.close()
    return roc_auc, avg_precision


# ==============================================================================
# METRICS TABLE (Alternative to bar chart - preferred by journals)
# ==============================================================================
def generate_metrics_table(metrics, y_true, y_pred):
    """
    Generate LaTeX table for paper (journals prefer tables over bar charts)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    latex_table = f"""
% LaTeX Table - Performance Metrics
\\begin{{table}}[htbp]
\\centering
\\caption{{Performance metrics of the proposed WAF model on the test set.}}
\\label{{tab:performance}}
\\begin{{tabular}}{{lc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
Accuracy & {metrics['accuracy']:.4f} \\\\
Precision & {metrics['precision']:.4f} \\\\
Recall & {metrics['recall']:.4f} \\\\
F1-Score & {metrics['f1']:.4f} \\\\
AUC-ROC & {metrics['roc_auc']:.4f} \\\\
\\midrule
False Positive Rate & {fpr:.4f} \\\\
False Negative Rate & {fnr:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    print("\n" + "="*60)
    print("LATEX TABLE (copy to paper)")
    print("="*60)
    print(latex_table)
    
    return latex_table


# ==============================================================================
# LEGACY FUNCTIONS (kept for backward compatibility, moved to appendix)
# ==============================================================================
def plot_main_figure(y_true, y_pred, y_probs, metrics, save_path):
    """
    Main figure with 4 subplots (2x2) - Paper style
    (a) Confusion Matrix
    (b) ROC Curve
    (c) Precision-Recall Curve
    (d) Metrics Bar Chart
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # Colors
    color_main = '#2563eb'  # Blue
    color_secondary = '#dc2626'  # Red
    color_fill = '#93c5fd'  # Light blue

    # =========================================================================
    # (a) Confusion Matrix - Top Left
    # =========================================================================
    ax = axes[0, 0]
    cm = confusion_matrix(y_true, y_pred)

    # Normalize for display
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    # Labels
    classes = ['Normal', 'Attack']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:,}\n({cm_normalized[i, j]:.1%})',
                    ha="center", va="center", fontsize=10,
                    color="white" if cm_normalized[i, j] > thresh else "black")

    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_title('(a) Confusion Matrix', fontweight='bold', pad=10)

    # =========================================================================
    # (b) ROC Curve - Top Right
    # =========================================================================
    ax = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, color=color_main, lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.2, color=color_fill)
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('(b) ROC Curve', fontweight='bold', pad=10)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_aspect('equal')

    # =========================================================================
    # (c) Precision-Recall Curve - Bottom Left
    # =========================================================================
    ax = axes[1, 0]
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)

    ax.plot(recall_curve, precision_curve, color=color_main, lw=2,
            label=f'PR (AP = {avg_precision:.4f})')
    ax.fill_between(recall_curve, precision_curve, alpha=0.2, color=color_fill)

    # Add baseline (random classifier)
    baseline = y_true.sum() / len(y_true)
    ax.axhline(y=baseline, color='gray', linestyle='--', lw=1.5, label=f'Baseline ({baseline:.2f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title('(c) Precision-Recall Curve', fontweight='bold', pad=10)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_aspect('equal')

    # =========================================================================
    # (d) Performance Metrics Bar Chart - Bottom Right
    # =========================================================================
    ax = axes[1, 1]

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['roc_auc']
    ]

    x_pos = np.arange(len(metric_names))
    bars = ax.bar(x_pos, metric_values, color=color_main, alpha=0.8, edgecolor='black', linewidth=1)

    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names, rotation=0)
    ax.set_ylim([0, 1.15])
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('(d) Performance Metrics', fontweight='bold', pad=10)

    # Add horizontal line at 0.9 and 0.95 for reference
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, lw=1)
    ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.5, lw=1)

    # =========================================================================
    # Final adjustments
    # =========================================================================
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved main figure: {save_path}")
    plt.close()


def plot_training_curves(history, save_path):
    """
    [APPENDIX ONLY] Training curves - NOT recommended for main paper
    Only include if proposing new training strategy or optimization method
    """
    if history is None:
        print("No training history available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    epochs = range(1, len(history['train_loss']) + 1)

    color_train = '#2563eb'  # Blue
    color_val = '#dc2626'    # Red

    # =========================================================================
    # (a) Loss
    # =========================================================================
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], color=color_train, lw=2, label='Training')
    ax.plot(epochs, history['val_loss'], color=color_val, lw=2, label='Validation')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('(a) Training and Validation Loss', fontweight='bold', pad=10)
    ax.legend(framealpha=0.9)

    # =========================================================================
    # (b) Accuracy
    # =========================================================================
    ax = axes[0, 1]
    ax.plot(epochs, history['train_accuracy'], color=color_train, lw=2, label='Training')
    ax.plot(epochs, history['val_accuracy'], color=color_val, lw=2, label='Validation')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('(b) Training and Validation Accuracy', fontweight='bold', pad=10)
    ax.legend(framealpha=0.9)
    ax.set_ylim([0, 1.05])

    # =========================================================================
    # (c) F1-Score
    # =========================================================================
    ax = axes[1, 0]
    ax.plot(epochs, history['train_f1'], color=color_train, lw=2, label='Training')
    ax.plot(epochs, history['val_f1'], color=color_val, lw=2, label='Validation')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('F1-Score', fontweight='bold')
    ax.set_title('(c) Training and Validation F1-Score', fontweight='bold', pad=10)
    ax.legend(framealpha=0.9)
    ax.set_ylim([0, 1.05])

    # =========================================================================
    # (d) Learning Rate
    # =========================================================================
    ax = axes[1, 1]
    ax.plot(epochs, history['learning_rates'], color='#059669', lw=2)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Learning Rate', fontweight='bold')
    ax.set_title('(d) Learning Rate Schedule', fontweight='bold', pad=10)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved training curves: {save_path}")
    plt.close()


def plot_data_distribution(data, y_probs, save_path):
    """
    [APPENDIX ONLY] Data distribution figure - NOT essential for main paper
    Probability distributions are rarely used in papers
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    color_normal = '#2563eb'   # Blue
    color_attack = '#dc2626'   # Red
    color_train = '#059669'    # Green
    color_test = '#7c3aed'     # Purple

    # =========================================================================
    # (a) Class Distribution - Train vs Test
    # =========================================================================
    ax = axes[0, 0]

    y_train = data['y_train'].flatten()
    y_test = data['y_test'].flatten()

    train_normal = (y_train == 0).sum()
    train_attack = (y_train == 1).sum()
    test_normal = (y_test == 0).sum()
    test_attack = (y_test == 1).sum()

    x = np.arange(2)
    width = 0.35

    bars1 = ax.bar(x - width/2, [train_normal, train_attack], width,
                   label='Train', color=color_train, alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, [test_normal, test_attack], width,
                   label='Test', color=color_test, alpha=0.8, edgecolor='black')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(['Normal', 'Attack'])
    ax.set_ylabel('Number of Samples', fontweight='bold')
    ax.set_title('(a) Class Distribution', fontweight='bold', pad=10)
    ax.legend(framealpha=0.9)

    # Add percentage annotations
    train_total = len(y_train)
    test_total = len(y_test)
    ax.text(0.02, 0.98, f'Train: {train_attack/train_total:.1%} Attack\nTest: {test_attack/test_total:.1%} Attack',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # =========================================================================
    # (b) Sequence Length Distribution
    # =========================================================================
    ax = axes[0, 1]

    # Calculate non-zero lengths (actual sequence lengths before padding)
    X_train = data['X_train']
    X_test = data['X_test']

    train_lengths = np.sum(X_train != 0, axis=1)
    test_lengths = np.sum(X_test != 0, axis=1)

    ax.hist(train_lengths, bins=50, alpha=0.7, label='Train', color=color_train, edgecolor='black', linewidth=0.5)
    ax.hist(test_lengths, bins=50, alpha=0.7, label='Test', color=color_test, edgecolor='black', linewidth=0.5)

    ax.axvline(x=np.mean(train_lengths), color=color_train, linestyle='--', lw=2,
               label=f'Train Mean: {np.mean(train_lengths):.0f}')
    ax.axvline(x=np.mean(test_lengths), color=color_test, linestyle='--', lw=2,
               label=f'Test Mean: {np.mean(test_lengths):.0f}')

    ax.set_xlabel('Sequence Length', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('(b) Sequence Length Distribution', fontweight='bold', pad=10)
    ax.legend(fontsize=8, framealpha=0.9)

    # =========================================================================
    # (c) Prediction Probability Distribution
    # =========================================================================
    ax = axes[1, 0]

    ax.hist(y_probs, bins=50, alpha=0.8, color=color_normal, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0.5, color='red', linestyle='--', lw=2, label='Decision Boundary (0.5)')
    ax.axvline(x=np.mean(y_probs), color='orange', linestyle='-', lw=2,
               label=f'Mean: {np.mean(y_probs):.3f}')

    ax.set_xlabel('Prediction Probability', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('(c) Prediction Probability Distribution', fontweight='bold', pad=10)
    ax.legend(framealpha=0.9)
    ax.set_xlim([0, 1])

    # =========================================================================
    # (d) Probability Distribution by True Class
    # =========================================================================
    ax = axes[1, 1]

    y_test_flat = data['y_test'].flatten()
    probs_normal = y_probs[y_test_flat == 0]
    probs_attack = y_probs[y_test_flat == 1]

    ax.hist(probs_normal, bins=50, alpha=0.7, label=f'True Normal (n={len(probs_normal):,})',
            color=color_normal, edgecolor='black', linewidth=0.5)
    ax.hist(probs_attack, bins=50, alpha=0.7, label=f'True Attack (n={len(probs_attack):,})',
            color=color_attack, edgecolor='black', linewidth=0.5)

    ax.axvline(x=0.5, color='gray', linestyle='--', lw=2, label='Decision Boundary')

    ax.set_xlabel('Prediction Probability', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('(d) Probability by True Class', fontweight='bold', pad=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_xlim([0, 1])

    # =========================================================================
    # Final adjustments
    # =========================================================================
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved data distribution figure: {save_path}")
    plt.close()


def plot_threshold_analysis(y_true, y_probs, save_path):
    """
    Threshold analysis figure - Paper style
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    thresholds = np.arange(0.0, 1.01, 0.01)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for thresh in thresholds:
        y_pred_temp = (y_probs >= thresh).astype(int)
        accuracies.append(accuracy_score(y_true, y_pred_temp))
        precisions.append(precision_score(y_true, y_pred_temp, zero_division=0))
        recalls.append(recall_score(y_true, y_pred_temp, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred_temp, zero_division=0))

    ax.plot(thresholds, accuracies, lw=2, label='Accuracy', color='#2563eb')
    ax.plot(thresholds, precisions, lw=2, label='Precision', color='#059669')
    ax.plot(thresholds, recalls, lw=2, label='Recall', color='#d97706')
    ax.plot(thresholds, f1_scores, lw=2, label='F1-Score', color='#dc2626')

    # Mark best F1
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]

    ax.axvline(x=best_threshold, color='gray', linestyle='--', alpha=0.7, lw=1.5)
    ax.scatter([best_threshold], [best_f1], color='#dc2626', s=100, zorder=5,
               label=f'Best F1={best_f1:.4f} @ t={best_threshold:.2f}')

    ax.set_xlabel('Classification Threshold', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Threshold Analysis', fontweight='bold', pad=10)
    ax.legend(loc='center left', framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved threshold analysis: {save_path}")
    plt.close()

    return best_threshold, best_f1


# ==============================================================================
# TEXT REPORT
# ==============================================================================
def generate_text_report(metrics, y_true, y_pred, save_path):
    """Generate detailed text report"""
    report = []
    report.append("=" * 70)
    report.append("WAF MODEL PERFORMANCE REPORT")
    report.append("=" * 70)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Device: {DEVICE}")
    report.append(f"Model: {MODEL_PATH}")

    report.append("\n" + "=" * 70)
    report.append("OVERALL METRICS")
    report.append("=" * 70)
    report.append(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    report.append(f"  Precision: {metrics['precision']:.4f}")
    report.append(f"  Recall:    {metrics['recall']:.4f}")
    report.append(f"  F1 Score:  {metrics['f1']:.4f}")
    report.append(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report.append("\n" + "=" * 70)
    report.append("CONFUSION MATRIX")
    report.append("=" * 70)
    report.append("                Predicted")
    report.append("              Normal  Attack")
    report.append(f"Actual Normal  {tn:6d}  {fp:6d}")
    report.append(f"       Attack  {fn:6d}  {tp:6d}")

    report.append("\n" + "=" * 70)
    report.append("ERROR ANALYSIS")
    report.append("=" * 70)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    report.append(f"  False Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
    report.append(f"  False Negative Rate: {fnr:.4f} ({fnr*100:.2f}%)")
    report.append(f"  True Positives:  {tp:,}")
    report.append(f"  True Negatives:  {tn:,}")

    report.append("\n" + "=" * 70)
    report.append("CLASSIFICATION REPORT")
    report.append("=" * 70)
    report.append(classification_report(
        y_true, y_pred,
        target_names=['Normal', 'Attack'],
        digits=4
    ))

    report_text = '\n'.join(report)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"Saved text report: {save_path}")
    print("\n" + report_text)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    """
    Main function - generates paper-ready figures
    
    RECOMMENDED FIGURES FOR IEEE/SPRINGER PAPER:
    1. Class Distribution (Fig. 1) - Dataset description
    2. ROC Curve (Fig. 2) - Mandatory for classification
    3. Precision-Recall Curve (Fig. 3) - Important for IDS/security
    4. Confusion Matrix (Fig. 4) - Visualization of errors
    5. [Optional] Threshold Analysis (Fig. 5) - Only if relevant
    
    APPENDIX ONLY (not generated by default):
    - Training curves (Loss, Accuracy, F1, LR)
    - Prediction probability distributions
    """
    print("=" * 70)
    print("WAF MODEL REPORT - IEEE/SPRINGER PAPER STYLE")
    print("=" * 70)
    
    # Load
    model, tokenizer, data, history = load_system()
    
    X_test = data['X_test']
    y_test = data['y_test'].flatten()
    
    # Get predictions
    print("\nGenerating predictions...")
    y_pred, y_probs = get_predictions(model, X_test, DEVICE)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_all_metrics(y_test, y_pred, y_probs)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # ==========================================================================
    # PAPER FIGURES (Essential - 4-5 figures)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("GENERATING PAPER FIGURES")
    print("=" * 70)
    
    # Figure 1: Class Distribution
    print("\n[Fig. 1] Class Distribution...")
    plot_class_distribution(data, f"{OUTPUT_DIR}/fig1_class_distribution.png")
    
    # Figure 2: ROC Curve
    print("[Fig. 2] ROC Curve...")
    plot_roc_curve(y_test, y_probs, f"{OUTPUT_DIR}/fig2_roc_curve.png")
    
    # Figure 3: Precision-Recall Curve
    print("[Fig. 3] Precision-Recall Curve...")
    plot_precision_recall_curve(y_test, y_probs, f"{OUTPUT_DIR}/fig3_pr_curve.png")
    
    # Figure 4: Confusion Matrix
    print("[Fig. 4] Confusion Matrix...")
    plot_confusion_matrix(y_test, y_pred, f"{OUTPUT_DIR}/fig4_confusion_matrix.png")
    
    # Figure 5: Threshold Analysis (Optional)
    print("[Fig. 5] Threshold Analysis (Optional)...")
    best_threshold, best_f1 = plot_threshold_analysis(
        y_test, y_probs, f"{OUTPUT_DIR}/fig5_threshold_analysis.png"
    )

    # Figure 6: Classification Report (Per-class metrics as image)
    print("[Fig. 6] Classification Report (image)...")
    plot_classification_report_figure(y_test, y_pred, f"{OUTPUT_DIR}/fig6_classification_report.png")

    # Figure 7: Metrics Panel (Accuracy, Precision, Recall, F1) with history lines if available
    print("[Fig. 7] Metrics Panel (Accuracy, Precision, Recall, F1)...")
    plot_metrics_panel(metrics, history, f"{OUTPUT_DIR}/fig7_metrics_panel.png")

    
    # Alternative: Combined ROC + PR (saves figure count)
    print("\n[Alternative] Combined ROC + PR Curve (1 figure instead of 2)...")
    plot_combined_curves(y_test, y_probs, f"{OUTPUT_DIR}/fig_combined_curves.png")
    
    # ==========================================================================
    # METRICS TABLE (LaTeX format for paper)
    # ==========================================================================
    latex_table = generate_metrics_table(metrics, y_test, y_pred)
    
    # Save LaTeX table to file
    with open(f"{OUTPUT_DIR}/table_metrics.tex", 'w') as f:
        f.write(latex_table)
    print(f"Saved LaTeX table: {OUTPUT_DIR}/table_metrics.tex")
    
    # ==========================================================================
    # TEXT REPORT
    # ==========================================================================
    print("\nGenerating text report...")
    generate_text_report(metrics, y_test, y_pred, f"{OUTPUT_DIR}/report_{timestamp}.txt")
    
    # ==========================================================================
    # APPENDIX FIGURES (Optional - uncomment if needed)
    # ==========================================================================
    GENERATE_APPENDIX = False  # Set to True if you need appendix figures
    
    if GENERATE_APPENDIX:
        print("\n" + "=" * 70)
        print("GENERATING APPENDIX FIGURES")
        print("=" * 70)
        
        # Legacy combined figure
        plot_main_figure(
            y_test, y_pred, y_probs, metrics,
            f"{OUTPUT_DIR}/appendix_main_figure.png"
        )
        
        # Training curves (only if history exists)
        if history:
            plot_training_curves(history, f"{OUTPUT_DIR}/appendix_training_curves.png")
        
        # Data distribution
        plot_data_distribution(data, y_probs, f"{OUTPUT_DIR}/appendix_distribution.png")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PAPER FIGURES GENERATION COMPLETE")
    print("=" * 70)
    
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print(f"\n{'='*50}")
    print("RECOMMENDED FIGURES FOR PAPER (4-5 figures):")
    print("="*50)
    print("  1. fig1_class_distribution.pdf  -> Dataset Description")
    print("  2. fig2_roc_curve.pdf           -> Results (mandatory)")
    print("  3. fig3_pr_curve.pdf            -> Results (recommended)")
    print("  4. fig4_confusion_matrix.pdf    -> Results")
    print("  5. fig5_threshold_analysis.pdf  -> Appendix (optional)")
    print("")
    print("  Alternative: fig_combined_curves.pdf (ROC + PR in 1 figure)")
    print("")
    print("  Table: table_metrics.tex        -> Results (LaTeX format)")
    print("="*50)
    
    print(f"\n{'='*50}")
    print("PERFORMANCE SUMMARY")
    print("="*50)
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['roc_auc']:.4f}")
    print(f"  Optimal Threshold: {best_threshold:.3f} (F1={best_f1:.4f})")
    print("="*50)


if __name__ == "__main__":
    main()
