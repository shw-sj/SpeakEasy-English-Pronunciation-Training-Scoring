"""
评价指标模块
============
包含：
  1. 分类指标：各类别 Precision / Recall / F1-score / Accuracy
  2. 评分算法评测：Pearson r / MAE / RMSE
  3. 混淆矩阵可视化（热力图）
  4. 训练曲线可视化
  5. 打分策略权重对比可视化
  6. 模型对比雷达图 / 柱状图

成员B：评价指标与可视化模块
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix,
)
from scipy.stats import pearsonr


# ====================================================================
# 第一部分：分类评价指标
# ====================================================================
class ClassificationMetrics:
    """多分类评价指标（字母26类 / 单词N类）"""

    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]

    def compute_all(self, y_true, y_pred):
        """
        :param y_true: 真实标签(整数索引) [n_samples]
        :param y_pred: 预测标签(整数索引) [n_samples]
        :return: dict
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        acc = accuracy_score(y_true, y_pred)
        p = precision_score(y_true, y_pred, average=None, zero_division=0)
        r = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

        macro_p = precision_score(y_true, y_pred, average='macro', zero_division=0)
        macro_r = recall_score(y_true, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        support = np.bincount(y_true, minlength=self.num_classes)

        return {
            "accuracy": float(acc),
            "macro_avg": {"precision": float(macro_p), "recall": float(macro_r),
                          "f1": float(macro_f1)},
            "micro_avg": {"f1": float(micro_f1)},
            "per_class": {
                self.class_names[i]: {
                    "precision": float(p[i]) if i < len(p) else 0.0,
                    "recall": float(r[i]) if i < len(r) else 0.0,
                    "f1": float(f1[i]) if i < len(f1) else 0.0,
                    "support": int(support[i]) if i < len(support) else 0,
                }
                for i in range(self.num_classes)
            },
            "confusion_matrix": cm,
        }

    def print_report(self, y_true, y_pred):
        """打印分类报告"""
        r = self.compute_all(y_true, y_pred)
        print("=" * 70)
        print("Classification Report")
        print("=" * 70)
        print(f"Accuracy: {r['accuracy']:.4f}")
        print(f"Macro F1: {r['macro_avg']['f1']:.4f}")
        print(f"Micro F1: {r['micro_avg']['f1']:.4f}")
        print("-" * 70)
        print(f"{'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
        print("-" * 70)
        for name, m in r["per_class"].items():
            print(f"{name:<8} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                  f"{m['f1']:>10.4f} {m['support']:>8d}")
        print("-" * 70)
        return r


# ====================================================================
# 第二部分：发音评分评测指标
# ====================================================================
class ScoringMetrics:
    """发音评分算法评测 —— 机器评分 vs 人工专家评分"""

    @staticmethod
    def evaluate(machine_scores, human_scores):
        """
        :return: {pearson_r, pearson_p, mae, rmse, r_squared, bias, n}
        """
        m = np.array(machine_scores).flatten()
        h = np.array(human_scores).flatten()
        pr, pp = pearsonr(h, m)
        mae = np.mean(np.abs(m - h))
        rmse = np.sqrt(np.mean((m - h) ** 2))
        ss_res = np.sum((h - m) ** 2)
        ss_tot = np.sum((h - np.mean(h)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        return {"pearson_r": float(pr), "pearson_p": float(pp),
                "mae": float(mae), "rmse": float(rmse),
                "r_squared": float(r2), "bias": float(np.mean(m - h)),
                "n": len(m)}

    @staticmethod
    def print_report(machine_scores, human_scores):
        r = ScoringMetrics.evaluate(machine_scores, human_scores)
        print("=" * 60)
        print("Scoring Evaluation Report")
        print("=" * 60)
        print(f"Samples: {r['n']}")
        print(f"Pearson r: {r['pearson_r']:.4f} (p={r['pearson_p']:.4f})")
        print(f"MAE: {r['mae']:.4f}")
        print(f"RMSE: {r['rmse']:.4f}")
        print(f"R^2: {r['r_squared']:.4f}")
        print(f"Bias: {r['bias']:.4f}")
        print("=" * 60)
        return r


# ====================================================================
# 第三部分：可视化函数
# ====================================================================

def plot_confusion_matrix(cm, class_names=None, figsize=(12, 10),
                          title="Confusion Matrix", save_path=None,
                          normalize=False, cmap='Blues'):
    """混淆矩阵热力图"""
    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    if normalize:
        cm_disp = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fmt = '.2f'
    else:
        cm_disp = cm
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_disp, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax)

    thresh = cm_disp.max() / 2
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                color = 'white' if cm_disp[i, j] > thresh else 'black'
                text = f'{cm_disp[i, j]:.2f}' if normalize else str(cm[i, j])
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=6 if n > 20 else 8, color=color)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90,
                       fontsize=7 if n > 20 else 9)
    ax.set_yticklabels(class_names, fontsize=7 if n > 20 else 9)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Confusion matrix saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_training_curves(train_losses, val_losses, val_accuracies=None,
                         save_path=None, figsize=(14, 5)):
    """训练Loss/Accuracy曲线"""
    epochs = range(1, len(train_losses) + 1)
    n_plots = 2 if val_accuracies is not None else 1

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    axes[0].plot(epochs, train_losses, 'b-', lw=1.5, label='Train Loss')
    axes[0].plot(epochs, val_losses, 'r-', lw=1.5, label='Val Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    if val_accuracies is not None:
        axes[1].plot(epochs, val_accuracies, 'g-', lw=1.5, label='Val Accuracy')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Training curves saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_per_class_f1(per_class_metrics, save_path=None, figsize=(14, 6)):
    """各类别F1-score柱状图"""
    if isinstance(per_class_metrics, dict):
        names = list(per_class_metrics.keys())
        f1_vals = [per_class_metrics[n]["f1"] for n in names]
    else:
        names = [str(i) for i in range(len(per_class_metrics))]
        f1_vals = list(per_class_metrics)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ['#2ecc71' if f >= 0.9 else '#3498db' if f >= 0.7
              else '#f39c12' if f >= 0.5 else '#e74c3c' for f in f1_vals]
    x = range(len(names))
    bars = ax.bar(x, f1_vals, color=colors, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90 if len(names) > 10 else 0)
    ax.set_ylabel('F1-Score'); ax.set_title('Per-Class F1-Score')
    ax.set_ylim(0, 1.05)
    avg = np.mean(f1_vals)
    ax.axhline(y=avg, color='red', linestyle='--', lw=1,
               label=f'Macro Avg={avg:.3f}')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_scoring_comparison(machine_scores, human_scores, save_path=None):
    """机器评分 vs 人工评分散点图 + 回归线"""
    m = np.array(machine_scores).flatten()
    h = np.array(human_scores).flatten()
    r, _ = pearsonr(h, m)
    mae = np.mean(np.abs(m - h))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(h, m, alpha=0.6, c='steelblue', edgecolors='white', s=60)

    lo = min(h.min(), m.min()) - 5
    hi = max(h.max(), m.max()) + 5
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Ideal (y=x)')

    z = np.polyfit(h, m, 1)
    p = np.poly1d(z)
    ax.plot([lo, hi], p([lo, hi]), 'g-', lw=1.5,
            label=f'Fit (y={z[0]:.2f}x+{z[1]:.0f})')

    ax.set_xlabel('Human Expert Score'); ax.set_ylabel('Machine Score')
    ax.set_title(f'Machine vs Human Scores\nPearson r={r:.4f}, MAE={mae:.2f}')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_weight_comparison(weight_configs, pearson_values, labels=None,
                           save_path=None, figsize=(12, 5)):
    """不同打分策略权重对比柱状图"""
    n = len(weight_configs)
    if labels is None:
        labels = [f'Config {i+1}' for i in range(n)]

    x = np.arange(n); width = 0.22
    conf_w = [w[0] for w in weight_configs]
    dtw_w = [w[1] for w in weight_configs]
    aco_w = [w[2] for w in weight_configs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.bar(x - width, conf_w, width, label='Confidence', color='#3498db')
    ax1.bar(x, dtw_w, width, label='DTW', color='#e74c3c')
    ax1.bar(x + width, aco_w, width, label='Acoustic', color='#2ecc71')
    ax1.set_ylabel('Weight'); ax1.set_title('Scoring Weight Distribution')
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45)
    ax1.set_ylim(0, 1); ax1.legend(); ax1.grid(True, alpha=0.3, axis='y')

    best_idx = np.argmax(pearson_values)
    colors = ['#2ecc71' if i == best_idx else '#3498db'
              for i in range(n)]
    bars = ax2.bar(x, pearson_values, color=colors, edgecolor='white')
    ax2.set_ylabel('Pearson r'); ax2.set_title('Scoring Accuracy by Config')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, pearson_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_model_comparison_bar(models_metrics, save_path=None, figsize=(14, 6)):
    """多模型对比柱状图（Accuracy + Params + Infer Time）"""
    names = list(models_metrics.keys())
    x = np.arange(len(names))

    acc = [models_metrics[m].get('accuracy', 0) for m in names]
    params = [models_metrics[m].get('params', 0) for m in names]
    infer = [models_metrics[m].get('infer_time_ms', 0) for m in names]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    colors_acc = ['#2ecc71' if a == max(acc) else '#3498db' for a in acc]
    axes[0].bar(x, acc, color=colors_acc, edgecolor='white')
    axes[0].set_title('Accuracy'); axes[0].set_ylim(0, 1)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=45)
    for i, v in enumerate(acc):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=9)

    axes[1].bar(x, params, color='#e67e22', edgecolor='white')
    axes[1].set_title('Parameters')
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=45)
    for i, v in enumerate(params):
        axes[1].text(i, v + max(params)*0.02, f'{v:,}', ha='center', fontsize=9)

    axes[2].bar(x, infer, color='#9b59b6', edgecolor='white')
    axes[2].set_title('Inference Time (ms)')
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=45)
    for i, v in enumerate(infer):
        axes[2].text(i, v + max(infer)*0.02, f'{v:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_model_radar(models_metrics, metric_names=None,
                     save_path=None, figsize=(10, 10)):
    """多模型对比雷达图"""
    if metric_names is None:
        metric_names = set()
        for m in models_metrics.values():
            metric_names.update(m.keys())
        metric_names = sorted(metric_names)

    n = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models_metrics)))

    for (name, metrics), color in zip(models_metrics.items(), colors):
        values = [metrics.get(m, 0) for m in metric_names]
        max_vals = {mn: max(models_metrics[mn2].get(mn, 0) for mn2 in models_metrics) + 1e-10
                    for mn in metric_names}
        vals_norm = [min(v / max_vals[mn], 1.0) for v, mn in zip(values, metric_names)]
        vals_norm += vals_norm[:1]
        ax.fill(angles, vals_norm, alpha=0.15, color=color)
        ax.plot(angles, vals_norm, 'o-', lw=2, label=name, color=color)

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_title('Model Comparison Radar', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_learning_progress(dates, scores, save_path=None, figsize=(10, 5)):
    """学习进度追踪图（日期 vs 平均分）"""
    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(dates))
    ax.plot(x, scores, 'b-o', lw=2, markersize=8, markerfacecolor='white')

    if len(scores) > 1:
        z = np.polyfit(x, scores, 1)
        ax.plot(x, np.poly1d(z)(x), 'r--', lw=1.5, alpha=0.7, label='Trend')

    ax.set_xticks(x); ax.set_xticklabels(dates, rotation=45)
    ax.set_xlabel('Date'); ax.set_ylabel('Average Score')
    ax.set_title('Learning Progress Tracking')
    ax.set_ylim(0, 105); ax.legend(); ax.grid(True, alpha=0.3)
    for i, s in enumerate(scores):
        ax.annotate(f'{s:.1f}', (i, s), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


# ====================================================================
# 第四部分：快速测试
# ====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Evaluation Metrics Module - Test")
    print("=" * 60)
    np.random.seed(42)

    n_classes = 26
    n_samples = 500
    class_names = [chr(65 + i) for i in range(n_classes)]

    # 模拟分类结果
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    flip = np.random.choice(n_samples, int(n_samples * 0.15), replace=False)
    y_pred[flip] = np.random.randint(0, n_classes, len(flip))

    # 测试1：分类指标
    print("\n[Test 1] Classification Metrics")
    clf = ClassificationMetrics(n_classes, class_names)
    r = clf.compute_all(y_true, y_pred)
    print(f"Accuracy: {r['accuracy']:.4f}, Macro F1: {r['macro_avg']['f1']:.4f}")

    # 测试2：混淆矩阵
    print("\n[Test 2] Confusion Matrix Heatmap")
    plot_confusion_matrix(r['confusion_matrix'], class_names,
                          title='Letter Pronunciation - Confusion Matrix',
                          save_path='confusion_matrix_test.png')

    # 测试3：训练曲线
    print("\n[Test 3] Training Curves")
    tl = np.exp(-np.linspace(0, 3, 100))*0.5 + np.random.randn(100)*0.02 + 0.1
    vl = np.exp(-np.linspace(0, 2.5, 100))*0.5 + np.random.randn(100)*0.03 + 0.15
    va = 1 - np.exp(-np.linspace(0, 3, 100))*0.8 - np.random.randn(100)*0.02
    plot_training_curves(tl, vl, va, save_path='training_curves_test.png')

    # 测试4：评分评测
    print("\n[Test 4] Scoring Evaluation")
    hs = np.random.uniform(40, 95, n_samples)
    ms = np.clip(hs + np.random.randn(n_samples)*8, 0, 100)
    ScoringMetrics.print_report(ms, hs)

    # 测试5：评分对比图
    print("\n[Test 5] Scoring Comparison Scatter")
    plot_scoring_comparison(ms, hs, save_path='scoring_comparison_test.png')

    # 测试6：模型对比
    print("\n[Test 6] Model Comparison Bar Chart")
    model_data = {
        "BP": {"accuracy": 0.823, "params": 285000, "infer_time_ms": 0.8},
        "CNN": {"accuracy": 0.891, "params": 145000, "infer_time_ms": 2.1},
    }
    plot_model_comparison_bar(model_data, save_path='model_comparison_test.png')

    print("\n[OK] All tests completed!")
