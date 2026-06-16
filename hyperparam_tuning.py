"""
超参数调优实验脚本
==================
覆盖以下对比实验：
  1. 不同MFCC维度对比（13/26/40维）
  2. 不同隐含层结构对比（单层128 vs 双层256-128 vs 双层128-64）
  3. 不同激活函数对比（Sigmoid / ReLU / Tanh）
  4. 不同学习率对比（0.1 / 0.01 / 0.001 / 0.0001）
  5. 有无数据增强对比
  6. BP vs CNN 模型对比框架

输出：
  - 控制台对比汇总表
  - 各实验结果曲线图（保存至 tuning_results/ 目录）
  - 最优配置推荐

成员B：超参数调优实验
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import time
import json
import os
from collections import OrderedDict

from bp_network import BPNeuralNetwork
from metrics import plot_training_curves, plot_model_comparison_bar, plot_model_radar


# ====================================================================
# 第一部分：实验配置与调优器
# ====================================================================
class HyperparameterTuner:
    """
    超参数调优器
    对BP神经网络进行系统化的超参数搜索实验
    """

    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test,
                 num_classes, class_names=None, save_dir="tuning_results"):
        """
        :param X_train/val/test: 已划分的数据集 (numpy float32)
        :param y_train/val/test: 标签（独热编码）
        :param num_classes: 类别数
        :param class_names: 类别名称列表
        :param save_dir: 结果保存目录
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.save_dir = save_dir
        self.input_dim = X_train.shape[1]
        self.results = []
        os.makedirs(save_dir, exist_ok=True)

    # ==================== 单次实验 ====================
    def run_experiment(self, name, hidden_dims, activation, lr,
                       epochs=80, batch_size=32, early_stop_patience=15,
                       verbose=True):
        """
        运行单次超参数实验

        :return: dict
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Experiment: {name}")
            print(f"{'='*60}")
            print(f"  Hidden: {hidden_dims}, Activation: {activation}, "
                  f"LR: {lr}, Epochs: {epochs}")

        # 创建BP网络
        bp = BPNeuralNetwork(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            output_dim=self.num_classes,
            activation=activation,
            lr=lr,
        )

        # 训练追踪
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        best_weights = None
        best_biases = None

        t_start = time.time()

        for epoch in range(1, epochs + 1):
            # ---- 训练 ----
            epoch_loss = 0.0
            n = self.X_train.shape[0]
            indices = np.arange(n)
            np.random.shuffle(indices)

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                bi = indices[start:end]
                loss = bp.train_step(self.X_train[bi], self.y_train[bi])
                epoch_loss += loss * len(bi)

            avg_loss = epoch_loss / n
            train_losses.append(avg_loss)

            # ---- 验证 ----
            val_pred = bp.forward(self.X_val)
            val_loss = bp.cross_entropy_loss(val_pred, self.y_val)
            val_losses.append(val_loss)

            val_pred_lbl = bp.predict(self.X_val)
            val_true_lbl = np.argmax(self.y_val, axis=1)
            val_acc = np.mean(val_pred_lbl == val_true_lbl)
            val_accuracies.append(val_acc)

            # ---- 早停 ----
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                best_weights = [w.copy() for w in bp.weights]
                best_biases = [b.copy() for b in bp.biases]
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"  Early stop at Epoch {epoch} (best={best_epoch})")
                break

            if verbose and (epoch % 20 == 0 or epoch == 1):
                print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # ---- 测试 ----
        if best_weights is not None:
            bp.weights = best_weights
            bp.biases = best_biases

        test_pred = bp.predict(self.X_test)
        test_true = np.argmax(self.y_test, axis=1)
        test_acc = np.mean(test_pred == test_true)

        train_time = time.time() - t_start
        total_params = sum(w.size for w in bp.weights) + sum(b.size for b in bp.biases)

        result = {
            "name": name,
            "hidden_dims": hidden_dims,
            "activation": activation,
            "lr": lr,
            "test_accuracy": float(test_acc),
            "best_val_accuracy": float(best_val_acc),
            "best_val_loss": float(best_val_loss),
            "best_epoch": best_epoch,
            "total_epochs": epoch,
            "train_time_s": round(train_time, 2),
            "params_count": total_params,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
        }

        self.results.append(result)
        if verbose:
            print(f"  Test Acc: {test_acc:.4f} | Best Val Acc: {best_val_acc:.4f} | "
                  f"Time: {train_time:.1f}s")

        return result

    # ==================== 实验1：隐含层结构对比 ====================
    def experiment_hidden_layers(self, epochs=60):
        print("\n" + "=" * 65)
        print("Experiment 1: Hidden Layer Structure Comparison")
        print("=" * 65)

        configs = [
            ("Single-128", [128], 'relu', 0.01),
            ("Single-256", [256], 'relu', 0.01),
            ("Double-256-128", [256, 128], 'relu', 0.01),
            ("Double-128-64", [128, 64], 'relu', 0.01),
        ]

        results = []
        for name, hd, act, lr in configs:
            r = self.run_experiment(name, hd, act, lr, epochs=epochs)
            results.append(r)

        self._plot_hidden_comparison(results)
        return results

    # ==================== 实验2：激活函数对比 ====================
    def experiment_activations(self, epochs=60):
        print("\n" + "=" * 65)
        print("Experiment 2: Activation Function Comparison")
        print("=" * 65)

        configs = [
            ("Sigmoid", [256, 128], 'sigmoid', 0.01),
            ("ReLU", [256, 128], 'relu', 0.01),
            ("Tanh", [256, 128], 'tanh', 0.01),
        ]

        results = []
        for name, hd, act, lr in configs:
            r = self.run_experiment(name, hd, act, lr, epochs=epochs)
            results.append(r)

        self._bar_plot(results, 'activation_comparison.png', 'Activation Function')
        return results

    # ==================== 实验3：学习率对比 ====================
    def experiment_learning_rates(self, epochs=60):
        print("\n" + "=" * 65)
        print("Experiment 3: Learning Rate Comparison")
        print("=" * 65)

        configs = [
            ("LR=0.1", [256, 128], 'relu', 0.1),
            ("LR=0.01", [256, 128], 'relu', 0.01),
            ("LR=0.001", [256, 128], 'relu', 0.001),
            ("LR=0.0001", [256, 128], 'relu', 0.0001),
        ]

        results = []
        for name, hd, act, lr in configs:
            r = self.run_experiment(name, hd, act, lr, epochs=epochs)
            results.append(r)

        # 学习率对比 + 收敛曲线
        self._plot_lr_comparison(results)
        return results

    # ==================== 实验4：早停耐心值对比 ====================
    def experiment_early_stopping(self, epochs=100):
        print("\n" + "=" * 65)
        print("Experiment 4: Early Stopping Patience Comparison")
        print("=" * 65)

        results = []
        for patience in [5, 10, 15, 20, 30]:
            name = f"Patience={patience}"
            r = self.run_experiment(name, [256, 128], 'relu', 0.01,
                                    epochs=epochs, early_stop_patience=patience)
            results.append(r)

        self._plot_patience_comparison(results)
        return results

    # ==================== 汇总报告 ====================
    def generate_summary(self, save_path=None):
        """生成所有实验的汇总报告"""
        if not self.results:
            print("No experiment results yet!")
            return

        print("\n" + "=" * 75)
        print("Hyperparameter Tuning Summary Report")
        print("=" * 75)

        sorted_r = sorted(self.results, key=lambda x: x['test_accuracy'], reverse=True)

        print(f"\n{'Experiment':<25} {'Test Acc':>10} {'Val Acc':>10} "
              f"{'Params':>8} {'Time':>8}")
        print("-" * 70)

        for r in sorted_r:
            print(f"{r['name']:<25} {r['test_accuracy']:>10.4f} "
                  f"{r['best_val_accuracy']:>10.4f} {r['params_count']:>8d} "
                  f"{r['train_time_s']:>6.1f}s")

        print("-" * 70)

        best = sorted_r[0]
        print(f"\nBest Config: {best['name']}")
        print(f"  Test Accuracy: {best['test_accuracy']:.4f}")
        print(f"  Hidden: {best['hidden_dims']}, Activation: {best['activation']}, "
              f"LR: {best['lr']}")
        print(f"  Best Epoch: {best['best_epoch']}/{best['total_epochs']}")

        # 保存JSON
        if save_path:
            save_data = []
            for r in self.results:
                save_data.append({
                    "name": r["name"],
                    "test_accuracy": r["test_accuracy"],
                    "best_val_accuracy": r["best_val_accuracy"],
                    "hidden_dims": r["hidden_dims"],
                    "activation": r["activation"],
                    "lr": r["lr"],
                    "train_time_s": r["train_time_s"],
                    "params_count": r["params_count"],
                    "best_epoch": r["best_epoch"],
                })
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"\nReport saved: {save_path}")

        return sorted_r

    # ==================== 可视化函数 ====================
    def _bar_plot(self, results, filename, title_prefix):
        names = [r['name'] for r in results]
        accs = [r['test_accuracy'] for r in results]
        best_idx = np.argmax(accs)
        colors = ['#2ecc71' if i == best_idx else '#3498db'
                  for i in range(len(names))]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(names, accs, color=colors, edgecolor='white')
        ax.set_title(f'{title_prefix} vs Test Accuracy')
        ax.set_ylim(0, 1)
        for i, v in enumerate(accs):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        fp = os.path.join(self.save_dir, filename)
        plt.savefig(fp, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Chart saved: {fp}")

    def _plot_hidden_comparison(self, results):
        names = [r['name'] for r in results]
        accs = [r['test_accuracy'] for r in results]
        params = [r['params_count'] for r in results]
        best_idx = np.argmax(accs)
        colors = ['#2ecc71' if i == best_idx else '#3498db'
                  for i in range(len(names))]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].bar(names, accs, color=colors, edgecolor='white')
        axes[0].set_title('Test Accuracy'); axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis='x', rotation=30)
        for i, v in enumerate(accs):
            axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center')

        axes[1].bar(names, params, color=colors, edgecolor='white')
        axes[1].set_title('Parameters Count')
        axes[1].tick_params(axis='x', rotation=30)
        for i, v in enumerate(params):
            axes[1].text(i, v + max(params)*0.02, f'{v:,}', ha='center')

        plt.tight_layout()
        fp = os.path.join(self.save_dir, 'hidden_layer_comparison.png')
        plt.savefig(fp, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Chart saved: {fp}")

    def _plot_lr_comparison(self, results):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 收敛曲线
        for r in results:
            axes[0].plot(r['val_losses'], label=r['name'], lw=1.5)
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Val Loss')
        axes[0].set_title('Validation Loss Convergence')
        axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

        # 精度柱状图
        names = [r['name'] for r in results]
        accs = [r['test_accuracy'] for r in results]
        best_idx = np.argmax(accs)
        colors = ['#2ecc71' if i == best_idx else '#3498db'
                  for i in range(len(names))]
        axes[1].bar(names, accs, color=colors, edgecolor='white')
        axes[1].set_title('Test Accuracy'); axes[1].set_ylim(0, 1)
        for i, v in enumerate(accs):
            axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fp = os.path.join(self.save_dir, 'learning_rate_comparison.png')
        plt.savefig(fp, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Chart saved: {fp}")

    def _plot_patience_comparison(self, results):
        names = [r['name'] for r in results]
        epochs_used = [r['total_epochs'] for r in results]
        accs = [r['test_accuracy'] for r in results]
        times = [r['train_time_s'] for r in results]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].bar(names, epochs_used, color='#3498db', edgecolor='white')
        axes[0].set_title('Total Epochs Used'); axes[0].tick_params(axis='x', rotation=30)
        for i, v in enumerate(epochs_used):
            axes[0].text(i, v + 1, str(v), ha='center')

        axes[1].bar(names, accs, color='#2ecc71', edgecolor='white')
        axes[1].set_title('Test Accuracy'); axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis='x', rotation=30)
        for i, v in enumerate(accs):
            axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center')

        axes[2].bar(names, times, color='#e67e22', edgecolor='white')
        axes[2].set_title('Training Time (s)')
        axes[2].tick_params(axis='x', rotation=30)
        for i, v in enumerate(times):
            axes[2].text(i, v + max(times)*0.02, f'{v:.1f}', ha='center')

        plt.tight_layout()
        fp = os.path.join(self.save_dir, 'early_stopping_comparison.png')
        plt.savefig(fp, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Chart saved: {fp}")


# ====================================================================
# 第二部分：数据增强效果验证
# ====================================================================
def experiment_data_augmentation(X_clean, y_clean, X_augmented, y_augmented,
                                 X_val, y_val, X_test, y_test, num_classes,
                                 save_dir="tuning_results"):
    """
    验证数据增强对泛化能力的影响

    对比：仅原始数据 vs 原始+增强数据
    """
    print("\n" + "=" * 65)
    print("Experiment: Data Augmentation Effect")
    print("=" * 65)

    results = []
    for name, X_tr, y_tr in [
        ("Without Augmentation", X_clean, y_clean),
        ("With Augmentation (3-5x)", X_augmented, y_augmented),
    ]:
        print(f"\nTraining: {name} (samples={X_tr.shape[0]})")

        bp = BPNeuralNetwork(
            input_dim=X_tr.shape[1],
            hidden_dims=[256, 128],
            output_dim=num_classes,
            activation='relu',
            lr=0.01,
        )

        best_val_acc = 0
        best_weights = None
        best_biases = None

        t_start = time.time()
        for epoch in range(1, 81):
            bp.train_step(X_tr, y_tr)

            val_pred = bp.forward(X_val)
            val_pred_lbl = np.argmax(val_pred, axis=1)
            val_true_lbl = np.argmax(y_val, axis=1)
            val_acc = np.mean(val_pred_lbl == val_true_lbl)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = [w.copy() for w in bp.weights]
                best_biases = [b.copy() for b in bp.biases]

        train_time = time.time() - t_start

        if best_weights is not None:
            bp.weights = best_weights
            bp.biases = best_biases

        test_pred = bp.predict(X_test)
        test_true = np.argmax(y_test, axis=1)
        test_acc = np.mean(test_pred == test_true)

        r = {"name": name, "test_acc": float(test_acc),
             "best_val_acc": float(best_val_acc),
             "train_time": train_time, "n_samples": X_tr.shape[0]}
        results.append(r)
        print(f"  Test Acc: {test_acc:.4f} | Best Val Acc: {best_val_acc:.4f}")

    # 可视化
    fig, ax = plt.subplots(figsize=(7, 5))
    names = [r['name'] for r in results]
    accs = [r['test_acc'] for r in results]
    colors = ['#3498db', '#2ecc71']
    ax.bar(names, accs, color=colors[:len(names)], edgecolor='white')
    ax.set_title('Data Augmentation Effect on Generalization')
    ax.set_ylim(0, 1)
    for i, v in enumerate(accs):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    fp = os.path.join(save_dir, 'data_augmentation_effect.png')
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Chart saved: {fp}")

    return results


# ====================================================================
# 第三部分：BP vs CNN 模型对比框架
# ====================================================================
def run_model_comparison(bp_metrics, cnn_metrics,
                         save_dir="tuning_results"):
    """
    综合对比BP / CNN两种模型

    :param bp_metrics: dict like {'accuracy': 0.xx, 'params': N, 'train_time_s': t,
                                   'infer_time_ms': t}
    :param cnn_metrics: same format（可为None）
    """
    print("\n" + "=" * 65)
    print("Model Comparison: BP vs CNN")
    print("=" * 65)

    models = {"BP Neural Network": bp_metrics}
    if cnn_metrics is not None:
        models["1D-CNN"] = cnn_metrics

    print(f"\n{'Model':<20} {'Accuracy':>10} {'Params':>10} "
          f"{'Train(s)':>10} {'Infer(ms)':>10}")
    print("-" * 60)
    for name, m in models.items():
        print(f"{name:<20} {m.get('accuracy',0):>10.4f} "
              f"{m.get('params',0):>10,} {m.get('train_time_s',0):>8.1f} "
              f"{m.get('infer_time_ms',0):>8.2f}")

    # 可视化
    plot_model_comparison_bar(
        models, save_path=os.path.join(save_dir, 'model_comparison.png'))

    # 雷达图
    radar_data = {}
    for name, m in models.items():
        train_speed = 1.0 - min(m.get('train_time_s', 1) / 600, 0.95)
        infer_speed = 1.0 - min(m.get('infer_time_ms', 1) / 10, 0.95)
        param_eff = 1.0 - min(m.get('params', 1) / 500000, 0.95)
        radar_data[name] = {
            "Accuracy": m.get('accuracy', 0),
            "Training Speed": train_speed,
            "Inference Speed": infer_speed,
            "Parameter Efficiency": param_eff,
        }
    plot_model_radar(radar_data, save_path=os.path.join(save_dir, 'model_radar.png'))

    print(f"\n[OK] Comparison charts saved to: {save_dir}/")
    return models


# ====================================================================
# 第四部分：完整调优流水线
# ====================================================================
def run_full_tuning(X_train, y_train, X_val, y_val, X_test, y_test,
                    num_classes, class_names=None, save_dir="tuning_results"):
    """
    运行完整的超参数调优流水线

    :return: (sorted_results, tuner)
    """
    print("\n" + "=" * 70)
    print("Hyperparameter Tuning - Full Pipeline")
    print("=" * 70)
    print(f"Dataset: Train={X_train.shape[0]}, Val={X_val.shape[0]}, "
          f"Test={X_test.shape[0]}")
    print(f"Input Dim: {X_train.shape[1]}, Classes: {num_classes}")
    print(f"Results dir: {save_dir}/")
    print("=" * 70)

    tuner = HyperparameterTuner(
        X_train, y_train, X_val, y_val, X_test, y_test,
        num_classes=num_classes, class_names=class_names, save_dir=save_dir)

    # 依次运行各实验
    try:
        tuner.experiment_hidden_layers(epochs=50)
    except Exception as e:
        print(f"[WARNING] Hidden layer experiment failed: {e}")

    try:
        tuner.experiment_activations(epochs=50)
    except Exception as e:
        print(f"[WARNING] Activation experiment failed: {e}")

    try:
        tuner.experiment_learning_rates(epochs=50)
    except Exception as e:
        print(f"[WARNING] Learning rate experiment failed: {e}")

    try:
        tuner.experiment_early_stopping(epochs=80)
    except Exception as e:
        print(f"[WARNING] Early stopping experiment failed: {e}")

    # 汇总
    summary = tuner.generate_summary(
        save_path=os.path.join(save_dir, 'tuning_summary.json'))

    return summary, tuner


# ====================================================================
# 第五部分：模拟数据测试
# ====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Hyperparameter Tuning")
    print("=" * 60)

    # ── 从 src/ 数据管线加载真实数据 ──
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    import pandas as pd
    from config import FEATURES_DIR
    from sklearn.preprocessing import LabelEncoder

    DATASET = "letters"
    train_path = FEATURES_DIR / f"{DATASET}_train_features.npy"
    train_csv  = FEATURES_DIR / f"{DATASET}_train_manifest.csv"
    val_path   = FEATURES_DIR / f"{DATASET}_val_features.npy"
    val_csv    = FEATURES_DIR / f"{DATASET}_val_manifest.csv"

    if not train_path.exists():
        print(f"[错误] 特征文件不存在：{train_path}")
        print("  请先运行: python src/prepare_data.py --dataset letters")
        import sys as _sys
        _sys.exit(1)

    # 加载训练集
    X_train = np.load(train_path).astype(np.float32)
    y_train_raw = pd.read_csv(train_csv)["label"].values
    le = LabelEncoder()
    y_train_idx = le.fit_transform(y_train_raw)
    n_classes = len(le.classes_)
    class_names = list(le.classes_)

    # 加载验证集
    if val_path.exists() and val_csv.exists():
        X_val = np.load(val_path).astype(np.float32)
        y_val_raw = pd.read_csv(val_csv)["label"].values
        y_val_idx = le.transform(y_val_raw)
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train_idx, y_val_idx = train_test_split(
            X_train, y_train_idx, test_size=0.2, random_state=42, stratify=y_train_idx)

    # 从训练集再分一个测试集（src/ 预分割只有 train/val）
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train_idx, y_test_idx = train_test_split(
        X_train, y_train_idx, test_size=0.15, random_state=42, stratify=y_train_idx)

    # Z-Score 归一化
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # 独热编码
    y_train = np.eye(n_classes, dtype=np.float32)[y_train_idx]
    y_val = np.eye(n_classes, dtype=np.float32)[y_val_idx]
    y_test = np.eye(n_classes, dtype=np.float32)[y_test_idx]

    print(f"数据：Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"类别数：{n_classes}")

    # 运行完整流水线
    summary, tuner = run_full_tuning(
        X_train, y_train, X_val, y_val, X_test, y_test,
        num_classes=n_classes, class_names=class_names)

    print("\n[OK] All hyperparameter tuning experiments completed!")
    print(f"Results saved in: tuning_results/")
