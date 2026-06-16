import argparse
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import sys
from pathlib import Path

# 导入你写的BP神经网络类
from bp_network import BPNeuralNetwork

# ── 桥接 src/ 数据管线 ──
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import pandas as pd
from config import FEATURES_DIR
from audio_feature import feature_dim


# ==================== 配置参数 ====================
# 以下参数可通过命令行 --dataset 覆盖
# 网络参数
INPUT_DIM = feature_dim()  # 78 = 13 MFCC × 3 (mfcc/Δ/ΔΔ) × 2 (mean/std)
HIDDEN_DIMS = [128, 64]    # 两层隐含层：增强表达能力
ACTIVATION = "relu"        # 激活函数：sigmoid/relu/tanh
LR = 0.005                 # 初始学习率（降低，减缓过拟合速度）
LR_DECAY = 0.95            # 每轮学习率衰减系数（0.95^epoch，逐步减小步长）
MOMENTUM = 0.9             # 动量系数
WEIGHT_DECAY = 1e-3        # L2 正则化强度（加大防过拟合）
DROPOUT_RATE = 0.3         # Dropout 比率（训练时随机丢弃30%神经元）
CLIP_GRAD = 5.0            # 梯度裁剪阈值
# 训练参数
EPOCHS = 100              # 最大训练轮数
EARLY_STOP_PATIENCE = 15  # 早停耐心值：验证损失连续N轮不降则停止
BATCH_SIZE = 64           # 批次大小（None表示全量训练）
VALIDATION_SPLIT = 0.2    # 验证集比例
RANDOM_SEED = 42          # 随机种子（保证可复现）

# 工具函数
def load_pipeline_data(mfcc_path, manifest_path):
    """
    加载 prepare_data.py 产出的特征和标签，返回原始 X, y, label_encoder。
    标签从 manifest.csv 的 label 列读取。
    """
    X = np.load(mfcc_path)
    df = pd.read_csv(manifest_path)
    y = df["label"].values
    print(f"加载特征：{mfcc_path}")
    print(f"  特征形状：X={X.shape}")
    print(f"  标签数：{len(y)}, 唯一标签：{np.unique(y)[:5]}...")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X.astype(np.float32), y_encoded, label_encoder


def load_and_preprocess_data(mfcc_path, manifest_path,
                             val_mfcc_path=None, val_manifest_path=None):
    """
    加载 prepare_data.py 产出的数据并预处理：
    1. 从 .npy + .csv manifest 加载特征和标签
    2. 标签独热编码
    3. 若已由 src/ 预分割，使用 train/val 分片；否则自行划分
    4. Z-Score 归一化
    """
    # 1. 加载训练数据
    X_train_raw, y_train_raw, label_encoder = load_pipeline_data(
        mfcc_path, manifest_path)

    # 2. 加载验证数据（若存在预分割文件）
    if val_mfcc_path and Path(val_mfcc_path).exists() and \
       val_manifest_path and Path(val_manifest_path).exists():
        X_val_raw, y_val_raw, _ = load_pipeline_data(val_mfcc_path, val_manifest_path)
        print("  使用 prepare_data.py 预分割的 train/val 分片")
    else:
        # 自行划分
        X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
            X_train_raw, y_train_raw,
            test_size=VALIDATION_SPLIT,
            random_state=RANDOM_SEED,
            stratify=y_train_raw,
        )
        print("  未找到预分割 val 文件，自行划分 train/val")

    # 3. 标签独热编码
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y_train = onehot_encoder.fit_transform(y_train_raw.reshape(-1, 1))
    y_val = onehot_encoder.transform(y_val_raw.reshape(-1, 1))
    output_dim = y_train.shape[1]
    print(f"  类别数：{output_dim}")

    # 4. Z-Score 归一化（用训练集统计量）
    mean = np.mean(X_train_raw, axis=0)
    std = np.std(X_train_raw, axis=0) + 1e-8
    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std

    print(f"  训练集：X={X_train.shape}, y={y_train.shape}")
    print(f"  验证集：X={X_val.shape}, y={y_val.shape}")
    return X_train, X_val, y_train, y_val, output_dim, label_encoder, mean, std

def load_combined_data():
    """加载字母 + 单词联合数据集，标签加前缀区分（letter:A, word:apple）。"""
    all_X_train, all_y_train_raw = [], []
    all_X_val, all_y_val_raw = [], []

    for prefix, ds in [("letter", "letters"), ("word", "words")]:
        train_path = FEATURES_DIR / f"{ds}_train_features.npy"
        train_csv = FEATURES_DIR / f"{ds}_train_manifest.csv"
        val_path = FEATURES_DIR / f"{ds}_val_features.npy"
        val_csv = FEATURES_DIR / f"{ds}_val_manifest.csv"

        if not train_path.exists():
            print(f"  [SKIP] {ds} 特征不存在，跳过")
            continue

        X_tr = np.load(train_path).astype(np.float32)
        df_tr = pd.read_csv(train_csv)
        y_tr = [f"{prefix}:{lbl}" for lbl in df_tr["label"].values]
        all_X_train.append(X_tr)
        all_y_train_raw.extend(y_tr)

        if val_path.exists() and val_csv.exists():
            X_v = np.load(val_path).astype(np.float32)
            df_v = pd.read_csv(val_csv)
            y_v = [f"{prefix}:{lbl}" for lbl in df_v["label"].values]
            all_X_val.append(X_v)
            all_y_val_raw.extend(y_v)

    X_train_raw = np.concatenate(all_X_train, axis=0)
    X_val_raw = np.concatenate(all_X_val, axis=0)
    y_train_raw_arr = np.array(all_y_train_raw)
    y_val_raw_arr = np.array(all_y_val_raw)

    # 联合 LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_train_raw_arr, y_val_raw_arr]))
    y_train = label_encoder.transform(y_train_raw_arr)
    y_val = label_encoder.transform(y_val_raw_arr)
    output_dim = len(label_encoder.classes_)

    # One-hot
    onehot = OneHotEncoder(sparse_output=False)
    y_train_oh = onehot.fit_transform(y_train.reshape(-1, 1))
    y_val_oh = onehot.transform(y_val.reshape(-1, 1))

    # Z-Score 归一化
    mean = np.mean(X_train_raw, axis=0)
    std = np.std(X_train_raw, axis=0) + 1e-8
    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std

    print(f"联合数据集 | 类别数={output_dim} "
          f"({sum(1 for c in label_encoder.classes_ if c.startswith('letter:'))}字母 + "
          f"{sum(1 for c in label_encoder.classes_ if c.startswith('word:'))}单词)")
    print(f"  训练集：X={X_train.shape}, y={y_train_oh.shape}")
    print(f"  验证集：X={X_val.shape}, y={y_val_oh.shape}")
    return X_train, X_val, y_train_oh, y_val_oh, output_dim, label_encoder, mean, std


def generate_batches(X, y, batch_size):
    """生成批次数据（迭代器）"""
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)  # 每轮打乱顺序

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]

# ==================== 主训练函数 ====================
def train_bp_network(dataset: str = "letters"):
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    model_save_path = str(save_dir / f"bp_{dataset}_model.npz")
    plot_save_path = str(save_dir / f"bp_{dataset}_curve.png")

    # 1. 数据加载与预处理
    if dataset == "both":
        print("加载字母 + 单词联合数据集 …")
        X_train, X_val, y_train, y_val, output_dim, label_encoder, norm_mean, norm_std = \
            load_combined_data()
    else:
        mfcc_path = str(FEATURES_DIR / f"{dataset}_train_features.npy")
        manifest_path = str(FEATURES_DIR / f"{dataset}_train_manifest.csv")
        val_path = str(FEATURES_DIR / f"{dataset}_val_features.npy")
        val_manifest = str(FEATURES_DIR / f"{dataset}_val_manifest.csv")
        try:
            X_train, X_val, y_train, y_val, output_dim, label_encoder, norm_mean, norm_std = \
                load_and_preprocess_data(
                    mfcc_path, manifest_path,
                    val_path, val_manifest,
                )
        except FileNotFoundError as e:
            print(f"数据文件未找到：{e}")
            print("请先运行 src/prepare_data.py 生成特征文件，或检查路径配置！")
            return

    # 2. 初始化BP神经网络
    bp_net = BPNeuralNetwork(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        output_dim=output_dim,
        activation=ACTIVATION,
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        dropout_rate=DROPOUT_RATE,
        clip_grad=CLIP_GRAD,
    )
    print(f"BP网络初始化完成：输入维度={INPUT_DIM}, 隐含层={HIDDEN_DIMS}, 输出维度={output_dim}")

    # 3. 训练过程记录
    train_losses = []    # 训练损失
    val_losses = []      # 验证损失
    val_accuracies = []  # 验证精度
    best_val_loss = float("inf")
    best_epoch = 0
    early_stop_counter = 0
    current_lr = LR      # 动态学习率（会衰减）

    # 4. 训练循环
    print("\n开始训练...")
    for epoch in range(1, EPOCHS + 1):
        # ---------- 学习率衰减 ----------
        current_lr = LR * (LR_DECAY ** (epoch - 1))
        bp_net.lr = current_lr  # 同步到网络对象

        # ---------- 训练阶段 ----------
        epoch_train_loss = 0.0
        num_batches = 0

        # 批次训练（如果batch_size为None则全量训练）
        if BATCH_SIZE is None:
            loss = bp_net.train_step(X_train, y_train)
            epoch_train_loss += loss
            num_batches = 1
        else:
            for X_batch, y_batch in generate_batches(X_train, y_train, BATCH_SIZE):
                loss = bp_net.train_step(X_batch, y_batch)
                epoch_train_loss += loss
                num_batches += 1

        # 计算本轮平均训练损失
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)

        # ---------- 验证阶段 ----------
        # 验证集前向传播计算损失（关闭Dropout）
        bp_net.training = False
        y_val_pred = bp_net.forward(X_val)
        avg_val_loss = bp_net.cross_entropy_loss(y_val_pred, y_val)
        val_losses.append(avg_val_loss)

        # 计算验证精度
        val_pred_labels = bp_net.predict(X_val)
        val_true_labels = np.argmax(y_val, axis=1)
        val_accuracy = np.mean(val_pred_labels == val_true_labels)
        val_accuracies.append(val_accuracy)

        # ---------- 打印日志 ----------
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{EPOCHS}] | LR:{current_lr:.6f} | 训练损失：{avg_train_loss:.4f} | "
                  f"验证损失：{avg_val_loss:.4f} | 验证精度：{val_accuracy:.4f}")

        # ---------- 早停检查 + 保存最佳模型 ----------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            early_stop_counter = 0
            # 保存当前最佳模型
            save_dict = {
                "input_dim": bp_net.input_dim,
                "hidden_dims": np.array(bp_net.hidden_dims),
                "output_dim": bp_net.output_dim,
                "activation": bp_net.activation,
                "lr": bp_net.lr,
                "momentum": bp_net.momentum,
                "weight_decay": bp_net.weight_decay,
                "dropout_rate": bp_net.dropout_rate,
                "clip_grad": bp_net.clip_grad,
                "label_encoder_classes": label_encoder.classes_,
                "norm_mean": norm_mean,
                "norm_std": norm_std,
            }
            for i, w in enumerate(bp_net.weights):
                save_dict[f"weight_{i}"] = w
            for i, b in enumerate(bp_net.biases):
                save_dict[f"bias_{i}"] = b
            np.savez(model_save_path, **save_dict)
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"\n[EarlyStop] 验证损失已连续 {EARLY_STOP_PATIENCE} 轮未改善")
                print(f"  最佳验证损失：{best_val_loss:.4f} (Epoch {best_epoch})")
                break

    # 5. 模型已在前面的训练循环中保存为最佳版本
    print(f"\n最佳模型已保存至：{model_save_path} "
          f"(验证损失最低：{best_val_loss:.4f} @ Epoch {best_epoch})")

    # 6. 训练曲线可视化
    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="训练损失", color="blue")
    plt.plot(val_losses, label="验证损失", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("训练/验证损失曲线")
    plt.legend()
    plt.grid(alpha=0.3)

    # 精度曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="验证精度", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("验证精度曲线")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
    print(f"训练曲线已保存至：{plot_save_path}")
    plt.show()

    # 7. 最终评估
    print(f"\n训练完成！共 {epoch} 轮（早停触发）" if early_stop_counter >= EARLY_STOP_PATIENCE
          else f"\n训练完成！共 {EPOCHS} 轮（达到最大轮数）")
    print(f"  最佳验证损失：{best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"  最终验证精度：{val_accuracies[best_epoch - 1]:.4f}")

# ==================== 模型加载函数（可选） ====================
def load_bp_model(model_path):
    """加载保存的BP模型（含归一化参数）"""
    data = np.load(model_path, allow_pickle=True)
    hidden_dims = data["hidden_dims"].tolist()
    bp_net = BPNeuralNetwork(
        input_dim=int(data["input_dim"]),
        hidden_dims=hidden_dims,
        output_dim=int(data["output_dim"]),
        activation=str(data["activation"]),
        lr=float(data["lr"]),
        momentum=float(data.get("momentum", 0.0)),
        weight_decay=float(data.get("weight_decay", 0.0)),
        dropout_rate=float(data.get("dropout_rate", 0.0)),
        clip_grad=float(data.get("clip_grad", 0.0)),
    )
    # 按索引恢复权重和偏置
    num_layers = len(hidden_dims) + 1
    weights, biases = [], []
    for i in range(num_layers):
        weights.append(data[f"weight_{i}"])
        biases.append(data[f"bias_{i}"])
    bp_net.weights = weights
    bp_net.biases = biases
    label_encoder = LabelEncoder()
    label_encoder.classes_ = data["label_encoder_classes"]
    norm_mean = data.get("norm_mean", None)
    norm_std = data.get("norm_std", None)
    return bp_net, label_encoder, norm_mean, norm_std

# ==================== 运行训练 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BP neural network")
    parser.add_argument("--dataset", choices=["letters", "words", "both"], default="letters",
                        help="Dataset to train on")
    args = parser.parse_args()

    ds = args.dataset
    mfcc_path = FEATURES_DIR / f"{ds}_train_features.npy"
    print(f"特征维度：{feature_dim()}")
    print(f"数据集：{ds}")
    print(f"训练数据：{mfcc_path}")
    print()

    if not mfcc_path.exists():
        print(f"[提示] 特征文件不存在：{mfcc_path}")
        print(f"  请先运行: python src/prepare_data.py --dataset {ds}")
        sys.exit(1)

    train_bp_network(ds)
