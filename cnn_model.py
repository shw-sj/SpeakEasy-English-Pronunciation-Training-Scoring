import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import sys
from pathlib import Path

# ── 桥接 src/ 数据管线 ──
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import pandas as pd
from config import FEATURES_DIR
from audio_feature import feature_dim
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 纯手写一维CNN模型（无框架）
class CNN1D:
    def __init__(self, input_dim, num_classes, activation='relu', lr=0.01,
                 momentum=0.9, weight_decay=1e-4, dropout_rate=0.0, clip_grad=5.0):
        """
        一维CNN适配MFCC语音特征分类
        :param input_dim: MFCC特征维度 78/156/234
        :param num_classes: 输出类别 26(字母)/N(单词)
        :param activation: 激活函数 relu/sigmoid/tanh
        :param lr: 学习率
        :param momentum: 动量系数（0=无动量），加速收敛 + 抑制震荡
        :param weight_decay: L2正则化强度，防止过拟合
        :param dropout_rate: Dropout比率（0=无Dropout），训练时随机丢弃神经元
        :param clip_grad: 梯度裁剪阈值（0=不裁剪），防止梯度爆炸
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.activation = activation.lower()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.clip_grad = clip_grad
        self.training = True  # 训练/推理模式，控制Dropout

        # ================= 超参数（可调整） =================
        self.conv1_filters = 16    # 第一层卷积核数量
        self.conv1_kernel = 3      # 卷积核大小
        self.pool1_size = 2        # 池化窗口大小

        self.conv2_filters = 32    # 第二层卷积核数量
        self.conv2_kernel = 3
        self.pool2_size = 2

        self.fc_units = 128        # 全连接层神经元
        # ===================================================

        # 初始化卷积层+全连接层参数
        self._init_params()

    # 激活函数及导数
    def _act(self, z):
        if self.activation == 'relu': return np.maximum(0, z)
        if self.activation == 'sigmoid': return 1 / (1 + np.exp(-z))
        if self.activation == 'tanh': return np.tanh(z)
        return z

    def _act_deriv(self, z, a):
        if self.activation == 'relu': return np.where(z > 0, 1, 0)
        if self.activation == 'sigmoid': return a * (1 - a)
        if self.activation == 'tanh': return 1 - a**2
        return np.ones_like(z)

    # Softmax
    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # 交叉熵损失
    def cross_entropy_loss(self, y_pred, y_true):
        eps = 1e-8
        return -np.sum(y_true * np.log(y_pred + eps)) / y_pred.shape[0]

    # ================= 初始化网络参数 =================
    def _init_params(self):
        # 卷积层1参数 (kernel_size, in_channels, out_channels)
        # He初始化：std = sqrt(2 / fan_in)，适配ReLU
        fan_in_w1 = self.conv1_kernel * 1  # kernel_len × in_channels
        self.W1 = np.random.randn(self.conv1_kernel, 1, self.conv1_filters) * np.sqrt(2.0 / fan_in_w1)
        self.b1 = np.zeros((1, 1, self.conv1_filters))

        # 卷积层2参数
        fan_in_w2 = self.conv1_kernel * self.conv1_filters
        self.W2 = np.random.randn(self.conv2_kernel, self.conv1_filters, self.conv2_filters) * np.sqrt(2.0 / fan_in_w2)
        self.b2 = np.zeros((1, 1, self.conv2_filters))

        # 自动计算展平后的维度
        self.flat_dim = self._get_flatten_dim()

        # 全连接层
        self.W3 = np.random.randn(self.flat_dim, self.fc_units) * np.sqrt(2.0 / self.flat_dim)
        self.b3 = np.zeros((1, self.fc_units))

        # 输出层
        self.W4 = np.random.randn(self.fc_units, self.num_classes) * np.sqrt(2.0 / self.fc_units)
        self.b4 = np.zeros((1, self.num_classes))

        # 动量速度（初始化为零）
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
        self.v_W3 = np.zeros_like(self.W3)
        self.v_b3 = np.zeros_like(self.b3)
        self.v_W4 = np.zeros_like(self.W4)
        self.v_b4 = np.zeros_like(self.b4)

        # Dropout mask（反向传播时需要复用）
        self.dropout_mask = None

    # 自动计算展平维度
    def _get_flatten_dim(self):
        x = np.random.randn(1, self.input_dim, 1)
        x = self._conv1d_forward(x, self.W1, self.b1)
        x, _ = self._maxpool1d_forward(x, self.pool1_size)
        x = self._conv1d_forward(x, self.W2, self.b2)
        x, _ = self._maxpool1d_forward(x, self.pool2_size)
        return x.shape[1] * x.shape[2]

    # ================= 一维卷积 前向/反向 =================
    def _conv1d_forward(self, x, w, b):
        batch, len_in, chan_in = x.shape
        kernel_len, _, chan_out = w.shape
        len_out = len_in - kernel_len + 1
        out = np.zeros((batch, len_out, chan_out))
        for i in range(len_out):
            # x[:, i:i+k, :] -> (batch, k, chan_in) -> add axis -> (batch, k, chan_in, 1)
            # w -> (k, chan_in, chan_out) -> broadcast
            out[:, i, :] = np.sum(x[:, i:i+kernel_len, :, np.newaxis] * w, axis=(1, 2)) + b
        return out

    def _conv1d_backward(self, dout, x, w):
        batch, len_in, chan_in = x.shape
        kernel_len, _, chan_out = w.shape
        len_out = dout.shape[1]
        dx = np.zeros_like(x)
        dw = np.zeros_like(w)
        db = np.sum(dout, axis=(0, 1), keepdims=True)  # 保持形状 (1, 1, chan_out)
        for i in range(len_out):
            # dout[:, i, :] -> (batch, chan_out) -> (batch, 1, 1, chan_out)
            # w -> (k, chan_in, chan_out)
            # 乘积形状 (batch, k, chan_in, chan_out)，对 chan_out(axis=3) 求和
            dx[:, i:i+kernel_len, :] += np.sum(dout[:, i:i+1, np.newaxis, :] * w, axis=3)
            # x[:, i:i+k, :, np.newaxis] -> (batch, k, chan_in, 1)
            # dout[:, i, np.newaxis, :] -> (batch, 1, chan_out)
            dw += np.sum(x[:, i:i+kernel_len, :, np.newaxis] * dout[:, i:i+1, np.newaxis, :], axis=0)
        return dx, dw, db

    # ================= 一维最大池化 前向/反向 =================
    def _maxpool1d_forward(self, x, pool_size):
        batch, len_in, chan = x.shape
        len_out = len_in // pool_size
        out = np.zeros((batch, len_out, chan))
        pool_mask = np.zeros_like(x)
        for i in range(len_out):
            window = x[:, i*pool_size:(i+1)*pool_size, :]
            out[:,i,:] = np.max(window, axis=1)
            mask = window == np.max(window, axis=1, keepdims=True)
            pool_mask[:, i*pool_size:(i+1)*pool_size, :] = mask
        return out, pool_mask

    def _maxpool1d_backward(self, dout, pool_size, pool_mask):
        batch, len_out, chan = dout.shape
        dx = np.zeros_like(pool_mask)
        for i in range(len_out):
            dx[:, i*pool_size:(i+1)*pool_size, :] = dout[:,i:i+1,:] * pool_mask[:, i*pool_size:(i+1)*pool_size, :]
        return dx

    # ================= 整体前向传播 =================
    def forward(self, x):
        # 输入形状: (batch, input_dim, 1)
        self.x = x
        
        # 卷积1 -> 激活 -> 池化1
        self.z1 = self._conv1d_forward(x, self.W1, self.b1)
        self.a1 = self._act(self.z1)
        self.p1, self.pool1_mask = self._maxpool1d_forward(self.a1, self.pool1_size)

        # 卷积2 -> 激活 -> 池化2
        self.z2 = self._conv1d_forward(self.p1, self.W2, self.b2)
        self.a2 = self._act(self.z2)
        self.p2, self.pool2_mask = self._maxpool1d_forward(self.a2, self.pool2_size)
        
        # 展平
        self.flat = self.p2.reshape(self.p2.shape[0], -1)

        # 全连接层
        self.z3 = np.dot(self.flat, self.W3) + self.b3
        self.a3 = self._act(self.z3)

        # Dropout（仅在全连接层激活后，训练时随机丢弃神经元）
        if self.training and self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*self.a3.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
            self.a3_dropout = self.a3 * self.dropout_mask
        else:
            self.dropout_mask = np.ones_like(self.a3)
            self.a3_dropout = self.a3

        # 输出层 + Softmax
        self.z4 = np.dot(self.a3_dropout, self.W4) + self.b4
        self.out = self._softmax(self.z4)
        return self.out

    # ================= 反向传播 =================
    def backward(self, y_true):
        batch = y_true.shape[0]

        # 输出层梯度 (Softmax+交叉熵)
        dz4 = self.out - y_true
        dW4 = np.dot(self.a3_dropout.T, dz4) / batch + self.weight_decay * self.W4
        db4 = np.sum(dz4, axis=0, keepdims=True) / batch

        # 全连接层梯度（通过Dropout mask反向传播）
        da3_dropout = np.dot(dz4, self.W4.T)
        da3 = da3_dropout * self.dropout_mask
        dz3 = da3 * self._act_deriv(self.z3, self.a3)
        dW3 = np.dot(self.flat.T, dz3) / batch + self.weight_decay * self.W3
        db3 = np.sum(dz3, axis=0, keepdims=True) / batch

        # 展平反向
        dflat = np.dot(dz3, self.W3.T)
        dp2 = dflat.reshape(self.p2.shape)

        # 池化2反向
        da2 = self._maxpool1d_backward(dp2, self.pool2_size, self.pool2_mask)
        dz2 = da2 * self._act_deriv(self.z2, self.a2)

        # 卷积2反向
        dp1, dW2, db2 = self._conv1d_backward(dz2, self.p1, self.W2)
        dW2 = dW2 / batch + self.weight_decay * self.W2
        db2 = db2 / batch

        # 池化1反向
        da1 = self._maxpool1d_backward(dp1, self.pool1_size, self.pool1_mask)
        dz1 = da1 * self._act_deriv(self.z1, self.a1)

        # 卷积1反向
        dx, dW1, db1 = self._conv1d_backward(dz1, self.x, self.W1)
        dW1 = dW1 / batch + self.weight_decay * self.W1
        db1 = db1 / batch

        # 收集所有梯度（用于梯度裁剪）
        grads = [dW1, db1, dW2, db2, dW3, db3, dW4, db4]
        params_W = [self.W1, self.W2, self.W3, self.W4]
        params_b = [self.b1, self.b2, self.b3, self.b4]
        vel_W = [self.v_W1, self.v_W2, self.v_W3, self.v_W4]
        vel_b = [self.v_b1, self.v_b2, self.v_b3, self.v_b4]

        # 梯度裁剪（防梯度爆炸）
        if self.clip_grad > 0:
            for g in grads:
                np.clip(g, -self.clip_grad, self.clip_grad, out=g)

        # 动量梯度下降更新参数
        for i in range(4):
            vel_W[i] = self.momentum * vel_W[i] - self.lr * grads[i * 2]       # dW
            vel_b[i] = self.momentum * vel_b[i] - self.lr * grads[i * 2 + 1]   # db
            params_W[i] += vel_W[i]
            params_b[i] += vel_b[i]

    # ================= 单步训练 =================
    def train_step(self, x, y):
        self.training = True  # 启用Dropout
        y_pred = self.forward(x)
        loss = self.cross_entropy_loss(y_pred, y)
        self.backward(y)
        return loss

    # ================= 预测 =================
    def predict(self, x):
        self.training = False  # 关闭Dropout
        # 输入自动增加通道维度
        if x.ndim == 2:
            x = x.reshape(x.shape[0], x.shape[1], 1)
        y_pred = self.forward(x)
        return np.argmax(y_pred, axis=1)

# ====================== 2. 训练工具函数（和BP完全一致） ======================
def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

def calculate_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def save_model(model, path='cnn_mfcc_model.pkl'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path='cnn_mfcc_model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

def generate_batches(X, y, batch_size):
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]

def load_combined_data():
    """加载字母 + 单词联合数据集。"""
    import pandas as _pd
    all_X_train, all_y_train_raw = [], []
    all_X_val, all_y_val_raw = [], []

    for prefix, ds in [("letter", "letters"), ("word", "words")]:
        train_path = FEATURES_DIR / f"{ds}_train_features.npy"
        train_csv = FEATURES_DIR / f"{ds}_train_manifest.csv"
        val_path = FEATURES_DIR / f"{ds}_val_features.npy"
        val_csv = FEATURES_DIR / f"{ds}_val_manifest.csv"

        if not train_path.exists():
            continue
        X_tr = np.load(train_path).astype(np.float32)
        df_tr = _pd.read_csv(train_csv)
        all_X_train.append(X_tr)
        all_y_train_raw.extend(f"{prefix}:{lbl}" for lbl in df_tr["label"].values)
        if val_path.exists() and val_csv.exists():
            X_v = np.load(val_path).astype(np.float32)
            df_v = _pd.read_csv(val_csv)
            all_X_val.append(X_v)
            all_y_val_raw.extend(f"{prefix}:{lbl}" for lbl in df_v["label"].values)

    X_train_raw = np.concatenate(all_X_train, axis=0)
    X_val_raw = np.concatenate(all_X_val, axis=0)
    all_y = np.array(all_y_train_raw + all_y_val_raw)
    y_tr_arr = np.array(all_y_train_raw)
    y_v_arr = np.array(all_y_val_raw)

    le = LabelEncoder().fit(all_y)
    y_train = le.transform(y_tr_arr)
    y_val = le.transform(y_v_arr)

    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0) + 1e-8
    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std

    num_classes = len(le.classes_)
    n_letter = sum(1 for c in le.classes_ if c.startswith("letter:"))
    n_word = sum(1 for c in le.classes_ if c.startswith("word:"))
    print(f"联合数据集 | 类别={num_classes} ({n_letter}字母 + {n_word}单词)")
    print(f"  训练：{X_train.shape} | 验证：{X_val.shape}")
    return X_train, X_val, y_train, y_val, num_classes, le, mean, std

# ====================== 3. CNN训练主函数 ======================
def load_pipeline_data(dataset="letters"):
    """从 prepare_data.py 产物加载训练/验证集"""
    train_path = FEATURES_DIR / f"{dataset}_train_features.npy"
    train_csv  = FEATURES_DIR / f"{dataset}_train_manifest.csv"
    val_path   = FEATURES_DIR / f"{dataset}_val_features.npy"
    val_csv    = FEATURES_DIR / f"{dataset}_val_manifest.csv"

    if not train_path.exists():
        print(f"[错误] 特征文件不存在：{train_path}")
        print("  请先运行: python src/prepare_data.py --dataset letters")
        return None

    X_train_raw = np.load(train_path).astype(np.float32)
    y_train_raw = pd.read_csv(train_csv)["label"].values

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)

    # 预分割验证集
    if val_path.exists() and val_csv.exists():
        X_val_raw = np.load(val_path).astype(np.float32)
        y_val_raw = pd.read_csv(val_csv)["label"].values
        y_val = le.transform(y_val_raw)
    else:
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X_train_raw, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Z-Score 归一化
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0) + 1e-8
    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std

    num_classes = len(le.classes_)
    print(f"数据加载完成 | 类别={num_classes} | 特征维度={X_train.shape[1]}")
    print(f"  训练集：{X_train.shape} | 验证集：{X_val.shape}")
    return X_train, X_val, y_train, y_val, num_classes, le, mean, std


def train_cnn_network(dataset="letters"):
    # ============== 【配置参数】和BP完全对齐 ==============
    ACTIVATION = "relu"           # 激活函数
    LEARNING_RATE = 0.005         # 初始学习率（与BP对齐）
    LR_DECAY = 0.98               # 每轮学习率衰减系数（减缓衰减）
    EPOCHS = 100                  # 最大训练轮数（和BP对齐）
    EARLY_STOP_PATIENCE = 15      # 早停耐心值
    BATCH_SIZE = 32
    MOMENTUM = 0.9                # 动量系数
    WEIGHT_DECAY = 1e-3           # L2 正则化强度
    DROPOUT_RATE = 0.3            # Dropout 比率
    CLIP_GRAD = 5.0               # 梯度裁剪阈值
    # ======================================================

    # 加载真实数据
    if dataset == "both":
        print("加载字母 + 单词联合数据集 …")
        data = load_combined_data()
        if data is None: return
    else:
        data = load_pipeline_data(dataset)
        if data is None: return
    X_train, X_val, y_train_idx, y_val_idx, NUM_CLASSES, label_encoder, norm_mean, norm_std = data
    MFCC_DIM = X_train.shape[1]

    print("="*60)
    print(f"CNN训练 | 数据集：{dataset} | 类别：{NUM_CLASSES} | MFCC：{MFCC_DIM}维")
    print("="*60)

    # 独热编码 + 增加CNN通道维度
    y_train = one_hot_encode(y_train_idx, NUM_CLASSES)
    y_val = one_hot_encode(y_val_idx, NUM_CLASSES)
    X_train = X_train.reshape(-1, MFCC_DIM, 1)
    X_val = X_val.reshape(-1, MFCC_DIM, 1)

    # 初始化CNN（含全部正则化参数）
    cnn = CNN1D(
        input_dim=MFCC_DIM,
        num_classes=NUM_CLASSES,
        activation=ACTIVATION,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        dropout_rate=DROPOUT_RATE,
        clip_grad=CLIP_GRAD,
    )

    # 训练记录
    train_loss, val_loss, val_acc = [], [], []
    best_val_loss = float("inf")
    best_epoch = 0
    early_stop_counter = 0

    # 训练循环
    print(f"\n开始训练CNN... (正则化: momentum={MOMENTUM}, weight_decay={WEIGHT_DECAY}, "
          f"dropout={DROPOUT_RATE}, clip_grad={CLIP_GRAD})\n")
    for epoch in range(EPOCHS):
        # 学习率衰减
        current_lr = LEARNING_RATE * (LR_DECAY ** epoch)
        cnn.lr = current_lr

        # 批次训练
        epoch_loss = 0
        for Xb, yb in generate_batches(X_train, y_train, BATCH_SIZE):
            epoch_loss += cnn.train_step(Xb, yb) * len(Xb)
        avg_loss = epoch_loss / len(X_train)
        train_loss.append(avg_loss)

        # 验证
        yp_val = cnn.predict(X_val)
        vl = cnn.cross_entropy_loss(cnn.forward(X_val), y_val)
        va = calculate_accuracy(yp_val, y_val_idx)
        val_loss.append(vl)
        val_acc.append(va)

        # 日志
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d} | LR:{current_lr:.6f} | 训练损失：{avg_loss:.4f} | "
                  f"验证损失：{vl:.4f} | 验证精度：{va:.2%}")

        # 早停检查 + 保存最优模型（基于验证损失）
        if vl < best_val_loss:
            best_val_loss = vl
            best_epoch = epoch
            early_stop_counter = 0
            save_model({
                "model": cnn,
                "label_encoder": label_encoder,
                "norm_mean": norm_mean,
                "norm_std": norm_std,
            }, f'models/best_cnn_{dataset}_model.pkl')
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"\n[早停] 验证损失已连续 {EARLY_STOP_PATIENCE} 轮未改善")
                print(f"  最佳验证损失：{best_val_loss:.4f} (Epoch {best_epoch})")
                break

    # 训练结果
    print(f"\n训练完成！共 {epoch + 1} 轮 | 最佳验证损失：{best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"  最佳验证精度：{val_acc[best_epoch]:.2%}")

    # 可视化 + 保存（与BP绘图完全一致）
    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="训练损失", color="blue")
    plt.plot(val_loss, label="验证损失", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("训练/验证损失曲线")
    plt.legend()
    plt.grid(alpha=0.3)

    # 精度曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_acc, label="验证精度", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("验证精度曲线")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    from pathlib import Path as _Path
    _Path("models").mkdir(exist_ok=True)
    plt.savefig(f"models/cnn_{dataset}_curve.png", dpi=300, bbox_inches="tight")
    print(f"训练曲线已保存：models/cnn_{dataset}_curve.png")
    plt.show()

# ====================== 运行训练 ======================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train 1D-CNN model")
    parser.add_argument("--dataset", choices=["letters", "words", "both"], default="letters",
                        help="Dataset to train on")
    args = parser.parse_args()

    ds = args.dataset
    if ds == "both":
        train_cnn_network("both")
    else:
        train_path = FEATURES_DIR / f"{ds}_train_features.npy"
        if not train_path.exists():
            print("[提示] 特征文件不存在，请先运行数据管线：")
            print(f"  python src/prepare_data.py --dataset {ds}")
        else:
            train_cnn_network(ds)