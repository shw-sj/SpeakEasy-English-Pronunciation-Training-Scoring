import numpy as np

class BPNeuralNetwork:
    """
    手写BP神经网络，适配MFCC语音特征分类
    网络结构：输入层 → 1~2层隐含层 → 输出层(Softmax)
    激活函数：sigmoid / relu / tanh
    损失函数：交叉熵损失(Cross Entropy) + Softmax
    """
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu', lr=0.01,
                 momentum=0.9, weight_decay=1e-4, dropout_rate=0.0, clip_grad=5.0):
        """
        初始化BP网络
        :param input_dim: 输入层维度（MFCC特征维度：78/156/234）
        :param hidden_dims: 隐含层神经元列表，长度1=1层隐含层，长度2=2层隐含层 例：[256] / [256,128]
        :param output_dim: 输出层维度（字母26，单词N）
        :param activation: 激活函数：sigmoid/relu/tanh
        :param lr: 学习率
        :param momentum: 动量系数（0=无动量），加速收敛 + 抑制震荡
        :param weight_decay: L2正则化强度，防止过拟合。越大越抑制大权重
        :param dropout_rate: Dropout比率（0=无Dropout），训练时随机丢弃神经元防过拟合
        :param clip_grad: 梯度裁剪阈值（0=不裁剪），防止梯度爆炸
        """

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation.lower()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.clip_grad = clip_grad
        self.training = True  # 训练/推理模式标志，控制Dropout行为
        self.num_hidden_layers = len(hidden_dims)  # 隐含层层数（1或2）

        # 校验隐含层层数（仅支持1~2层）
        assert self.num_hidden_layers in [1, 2], "隐含层仅支持1层或2层！"

        # 1. 初始化权重和偏置（Xavier初始化，避免梯度消失/爆炸）
        self.weights = []
        self.biases = []

        # 输入层 → 第一层隐含层
        self.weights.append(np.random.randn(input_dim, hidden_dims[0]) * np.sqrt(2.0 / input_dim))
        self.biases.append(np.zeros((1, hidden_dims[0])))

        # 若有第二层隐含层：第一层隐含层 → 第二层隐含层
        if self.num_hidden_layers == 2:
            self.weights.append(np.random.randn(hidden_dims[0], hidden_dims[1]) * np.sqrt(2.0 / hidden_dims[0]))
            self.biases.append(np.zeros((1, hidden_dims[1])))

        # 最后一层隐含层 → 输出层
        last_hidden_dim = hidden_dims[-1]
        self.weights.append(np.random.randn(last_hidden_dim, output_dim) * np.sqrt(2.0 / last_hidden_dim))
        self.biases.append(np.zeros((1, output_dim)))

        # 存储前向传播的中间值（反向传播用）
        self.layer_z = []               # 每层线性加权和 z = Wx + b
        self.layer_a = []               # 每层激活后的值 a = activate(z)，不含Dropout
        self.dropout_masks = []         # Dropout mask（反向传播时需要复用）

        # 动量速度（初始化为0）
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.v_biases = [np.zeros_like(b) for b in self.biases]

    # ==================== 激活函数及导数 ====================
    def _sigmoid(self, z):
        # 裁剪极端值防止 np.exp 溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _sigmoid_deriv(self, a):
        # Sigmoid导数用输出值计算：a*(1-a)，效率更高
        return a * (1 - a)

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_deriv(self, z):
        return np.where(z > 0, 1, 0)

    def _tanh(self, z):
        return np.tanh(z)

    def _tanh_deriv(self, a):
        # Tanh导数用输出值计算：1-a²
        return 1 - np.power(a, 2)

    def _activate(self, z):
        """根据配置调用对应激活函数"""
        if self.activation == 'sigmoid':
            return self._sigmoid(z)
        elif self.activation == 'relu':
            return self._relu(z)
        elif self.activation == 'tanh':
            return self._tanh(z)
        else:
            raise ValueError("激活函数仅支持：sigmoid/relu/tanh")

    def _activate_deriv(self, z, a):
        """根据配置调用对应激活函数导数"""
        if self.activation == 'sigmoid':
            return self._sigmoid_deriv(a)
        elif self.activation == 'relu':
            return self._relu_deriv(z)
        elif self.activation == 'tanh':
            return self._tanh_deriv(a)
        else:
            raise ValueError("激活函数仅支持：sigmoid/relu/tanh")

    # ==================== Softmax（输出层专用） ====================
    def _softmax(self, z):
        """数值稳定版Softmax，避免指数爆炸"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # ==================== 前向传播 ====================
    def forward(self, X):
        """
        前向传播
        :param X: 输入数据，shape=[样本数, 输入维度]
        :return: 输出层Softmax结果
        """
        self.layer_z = []       # 清空中间值
        self.layer_a = []
        self.dropout_masks = [] # 清空Dropout mask

        # 输入层
        a = X
        self.layer_a.append(a)

        # 遍历所有隐含层 + 输出层
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.layer_z.append(z)

            if i == len(self.weights) - 1:
                # 输出层：Softmax，无Dropout
                a = self._softmax(z)
                self.layer_a.append(a)  # 输出结果也存入，供backward取用
            else:
                # 隐含层：先激活，再Dropout
                a_activated = self._activate(z)
                self.layer_a.append(a_activated)  # 存储激活后、Dropout前的值（供反向传播导数用）
                # Dropout：训练时随机丢弃神经元
                if self.training and self.dropout_rate > 0:
                    mask = (np.random.rand(*a_activated.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
                    self.dropout_masks.append(mask)
                    a = a_activated * mask
                else:
                    self.dropout_masks.append(np.ones_like(a_activated))
                    a = a_activated

        return self.layer_a[-1]  # 返回输出层结果

    # ==================== 交叉熵损失 ====================
    def cross_entropy_loss(self, y_pred, y_true):
        """
        交叉熵损失（配合Softmax）
        :param y_pred: 网络输出(Softmax后)，shape=[样本数, 输出维度]
        :param y_true: 真实标签(独热编码)，shape=[样本数, 输出维度]
        :return: 损失值
        """
        epsilon = 1e-8  # 防止log(0)报错
        loss = -np.sum(y_true * np.log(y_pred + epsilon)) / y_pred.shape[0]
        return loss

    # ==================== 反向传播（核心） ====================
    def backward(self, y_true):
        """
        反向传播计算梯度
        :param y_true: 真实标签(独热编码)
        """
        num_samples = y_true.shape[0]
        # 存储梯度（累积列表）
        grad_W = []
        grad_b = []

        # ------------------- 1. 输出层梯度 -------------------
        # Softmax+交叉熵的简化梯度：dL/dz = y_pred - y_true
        a_out = self.layer_a[-1]
        dz = a_out - y_true

        # 输出层权重、偏置梯度（+ L2 正则化梯度）
        # 注意：输入到输出层的值包含Dropout，需使用post-dropout值
        a_input_to_output = self.layer_a[-2] * self.dropout_masks[-1]
        dW = np.dot(a_input_to_output.T, dz) / num_samples + self.weight_decay * self.weights[-1]
        db = np.sum(dz, axis=0, keepdims=True) / num_samples
        grad_W.append(dW)
        grad_b.append(db)

        # ------------------- 2. 反向传播到隐含层 -------------------
        # 倒序遍历权重（跳过输出层，处理隐含层）
        for i in reversed(range(len(self.weights) - 1)):
            # dL/d(post_dropout_of_layer_i)
            dL_da_post = np.dot(dz, self.weights[i + 1].T)
            # 通过Dropout mask反向传播
            dL_da = dL_da_post * self.dropout_masks[i]
            # 通过激活函数反向传播（使用pre-dropout的激活值计算导数）
            dz = dL_da * self._activate_deriv(self.layer_z[i], self.layer_a[i + 1])

            # 权重梯度：输入也需使用post-dropout值
            if i == 0:
                a_input = self.layer_a[0]   # 第一层输入 = X，无Dropout
            else:
                a_input = self.layer_a[i] * self.dropout_masks[i - 1]

            dW = np.dot(a_input.T, dz) / num_samples + self.weight_decay * self.weights[i]
            db = np.sum(dz, axis=0, keepdims=True) / num_samples
            grad_W.append(dW)
            grad_b.append(db)

        # 反转梯度（与权重顺序一致）
        grad_W.reverse()
        grad_b.reverse()

        # ------------------- 3. 梯度裁剪（防梯度爆炸） -------------------
        if self.clip_grad > 0:
            for g in grad_W + grad_b:
                np.clip(g, -self.clip_grad, self.clip_grad, out=g)

        # ------------------- 4. 动量梯度下降更新参数 -------------------
        for i in range(len(self.weights)):
            self.v_weights[i] = self.momentum * self.v_weights[i] - self.lr * grad_W[i]
            self.v_biases[i] = self.momentum * self.v_biases[i] - self.lr * grad_b[i]
            self.weights[i] += self.v_weights[i]
            self.biases[i] += self.v_biases[i]

    # ==================== 单步训练 ====================
    def train_step(self, X, y_true):
        """单步训练：前向→计算损失→反向→更新参数"""
        self.training = True
        y_pred = self.forward(X)
        loss = self.cross_entropy_loss(y_pred, y_true)
        self.backward(y_true)
        return loss

    # ==================== 预测 ====================
    def predict(self, X):
        """预测类别：返回概率最大的索引（自动关闭Dropout）"""
        self.training = False
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
