# SpeakEasy — English Pronunciation Training & Scoring

基于深度学习的英语**字母发音**智能评测系统。支持 A-Z 共 26 个字母的发音识别与三维度融合评分。

## 项目结构

```
├── main.py                  # GUI 应用入口
├── src/
│   ├── gui_app.py           # 图形界面主程序
│   ├── audio_collector.py   # 音频采集模块
│   ├── audio_preprocess.py  # 音频预处理（降噪、VAD、归一化）
│   ├── audio_feature.py     # MFCC 时序特征提取
│   ├── data_augmentation.py # 数据增强（噪声、音高、速度、混响等）
│   ├── template_builder.py  # 标准发音模板库构建
│   ├── prepare_data.py      # 数据集准备（特征提取 + 划分）
│   ├── config.py            # 全局路径/常量配置
│   └── train_utils.py       # 通用训练工具
├── lstm_network.py          # 双向 LSTM + 注意力池化模型
├── lstm_train.py            # LSTM 训练脚本
├── dtw_matcher.py           # DTW 动态时间规整匹配器
├── pronunciation_scorer.py  # 三维度融合发音评分器（核心模块）
├── metrics.py               # 评价指标与可视化（分类/评分/图表）
├── api.py                   # 外部 API 集成（Free Dictionary / 有道 / DeepSeek）
├── hyperparam_tuning.py     # 超参数调优
├── config.json              # 全局配置文件
├── data/
│   ├── templates/           # 标准发音模板音频
│   ├── processed/           # 增强后的音频数据（按说话人/字母组织）
│   └── manifests/           # 数据集划分清单
└── results/                 # 训练结果（模型权重 + 训练曲线）
```

## 核心功能

### 1. 语音采集与预处理
- 实时录音采集，支持 GUI 交互
- 音频预处理：降噪、VAD 静音切除、响度归一化
- 数据增强：白噪声/粉红噪声/棕色噪声、音高偏移、语速变化、房间混响、RIR 卷积混响、音量调节等

### 2. MFCC 特征提取
- 每帧 39 维特征：13 维 MFCC + 13 维 Δ + 13 维 ΔΔ
- 保留完整时间轴 `(T, 39)`，供 LSTM 学习发音过程
- 帧长 25ms，帧移 10ms，16kHz 采样率

### 3. 深度学习模型（字母识别）

| 模型 | 架构特点 | 输入维度 |
|------|---------|---------|
| **BiLSTM** | 双向 LSTM + 时间注意力池化 + LayerNorm | 可变长度 `(T, 39)` 时序特征 |

训练特性：
- **Label smoothing**：减少过度自信
- **ReduceLROnPlateau**：根据验证损失自动降低学习率
- **Gradient Clipping**：稳定循环网络训练

### 4. 三维度融合发音评分 ⭐

```
最终得分 = 0.40 × 置信度评分 + 0.35 × DTW 对齐评分 + 0.25 × 声学相似度评分
```

| 维度 | 权重 | 方法 |
|------|------|------|
| 置信度评分 | 40% | 神经网络 Softmax 目标类概率 |
| DTW 对齐评分 | 35% | 用户 MFCC 序列与标准模板 DTW 规整距离 |
| 声学相似度 | 25% | F0 基频 + 共振峰 + 能量包络余弦相似度 |

评分等级：
- 90-100：优秀 ★★★★★
- 75-89：良好 ★★★★☆
- 60-74：一般 ★★★☆☆
- 0-59：需努力 ★★☆☆☆

### 5. 外部 API 集成
- **Free Dictionary API**：查询单词音标、释义、标准发音音频
- **有道智云语音评测 API**：对用户发音进行评分（综合/准确度/流利度/语速）
- **DeepSeek API**：结合评分结果给出智能改进建议

## 快速开始

### 环境要求
- Python 3.10+
- PyTorch 2.0+
- 依赖安装：`pip install -r requirements.txt`

### 数据准备

```bash
# 1. 采集原始音频
python src/audio_collector.py

# 2. 数据增强
python src/data_augmentation.py

# 3. 提取时序特征并划分数据集
python src/prepare_data.py --export-sequences
```

### 模型训练

```bash
# 训练 LSTM 网络
python lstm_train.py
```

### 启动 GUI 应用

```bash
python main.py
```

### 命令行评分

```python
from pronunciation_scorer import PronunciationScorer, ScoreResult

scorer = PronunciationScorer()
result = scorer.score(probs, target_idx,
                      user_mfcc_seq=user_seq, template_mfcc_seq=ref_seq,
                      user_audio=user_wav, template_audio=ref_wav)
print(result.total_score, result.grade, result.star_rating)
```

## 配置文件

[config.json](config.json) 主要配置项：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `sample_rate` | 采样率 | 16000 Hz |
| `mfcc.n_mfcc` | MFCC 系数数量 | 13 |
| `mfcc.n_mel_filters` | Mel 滤波器组数量 | 26 |
| `models.lstm_letters` | LSTM 模型权重路径 | results/best_lstm_letters_best_acc.pth |
| `standard_audio_dir` | 标准发音模板目录 | data/templates/gui_standard |
| `default_duration_seconds` | 默认录音时长 | 1.5s |

## 评价指标

- **分类指标**：各类别 Precision / Recall / F1-score / Accuracy
- **评分评测**：Pearson r / MAE / RMSE（机器评分 vs 人工评分）
- **可视化**：混淆矩阵热力图、训练曲线、F1 柱状图、模型对比雷达图
- 详见 [metrics.py](metrics.py)
