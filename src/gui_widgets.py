"""Custom widgets and helper classes for SpeakEasy GUI."""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QEasingCurve, QPropertyAnimation, QRectF, Qt, pyqtProperty
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from audio_feature import compute_mfcc_frames
from audio_preprocess import load_audio, preprocess_audio
from config import PROJECT_ROOT, SAMPLE_RATE, TEMPLATES_DIR

ROOT = PROJECT_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bp_network import BPNetwork
from cnn_network import CNN1D

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "SimSun",
    "Noto Sans CJK SC", "Arial Unicode MS",
]
matplotlib.rcParams["axes.unicode_minus"] = False


CHINESE_MEANING = {
    "apple": "苹果", "book": "书", "cat": "猫", "dog": "狗",
    "egg": "鸡蛋", "fish": "鱼", "goat": "山羊", "hat": "帽子",
    "ice": "冰", "jam": "果酱", "key": "钥匙", "lion": "狮子",
    "map": "地图", "net": "网", "owl": "猫头鹰", "pen": "钢笔",
    "queen": "女王", "rat": "老鼠", "sun": "太阳", "tree": "树",
    "water": "水",
}


@dataclass
class PredictionResult:
    label: str
    confidence: float
    score: int
    labels: list[str]
    probabilities: np.ndarray


def display_label(label: str) -> str:
    return label.split(":", 1)[1] if ":" in label else label


def score_text(score: int) -> tuple[str, str, str]:
    if score >= 90:
        return "优秀！", "★★★★★", "#3bb273"
    if score >= 75:
        return "良好", "★★★★☆", "#4b8fe8"
    if score >= 60:
        return "一般，需要练习", "★★★☆☆", "#f2a541"
    return "再来一次！", "★★☆☆☆", "#e85d5d"


def _score_color(s: float) -> str:
    if s >= 90: return "#3bb273"
    if s >= 75: return "#4b8fe8"
    if s >= 60: return "#f2a541"
    return "#e85d5d"


# ═══════════════════════════════════════════════════════
# Custom Widgets
# ═══════════════════════════════════════════════════════

class CircularScoreWidget(QWidget):
    def __init__(self, size: int = 130) -> None:
        super().__init__()
        self._value = 0
        self._color = QColor("#e85d5d")
        self._base_size = size
        self._font_scale = 1.0
        self.setMinimumSize(size, size)
        self.setMaximumSize(size, size)

    def get_value(self) -> int: return self._value

    def set_value(self, value: int) -> None:
        self._value = max(0, min(100, int(value)))
        self._color = QColor(_score_color(self._value))
        self.update()

    value = pyqtProperty(int, get_value, set_value)

    def animate_to(self, value: int) -> None:
        self.animation = QPropertyAnimation(self, b"value", self)
        self.animation.setDuration(650)
        self.animation.setStartValue(self._value)
        self.animation.setEndValue(value)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        self.animation.start()

    def set_scale(self, scale: float) -> None:
        """按比例缩放整个组件"""
        self._font_scale = scale
        sz = int(self._base_size * scale)
        self.setMinimumSize(sz, sz)
        self.setMaximumSize(sz, sz)
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        side = min(self.width(), self.height()) - int(16 * self._font_scale)
        rect = QRectF(
            (self.width() - side) / 2, (self.height() - side) / 2, side, side)

        pen_w = max(1, int(12 * self._font_scale))
        base_pen = QPen(QColor("#e8edf3"), pen_w)
        base_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(base_pen)
        painter.drawArc(rect, 0, 360 * 16)

        value_pen = QPen(self._color, pen_w)
        value_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(value_pen)
        painter.drawArc(rect, 90 * 16, -int(360 * 16 * self._value / 100))

        painter.setPen(QColor("#253043"))
        font_sz = max(8, int(22 * self._font_scale))
        painter.setFont(QFont("Microsoft YaHei", font_sz, QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, str(self._value))


class ChartPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.figure = Figure(figsize=(8, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.clear()

    def clear(self) -> None:
        self.figure.clear()
        axes = self.figure.subplots(2, 2)
        titles = ["录音波形", "MFCC频谱", "置信度分布", "标准/用户对比"]
        for ax, title in zip(axes.flat, titles):
            ax.set_title(title)
            ax.grid(alpha=0.2)
        self.canvas.draw_idle()

    def update_live_wave(self, audio: np.ndarray) -> None:
        self.figure.axes[0].clear()
        ax = self.figure.axes[0]
        if audio.size:
            t = np.arange(len(audio)) / SAMPLE_RATE
            ax.plot(t, audio, color="#4b8fe8", linewidth=1.0)
        ax.set_title("实时录音波形")
        ax.set_xlabel("时间(s)")
        ax.set_ylabel("振幅")
        ax.grid(alpha=0.2)
        self.canvas.draw_idle()

    def update_result(self, audio, labels, probabilities, target, standard_audio):
        self.figure.clear()
        ax_wave, ax_mfcc, ax_conf, ax_cmp = self.figure.subplots(2, 2).flat

        t = np.arange(len(audio)) / SAMPLE_RATE
        ax_wave.plot(t, audio, color="#4b8fe8", linewidth=1.0)
        ax_wave.set_title("用户录音波形")
        ax_wave.set_xlabel("时间(s)")
        ax_wave.grid(alpha=0.2)

        processed = preprocess_audio(audio, SAMPLE_RATE)
        mfcc = compute_mfcc_frames(processed, SAMPLE_RATE)
        ax_mfcc.imshow(mfcc.T, aspect="auto", origin="lower", cmap="magma")
        ax_mfcc.set_title("用户MFCC频谱")
        ax_mfcc.set_xlabel("帧")
        ax_mfcc.set_ylabel("MFCC")

        shown_labels = [display_label(x) for x in labels]
        colors = ["#9fb5d6"] * len(labels)
        target_display = display_label(target or "")
        for i, lbl in enumerate(shown_labels):
            if lbl == target_display:
                colors[i] = "#f2a541"
        ax_conf.bar(shown_labels, probabilities * 100, color=colors)
        ax_conf.set_title("置信度分布")
        ax_conf.set_ylabel("%")
        ax_conf.tick_params(axis="x", labelrotation=90, labelsize=7)
        ax_conf.set_ylim(0, max(100, float(probabilities.max() * 115)))

        ax_cmp.set_title("标准/用户波形对比")
        ax_cmp.plot(t, audio, color="#4b8fe8", linewidth=1.0, label="用户")
        if standard_audio is not None and standard_audio.size:
            ts = np.arange(len(standard_audio)) / SAMPLE_RATE
            offset = max(0.2, float(np.max(np.abs(audio))) + 0.1)
            ax_cmp.plot(ts, standard_audio + offset, color="#3bb273",
                        linewidth=1.0, label="标准")
        ax_cmp.legend(loc="upper right")
        ax_cmp.grid(alpha=0.2)

        self.figure.tight_layout()
        self.canvas.draw_idle()


class ModelManager:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache: dict = {}

    def checkpoint_path(self, model_name: str, dataset: str) -> Path:
        if model_name == "BP":
            return ROOT / "results" / f"bp_{dataset}_best_acc.pth"
        if model_name == "CNN":
            return ROOT / "results" / f"best_cnn_{dataset}_best_acc.pth"
        raise ValueError(model_name)

    def load(self, model_name: str, dataset: str):
        key = (model_name, dataset)
        if key in self.cache:
            return self.cache[key]
        path = self.checkpoint_path(model_name, dataset)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在：{path}")
        data = torch.load(path, map_location=self.device, weights_only=False)
        labels = [str(x) for x in data["label_encoder_classes"]]
        mean = data.get("norm_mean")
        std = data.get("norm_std")
        output_dim = int(data.get("output_dim", data.get("num_classes", len(labels))))
        input_dim = int(data["input_dim"])
        if model_name == "BP":
            model = BPNetwork(input_size=input_dim, output_size=output_dim,
                              dropout_rate=float(data.get("dropout_rate", 0.0)),
                              task=str(data.get("task", dataset))).to(self.device)
        else:
            model = CNN1D(input_dim=input_dim, num_classes=output_dim,
                          dropout_rate=float(data.get("dropout_rate", 0.3))).to(self.device)
        model.load_state_dict(data["state_dict"])
        model.eval()
        loaded = (model, labels, mean, std)
        self.cache[key] = loaded
        return loaded

    def predict(self, model_name: str, dataset: str, feature: np.ndarray) -> PredictionResult:
        if model_name == "融合模型":
            bp = self._predict_one("BP", dataset, feature)
            cnn = self._predict_one("CNN", dataset, feature)
            if bp.labels != cnn.labels:
                return cnn
            probs = (bp.probabilities + cnn.probabilities) / 2
            idx = int(np.argmax(probs))
            return PredictionResult(bp.labels[idx], float(probs[idx]),
                                    int(round(probs[idx] * 100)), bp.labels, probs)
        return self._predict_one(model_name, dataset, feature)

    def _predict_one(self, model_name: str, dataset: str, feature: np.ndarray) -> PredictionResult:
        model, labels, mean, std = self.load(model_name, dataset)
        x = feature.astype(np.float32)
        if mean is not None and std is not None:
            x = (x - mean) / (std + 1e-8)
        x_tensor = torch.as_tensor(x[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return PredictionResult(labels[idx], float(probs[idx]),
                                int(round(probs[idx] * 100)), labels, probs)


class StandardSpeech:
    def __init__(self) -> None:
        self.output_dir = TEMPLATES_DIR / "gui_standard"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, text: str) -> Path:
        safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
        return self.output_dir / f"{safe}.wav"

    def ensure_audio(self, text: str, slow: bool = False) -> Path:
        suffix = "_slow" if slow else ""
        path = self.path_for(text + suffix)
        if path.exists():
            return path
        import pyttsx3
        engine = pyttsx3.init()
        rate = engine.getProperty("rate")
        engine.setProperty("rate", int(rate * (0.75 if slow else 1.0)))
        engine.save_to_file(text, str(path))
        engine.runAndWait()
        return path

    def load(self, text: str, slow: bool = False) -> np.ndarray | None:
        path = self.ensure_audio(text, slow)
        if not path.exists():
            return None
        audio, _ = load_audio(str(path), SAMPLE_RATE)
        return audio

    def play(self, text: str, slow: bool = False) -> np.ndarray | None:
        audio = self.load(text, slow)
        if audio is None:
            return None
        sd.play(audio, SAMPLE_RATE)
        return audio


class StatsDialog(QDialog):
    def __init__(self, history: list[dict], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("学习统计")
        self.resize(820, 520)
        layout = QVBoxLayout(self)
        fig = Figure(figsize=(8, 5), tight_layout=True)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        ax1, ax2, ax3 = fig.subplots(1, 3)
        if not history:
            ax1.text(0.5, 0.5, "暂无练习记录", ha="center", va="center")
            ax1.axis("off"); ax2.axis("off"); ax3.axis("off")
            canvas.draw_idle()
            return
        labels = sorted({h["target"] for h in history})
        means = [np.mean([h["score"] for h in history if h["target"] == l]) for l in labels]
        counts = [sum(1 for h in history if h["target"] == l) for l in labels]
        scores = [h["score"] for h in history]
        ax1.bar(labels, means, color="#4b8fe8")
        ax1.set_title("平均分"); ax1.tick_params(axis="x", labelrotation=90, labelsize=8); ax1.set_ylim(0, 100)
        ax2.bar(labels, counts, color="#3bb273")
        ax2.set_title("练习次数"); ax2.tick_params(axis="x", labelrotation=90, labelsize=8)
        ax3.plot(range(1, len(scores)+1), scores, marker="o", color="#f2a541")
        ax3.set_title("进步曲线"); ax3.set_xlabel("次数"); ax3.set_ylabel("分数")
        ax3.set_ylim(0, 100); ax3.grid(alpha=0.2)
        canvas.draw_idle()


class SettingsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.resize(430, 260)
        layout = QVBoxLayout(self)
        devices = sd.query_devices()
        self.device_box = QComboBox()
        for idx, device in enumerate(devices):
            if int(device.get("max_input_channels", 0)) > 0:
                self.device_box.addItem(f"{idx}: {device['name']}", idx)
        self.sensitivity = QSlider(Qt.Horizontal)
        self.sensitivity.setRange(1, 100); self.sensitivity.setValue(35)
        self.voice_box = QComboBox()
        self.voice_box.addItems(["系统默认音色", "美式女声优先", "美式男声优先"])
        layout.addWidget(QLabel("录音设备")); layout.addWidget(self.device_box)
        layout.addWidget(QLabel("录音灵敏度")); layout.addWidget(self.sensitivity)
        layout.addWidget(QLabel("TTS音色")); layout.addWidget(self.voice_box)
        close_btn = QPushButton("确定"); close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


# ═══════════════════════════════════════════════════════
# Training pages (left side)
# ═══════════════════════════════════════════════════════

class TrainingPage(QWidget):
    """字母/单词训练页 —— 大字显示 + 提示"""
    def __init__(self, title: str, big_text: str, subtitle: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        self.title = QLabel(title)
        self.title.setObjectName("sectionTitle")
        self.prompt = QLabel(big_text)
        self.prompt.setAlignment(Qt.AlignCenter)
        self.prompt.setObjectName("bigPrompt")
        self.subtitle = QLabel(subtitle)
        self.subtitle.setAlignment(Qt.AlignCenter)
        self.subtitle.setObjectName("hintText")
        self.subtitle.setWordWrap(True)
        layout.addWidget(self.title)
        layout.addWidget(self.prompt)
        layout.addWidget(self.subtitle)


# 自由朗读随机单词库
FREE_WORD_BANK = [
    "apple", "book", "cat", "dog", "egg", "fish", "goat", "hat", "ice",
    "jam", "key", "lion", "map", "owl", "pen", "queen", "rat",
    "sun", "tree", "water",
    "hello", "world", "beautiful", "different", "important",
    "student", "teacher", "family", "friend", "music", "computer",
    "language", "breakfast", "adventure", "chocolate", "elephant",
    "guitar", "hospital", "kitchen", "library", "mountain", "ocean",
    "piano", "rainbow", "sunshine", "telephone", "umbrella", "village",
    "weather", "yesterday", "animal", "basketball", "camera", "diamond",
    "english", "flower", "garden", "holiday", "internet", "journey",
    "knowledge", "morning", "notebook", "orange", "picture", "question",
    "restaurant", "sandwich", "tomorrow", "university", "vacation",
    "window", "afternoon", "birthday", "dictionary", "exercise",
    "favorite", "goodbye", "homework", "island", "jacket",
]


class FreeReadingInput(QWidget):
    """自由朗读输入区 —— 单词输入+查询+随机/顺序浏览+提示"""

    def __init__(self) -> None:
        super().__init__()
        self.word_info_cache: dict | None = None
        self._browse_idx = 0  # 顺序浏览索引

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("自由朗读模式")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        # 输入行
        input_row = QHBoxLayout()
        input_row.setSpacing(6)
        self.word_input = QLineEdit()
        self.word_input.setPlaceholderText("输入要练习的英文单词，如 apple")
        self.word_input.setMinimumHeight(32)
        self.word_input.setStyleSheet(
            "QLineEdit { font-size: 15px; padding: 4px 8px; "
            "border: 2px solid #d7e2ef; border-radius: 8px; "
            "background: white; }"
            "QLineEdit:focus { border-color: #4b8fe8; }")
        input_row.addWidget(self.word_input, stretch=1)
        self.lookup_btn = QPushButton("🔍 查询")
        self.lookup_btn.setMinimumHeight(32)
        self.lookup_btn.setObjectName("accentBtn")
        input_row.addWidget(self.lookup_btn)
        layout.addLayout(input_row)

        # 浏览按钮行：上一个 | 随机 | 下一个 | 顺序/随机切换
        browse_row = QHBoxLayout()
        browse_row.setSpacing(6)
        self.prev_word_btn = QPushButton("⏮")
        self.prev_word_btn.setMinimumHeight(34)
        self.prev_word_btn.setToolTip("上一个单词")
        self.next_word_btn = QPushButton("⏭")
        self.next_word_btn.setMinimumHeight(34)
        self.next_word_btn.setToolTip("下一个单词")
        self.random_btn = QPushButton("🎲 随机")
        self.random_btn.setMinimumHeight(34)
        self.browse_mode_box = QComboBox()
        self.browse_mode_box.addItems(["🎲 随机浏览", "📋 顺序浏览"])
        self.browse_mode_box.setMinimumHeight(34)
        self.browse_label = QLabel("")
        self.browse_label.setAlignment(Qt.AlignCenter)
        self.browse_label.setStyleSheet("font-size: 13px; color: #888;")
        browse_row.addWidget(self.browse_mode_box)
        browse_row.addWidget(self.prev_word_btn)
        browse_row.addWidget(self.browse_label)
        browse_row.addWidget(self.next_word_btn)
        browse_row.addWidget(self.random_btn)
        layout.addLayout(browse_row)

        # 提示标签
        self.hint_label = QLabel("输入单词查询，或使用下方按钮浏览词库")
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setObjectName("freeHint")
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(
            "font-size: 14px; color: #64748b; padding: 8px; "
            "background: #f7fafd; border-radius: 8px;")
        layout.addWidget(self.hint_label)

        layout.addStretch()
        self._update_browse_label()

    def _update_browse_label(self) -> None:
        self.browse_label.setText(f"{self._browse_idx + 1}/{len(FREE_WORD_BANK)}")

    def set_word_info(self, info: dict | None) -> None:
        self.word_info_cache = info
        if info is None:
            self.hint_label.setText("未找到该单词，请检查拼写")
            return
        word = info.get('word', '')
        phonetic = info.get('phonetic', 'N/A')
        cn = info.get('definitions_cn', '')
        if cn:
            self.hint_label.setText(f"🎯 {word}  |  音标: {phonetic}  |  中文: {cn}\n请点击录音按钮朗读此单词")
        else:
            self.hint_label.setText(f"🎯 单词: {word}   音标: {phonetic}\n请点击录音按钮朗读此单词")

    def set_random_word(self) -> str:
        """随机选择一个单词填入输入框并返回"""
        word = random.choice(FREE_WORD_BANK)
        self._browse_idx = FREE_WORD_BANK.index(word)
        self._update_browse_label()
        self.word_input.setText(word)
        return word

    def next_word(self) -> str:
        """浏览下一个单词"""
        self._browse_idx = (self._browse_idx + 1) % len(FREE_WORD_BANK)
        self._update_browse_label()
        word = FREE_WORD_BANK[self._browse_idx]
        self.word_input.setText(word)
        return word

    def prev_word(self) -> str:
        """浏览上一个单词"""
        self._browse_idx = (self._browse_idx - 1) % len(FREE_WORD_BANK)
        self._update_browse_label()
        word = FREE_WORD_BANK[self._browse_idx]
        self.word_input.setText(word)
        return word

    @property
    def browse_sequential(self) -> bool:
        return self.browse_mode_box.currentIndex() == 1

    def browse_next(self) -> str:
        """根据当前模式(随机/顺序)获取下一个单词"""
        if self.browse_sequential:
            return self.next_word()
        else:
            return self.set_random_word()


# ═══════════════════════════════════════════════════════
# Free-mode result panel (right side)
# ═══════════════════════════════════════════════════════

class FreeModeResultPanel(QWidget):
    """自由朗读模式的右侧结果面板 —— 单词详情 + 有道评分 + AI建议"""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── 单词信息卡片 ──
        self.word_card = QGroupBox("📖 单词详情")
        wc_layout = QVBoxLayout(self.word_card)
        wc_layout.setSpacing(6)
        self.phonetic_label = QLabel("音标：--")
        self.phonetic_label.setStyleSheet("font-size: 24px; font-weight: 700; color: #4b8fe8;")
        self.def_label = QLabel("释义：--")
        self.def_label.setWordWrap(True)
        self.def_label.setStyleSheet("font-size: 15px; color: #253043; line-height: 1.6;")
        self.audio_status = QLabel("标准发音：--")
        self.audio_status.setStyleSheet("font-size: 13px; color: #64748b;")
        wc_layout.addWidget(self.phonetic_label)
        wc_layout.addWidget(self.def_label)
        wc_layout.addWidget(self.audio_status)
        layout.addWidget(self.word_card)

        # ── 控制按钮 ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.free_standard_btn = QPushButton("🔊 听标准发音")
        self.free_standard_btn.setEnabled(False)
        self.free_standard_btn.setMinimumHeight(36)
        self.free_slow_check = QCheckBox("慢速")
        btn_row.addWidget(self.free_standard_btn)
        btn_row.addWidget(self.free_slow_check)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── 有道评分卡片 ──
        self.score_card = QGroupBox("📊 有道智云发音评分")
        sc_layout = QGridLayout(self.score_card)
        sc_layout.setSpacing(8)

        self.yd_overall = QLabel("--")
        self.yd_overall.setAlignment(Qt.AlignCenter)
        self.yd_overall.setStyleSheet(
            "font-size: 52px; font-weight: 900; color: #3bb273;")
        self.yd_overall_label = QLabel("综合评分")
        self.yd_overall_label.setAlignment(Qt.AlignCenter)
        self.yd_overall_label.setStyleSheet("font-size: 14px; color: #64748b;")

        self.yd_pron = QLabel("准确度\n--")
        self.yd_pron.setAlignment(Qt.AlignCenter)
        self.yd_pron.setStyleSheet("font-size: 18px; font-weight: 700;")
        self.yd_fluency = QLabel("流利度\n--")
        self.yd_fluency.setAlignment(Qt.AlignCenter)
        self.yd_fluency.setStyleSheet("font-size: 18px; font-weight: 700;")
        self.yd_speed = QLabel("语速\n--")
        self.yd_speed.setAlignment(Qt.AlignCenter)
        self.yd_speed.setStyleSheet("font-size: 18px; font-weight: 700;")

        sc_layout.addWidget(self.yd_overall, 0, 0, 3, 1)
        sc_layout.addWidget(self.yd_overall_label, 3, 0)
        sc_layout.addWidget(self.yd_pron, 0, 1)
        sc_layout.addWidget(self.yd_fluency, 1, 1)
        sc_layout.addWidget(self.yd_speed, 2, 1)

        self.yd_suggestions = QLabel("")
        self.yd_suggestions.setWordWrap(True)
        self.yd_suggestions.setStyleSheet(
            "font-size: 15px; color: #f2a541; padding: 6px; "
            "background: #fff8e8; border-radius: 6px;")
        sc_layout.addWidget(self.yd_suggestions, 4, 0, 1, 2)
        layout.addWidget(self.score_card)

        # ── DeepSeek AI 建议卡片 (stretch=1 撑满右下全部剩余空间) ──
        self.ds_card = QGroupBox("🤖 DeepSeek AI 改进建议")
        ds_layout = QVBoxLayout(self.ds_card)
        ds_layout.setContentsMargins(8, 14, 8, 8)
        self.ds_text = QTextEdit()
        self.ds_text.setReadOnly(True)
        self.ds_text.setPlaceholderText("录音评分完成后，AI 将在此给出详细的发音改进建议...")
        self.ds_text.setStyleSheet(
            "QTextEdit { font-size: 16px; border: 1px solid #e0e7f0; "
            "border-radius: 8px; padding: 14px; background: #fafcfd; "
            "line-height: 1.7; }")
        ds_layout.addWidget(self.ds_text)
        layout.addWidget(self.ds_card, stretch=1)

    # ── 更新方法 ──
    def set_word_info(self, info: dict | None) -> None:
        if info is None:
            self.phonetic_label.setText("音标：未找到")
            self.def_label.setText("释义：请检查单词拼写后重试")
            self.audio_status.setText("标准发音：不可用")
            self.free_standard_btn.setEnabled(False)
            return
        self.phonetic_label.setText(f"音标：{info.get('phonetic', 'N/A')}")
        cn = info.get('definitions_cn', '')
        en_defs = info.get('definitions', [])
        if cn and en_defs:
            self.def_label.setText(f"中文：{cn}\n英文：{'  |  '.join(en_defs)}")
        elif cn:
            self.def_label.setText(f"中文：{cn}")
        elif en_defs:
            self.def_label.setText("英文：" + "  |  ".join(en_defs))
        else:
            self.def_label.setText("释义：N/A")
        audio_url = info.get('audio_url', 'N/A')
        self.audio_status.setText(
            "标准发音：✅ 在线可用" if (audio_url and audio_url != 'N/A')
            else "标准发音：⚠️ 将使用TTS合成")
        self.free_standard_btn.setEnabled(True)

    def set_youdao_result(self, result: dict | None) -> None:
        # Note: set_scale() handles base styles; here we just update the value-dependent colors
        if result is None or result.get("errorCode") != "0":
            self.yd_overall.setText("--")
            self.yd_overall.setStyleSheet(self.yd_overall.styleSheet().replace("#3bb273", "#ccc"))
            self.yd_pron.setText("准确度\n--")
            self.yd_fluency.setText("流利度\n--")
            self.yd_speed.setText("语速\n--")
            self.yd_suggestions.setText("评分失败，请重试")
            return
        overall = result.get("overall", 0)
        pron = result.get("pronunciation", 0)
        fluency = result.get("fluency", 0)
        speed = result.get("speed", 0)
        c = _score_color(overall)
        self.yd_overall.setText(f"{overall:.0f}")
        self.yd_overall.setStyleSheet(f"font-size: {self._s(52)}; font-weight: 900; color: {c};")
        self.yd_pron.setText(f"准确度\n{pron:.0f}/100")
        self.yd_pron.setStyleSheet(f"font-size: {self._s(18)}; font-weight: 700; color: {_score_color(pron)};")
        self.yd_fluency.setText(f"流利度\n{fluency:.0f}/100")
        self.yd_fluency.setStyleSheet(f"font-size: {self._s(18)}; font-weight: 700; color: {_score_color(fluency)};")
        self.yd_speed.setText(f"语速\n{speed:.0f} wpm")
        suggestions = result.get("suggestions", [])
        self.yd_suggestions.setText("  |  ".join(suggestions) if suggestions else "✨ 发音很好！")

    def set_deepseek_advice(self, text: str) -> None:
        self.ds_text.setPlainText(text if text else "（未能获取 AI 建议，请稍后重试）")

    def _s(self, base: int) -> str:
        return f"{int(base * getattr(self, '_panel_scale', 1.0))}px"

    def set_scale(self, sc: float) -> None:
        """按比例缩放面板内所有字体"""
        self._panel_scale = sc
        s = lambda b: self._s(b)
        self.phonetic_label.setStyleSheet(f"font-size: {s(24)}; font-weight: 700; color: #4b8fe8;")
        self.def_label.setStyleSheet(f"font-size: {s(15)}; color: #253043; line-height: 1.6;")
        self.audio_status.setStyleSheet(f"font-size: {s(13)}; color: #64748b;")
        self.yd_overall.setStyleSheet(f"font-size: {s(52)}; font-weight: 900; color: #3bb273;")
        self.yd_overall_label.setStyleSheet(f"font-size: {s(14)}; color: #64748b;")
        self.yd_pron.setStyleSheet(f"font-size: {s(18)}; font-weight: 700;")
        self.yd_fluency.setStyleSheet(f"font-size: {s(18)}; font-weight: 700;")
        self.yd_speed.setStyleSheet(f"font-size: {s(18)}; font-weight: 700;")
        self.yd_suggestions.setStyleSheet(
            f"font-size: {s(15)}; color: #f2a541; padding: {s(6)}; "
            f"background: #fff8e8; border-radius: {s(6)};")
        self.ds_text.setStyleSheet(
            f"QTextEdit {{ font-size: {s(16)}; border: 1px solid #e0e7f0; "
            f"border-radius: {s(8)}; padding: {s(14)}; background: #fafcfd; "
            f"line-height: 1.7; }}")
        self.free_standard_btn.setMinimumHeight(int(36 * sc))