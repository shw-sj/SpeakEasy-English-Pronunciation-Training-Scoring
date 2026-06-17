"""PyQt5 desktop GUI for SpeakEasy pronunciation training."""

from __future__ import annotations

import json
import math
import random
import sys
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QEasingCurve, QPropertyAnimation, QRectF, Qt, QTimer, pyqtProperty
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSizePolicy,
    QSpacerItem,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from audio_feature import compute_mfcc_frames, extract_features
from audio_preprocess import load_audio, preprocess_audio
from config import DEFAULT_WORDS, LETTERS, PROJECT_ROOT, SAMPLE_RATE, TEMPLATES_DIR

ROOT = PROJECT_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import get_word_info, score_pronunciation_youdao, get_deepseek_suggestions
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

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        side = min(self.width(), self.height()) - 16
        rect = QRectF(
            (self.width() - side) / 2, (self.height() - side) / 2, side, side)

        base_pen = QPen(QColor("#e8edf3"), 12)
        base_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(base_pen)
        painter.drawArc(rect, 0, 360 * 16)

        value_pen = QPen(self._color, 12)
        value_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(value_pen)
        painter.drawArc(rect, 90 * 16, -int(360 * 16 * self._value / 100))

        painter.setPen(QColor("#253043"))
        painter.setFont(QFont("Microsoft YaHei", 32, QFont.Bold))
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
            return ROOT / "results" / f"bp_{dataset}_model.pth"
        if model_name == "CNN":
            return ROOT / "results" / f"best_cnn_{dataset}_model.pth"
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


class FreeReadingInput(QWidget):
    """自由朗读输入区 —— 只有单词输入+查询+提示，简洁"""
    def __init__(self) -> None:
        super().__init__()
        self.word_info_cache: dict | None = None

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
        self.word_input.setMinimumHeight(42)
        self.word_input.setStyleSheet(
            "QLineEdit { font-size: 18px; padding: 6px 10px; "
            "border: 2px solid #d7e2ef; border-radius: 8px; "
            "background: white; }"
            "QLineEdit:focus { border-color: #4b8fe8; }")
        input_row.addWidget(self.word_input, stretch=1)
        self.lookup_btn = QPushButton("🔍 查询")
        self.lookup_btn.setMinimumHeight(38)
        self.lookup_btn.setObjectName("accentBtn")
        input_row.addWidget(self.lookup_btn)
        layout.addLayout(input_row)

        # 提示标签
        self.hint_label = QLabel("请输入单词并点击查询，然后录音朗读")
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setObjectName("freeHint")
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(
            "font-size: 17px; color: #64748b; padding: 10px; "
            "background: #f7fafd; border-radius: 8px;")
        layout.addWidget(self.hint_label)

        layout.addStretch()

    def set_word_info(self, info: dict | None) -> None:
        self.word_info_cache = info
        if info is None:
            self.hint_label.setText("未找到该单词，请检查拼写")
            return
        word = info.get('word', '')
        phonetic = info.get('phonetic', 'N/A')
        self.hint_label.setText(f"🎯 单词: {word}   音标: {phonetic}\n请点击录音按钮朗读此单词")


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
        self.phonetic_label.setStyleSheet("font-size: 64px; font-weight: 700; color: #4b8fe8;")
        self.def_label = QLabel("释义：--")
        self.def_label.setWordWrap(True)
        self.def_label.setStyleSheet("font-size: 36px; color: #253043; line-height: 1.6;")
        self.audio_status = QLabel("标准发音：--")
        self.audio_status.setStyleSheet("font-size: 30px; color: #64748b;")
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
            "font-size: 128px; font-weight: 900; color: #3bb273;")
        self.yd_overall_label = QLabel("综合评分")
        self.yd_overall_label.setAlignment(Qt.AlignCenter)
        self.yd_overall_label.setStyleSheet("font-size: 32px; color: #64748b;")

        self.yd_pron = QLabel("准确度\n--")
        self.yd_pron.setAlignment(Qt.AlignCenter)
        self.yd_pron.setStyleSheet("font-size: 40px; font-weight: 700;")
        self.yd_fluency = QLabel("流利度\n--")
        self.yd_fluency.setAlignment(Qt.AlignCenter)
        self.yd_fluency.setStyleSheet("font-size: 40px; font-weight: 700;")
        self.yd_speed = QLabel("语速\n--")
        self.yd_speed.setAlignment(Qt.AlignCenter)
        self.yd_speed.setStyleSheet("font-size: 40px; font-weight: 700;")

        sc_layout.addWidget(self.yd_overall, 0, 0, 3, 1)
        sc_layout.addWidget(self.yd_overall_label, 3, 0)
        sc_layout.addWidget(self.yd_pron, 0, 1)
        sc_layout.addWidget(self.yd_fluency, 1, 1)
        sc_layout.addWidget(self.yd_speed, 2, 1)

        self.yd_suggestions = QLabel("")
        self.yd_suggestions.setWordWrap(True)
        self.yd_suggestions.setStyleSheet(
            "font-size: 34px; color: #f2a541; padding: 6px; "
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
            "QTextEdit { font-size: 48px; border: 1px solid #e0e7f0; "
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
        defs = info.get('definitions', [])
        self.def_label.setText("释义：" + "  |  ".join(defs) if defs else "释义：N/A")
        audio_url = info.get('audio_url', 'N/A')
        self.audio_status.setText(
            "标准发音：✅ 在线可用" if (audio_url and audio_url != 'N/A')
            else "标准发音：⚠️ 将使用TTS合成")
        self.free_standard_btn.setEnabled(True)

    def set_youdao_result(self, result: dict | None) -> None:
        if result is None or result.get("errorCode") != "0":
            self.yd_overall.setText("--")
            self.yd_overall.setStyleSheet("font-size: 64px; font-weight: 900; color: #ccc;")
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
        self.yd_overall.setStyleSheet(f"font-size: 128px; font-weight: 900; color: {c};")
        self.yd_pron.setText(f"准确度\n{pron:.0f}/100")
        self.yd_pron.setStyleSheet(f"font-size: 40px; font-weight: 700; color: {_score_color(pron)};")
        self.yd_fluency.setText(f"流利度\n{fluency:.0f}/100")
        self.yd_fluency.setStyleSheet(f"font-size: 40px; font-weight: 700; color: {_score_color(fluency)};")
        self.yd_speed.setText(f"语速\n{speed:.0f} wpm")
        suggestions = result.get("suggestions", [])
        self.yd_suggestions.setText("  |  ".join(suggestions) if suggestions else "✨ 发音很好！")

    def set_deepseek_advice(self, text: str) -> None:
        self.ds_text.setPlainText(text if text else "（未能获取 AI 建议，请稍后重试）")


# ═══════════════════════════════════════════════════════
# Main Window
# ═══════════════════════════════════════════════════════

class SpeakEasyWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SpeakEasy · 英语口语发音训练系统")
        self.resize(1450, 920)

        self.model_manager = ModelManager()
        self.standard = StandardSpeech()
        self.history: list[dict] = []
        self.current_audio = np.array([], dtype=np.float32)
        self.current_standard: np.ndarray | None = None
        self.recording = False
        self.frames: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None
        self.current_letter = random.choice(LETTERS)
        self.current_letter_idx = LETTERS.index(self.current_letter)
        self.current_word = random.choice(DEFAULT_WORDS)
        self.current_word_idx = DEFAULT_WORDS.index(self.current_word)
        self.order_mode = "random"  # "random" or "sequential"
        self.nav_stack: list[dict] = []  # 导航历史：上一步状态

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_live_wave)

        self.build_ui()
        self.apply_style()
        self.update_prompt()

    # ── build_ui ──────────────────────────────────────

    def build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(20, 14, 20, 18)
        outer.setSpacing(12)

        # ---- 顶部标题栏 ----
        header = QLabel("🎓 SpeakEasy · 英语口语发音训练系统")
        header.setObjectName("appTitle")
        header.setAlignment(Qt.AlignCenter)
        outer.addWidget(header)

        # ---- 主体：左栏 + 右栏 ----
        body = QHBoxLayout()
        body.setSpacing(16)
        outer.addLayout(body, stretch=1)

        # ============ 左边栏 (440px) ============
        self.left_frame = QFrame()
        self.left_frame.setObjectName("sidePanel")
        self.left_frame.setMinimumWidth(460)
        left_layout = QVBoxLayout(self.left_frame)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)

        # -- 模式标签 --
        self.tabs = QTabWidget(self.left_frame)
        self._tab_letter = QWidget(self.tabs)
        self._tab_word = QWidget(self.tabs)
        self._tab_free = QWidget(self.tabs)
        self.tabs.addTab(self._tab_letter, "🔤 字母训练")
        self.tabs.addTab(self._tab_word, "📝 单词训练")
        self.tabs.addTab(self._tab_free, "🎯 自由朗读")
        self.tabs.currentChanged.connect(self.update_prompt)
        left_layout.addWidget(self.tabs)

        # -- 题目/输入页栈 --
        self.page_stack = QStackedWidget()
        self.letter_page = TrainingPage("随机出题", self.current_letter,
                                        "请朗读屏幕中的大写字母")
        self.word_page = TrainingPage("单词卡片", self.current_word,
                                      self.word_subtitle(self.current_word))
        self.free_input = FreeReadingInput()
        self.page_stack.addWidget(self.letter_page)
        self.page_stack.addWidget(self.word_page)
        self.page_stack.addWidget(self.free_input)
        left_layout.addWidget(self.page_stack)

        # -- 模型选择（字母/单词模式专用） --
        self.model_section = QWidget()
        model_row = QHBoxLayout(self.model_section)
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.setSpacing(8)
        model_row.addWidget(QLabel("识别模型"))
        self.model_box = QComboBox()
        self.model_box.addItems(["BP", "CNN", "融合模型"])
        self.model_box.setMinimumHeight(32)
        model_row.addWidget(self.model_box)
        left_layout.addWidget(self.model_section)

        # -- 录音按钮（大） --
        self.record_btn = QPushButton("🎤  开始录音")
        self.record_btn.setObjectName("recordBtn")
        self.record_btn.setMinimumHeight(48)
        self.record_btn.clicked.connect(self.toggle_recording)
        left_layout.addWidget(self.record_btn)

        # -- 播放控制行 --
        play_row = QHBoxLayout()
        play_row.setSpacing(6)
        self.listen_btn = QPushButton("🔊 听标准")
        self.listen_btn.clicked.connect(self.play_standard)
        self.self_listen_btn = QPushButton("🎙️ 听自己")
        self.self_listen_btn.clicked.connect(self.play_self_recording)
        self.slow_check = QCheckBox("慢速")
        play_row.addWidget(self.listen_btn)
        play_row.addWidget(self.self_listen_btn)
        play_row.addWidget(self.slow_check)
        play_row.addStretch()
        left_layout.addLayout(play_row)

        # -- 导航行（字母/单词模式专用） --
        self.nav_section = QWidget()
        nav_row = QHBoxLayout(self.nav_section)
        nav_row.setContentsMargins(0, 0, 0, 0)
        nav_row.setSpacing(6)
        self.order_box = QComboBox()
        self.order_box.addItems(["🎲 随机出题", "📋 顺序出题"])
        self.order_box.setMinimumHeight(32)
        self.order_box.currentIndexChanged.connect(self.on_order_changed)
        self.prev_btn = QPushButton("⏮ 上一个")
        self.prev_btn.clicked.connect(self.prev_prompt)
        self.prev_btn.setEnabled(False)
        self.next_btn = QPushButton("下一个 ⏭")
        self.next_btn.clicked.connect(self.next_prompt)
        self.stats_btn = QPushButton("📊 学习统计")
        self.stats_btn.clicked.connect(self.show_stats)
        self.settings_btn = QPushButton("⚙ 设置")
        self.settings_btn.clicked.connect(self.show_settings)
        nav_row.addWidget(self.order_box)
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.next_btn)
        nav_row.addWidget(self.stats_btn)
        nav_row.addWidget(self.settings_btn)
        left_layout.addWidget(self.nav_section)

        # -- 识别结果（字母/单词模式专用） --
        self.result_box = QGroupBox("📋 识别结果")
        result_layout = QGridLayout(self.result_box)
        result_layout.setSpacing(6)
        self.pred_label = QLabel("--")
        self.pred_label.setObjectName("resultText")
        self.conf_label = QLabel("置信度：--")
        self.feedback_label = QLabel("等待录音")
        self.feedback_label.setObjectName("feedbackText")
        self.feedback_label.setWordWrap(True)
        self.score_widget = CircularScoreWidget(150)
        result_layout.addWidget(self.pred_label, 0, 0, 1, 2)
        result_layout.addWidget(self.score_widget, 1, 0, 2, 1)
        result_layout.addWidget(self.conf_label, 1, 1)
        result_layout.addWidget(self.feedback_label, 2, 1)
        left_layout.addWidget(self.result_box)

        # -- 历史记录 --
        history_box = QGroupBox("📜 最近10次练习")
        history_layout = QVBoxLayout(history_box)
        history_layout.setContentsMargins(6, 10, 6, 6)
        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(80)
        history_layout.addWidget(self.history_list)
        left_layout.addWidget(history_box, stretch=1)

        # ============ 右边栏 (自适应) ============
        self.right_frame = QFrame()
        self.right_frame.setObjectName("rightPanel")
        right_layout = QVBoxLayout(self.right_frame)
        right_layout.setContentsMargins(14, 14, 14, 14)

        self.right_stack = QStackedWidget()
        self.chart_panel = ChartPanel()
        self.free_result_panel = FreeModeResultPanel()
        self.right_stack.addWidget(self.chart_panel)       # index 0
        self.right_stack.addWidget(self.free_result_panel)  # index 1
        right_layout.addWidget(self.right_stack)
        body.addWidget(self.left_frame, stretch=1)
        body.addWidget(self.right_frame, stretch=2)

        # -- 信号连接 --
        self.free_input.lookup_btn.clicked.connect(self.do_word_lookup)
        self.free_input.word_input.returnPressed.connect(self.do_word_lookup)
        self.free_result_panel.free_standard_btn.clicked.connect(self.play_free_standard)

    # ── apply_style ───────────────────────────────────

    def apply_style(self) -> None:
        self.setStyleSheet("""
        /* 全局 */
        QMainWindow { background: #f0f4f8; }
        QLabel#appTitle {
            color: #1a2332; font-size: 34px; font-weight: 800;
            padding: 8px 0; letter-spacing: 2px;
        }

        /* 面板 */
        QFrame#sidePanel {
            background: #ffffff; border: 1px solid #e2e8f0;
            border-radius: 12px;
        }
        QFrame#rightPanel {
            background: #ffffff; border: 1px solid #e2e8f0;
            border-radius: 12px;
        }

        /* 标签页 */
        QTabWidget::pane { border: 0; }
        QTabBar::tab {
            background: #f1f5f9; color: #475569; border-radius: 8px;
            padding: 12px 20px; margin-right: 5px; font-size: 18px; font-weight: 600;
        }
        QTabBar::tab:selected { background: #4b8fe8; color: white; }

        /* 标题 */
        QLabel#sectionTitle {
            font-size: 24px; font-weight: 700; color: #1a2332;
        }
        QLabel#bigPrompt {
            font-size: 110px; font-weight: 800; color: #4b8fe8;
            background: #eef5ff; border-radius: 10px; padding: 20px;
        }
        QLabel#hintText {
            font-size: 19px; color: #64748b; line-height: 1.5;
        }

        /* 通用按钮 */
        QPushButton {
            background: #f1f5f9; color: #334155; border: 1px solid #e2e8f0;
            border-radius: 8px; padding: 10px 14px; font-size: 16px; font-weight: 600;
        }
        QPushButton:hover { background: #e2e8f0; }

        /* 录音按钮 */
        QPushButton#recordBtn {
            background: #3bb273; color: white; border: 0;
            font-size: 24px; font-weight: 700; border-radius: 12px;
        }
        QPushButton#recordBtn:hover { background: #319762; }

        /* 强调按钮 */
        QPushButton#accentBtn {
            background: #4b8fe8; color: white; border: 0;
        }
        QPushButton#accentBtn:hover { background: #3a7bd5; }

        /* 下拉框 */
        QComboBox {
            background: white; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 7px 12px; font-size: 16px;
        }
        QComboBox:hover { border-color: #4b8fe8; }

        /* 分组框 */
        QGroupBox {
            font-size: 18px; font-weight: 700; color: #1a2332;
            border: 1px solid #e2e8f0; border-radius: 10px;
            margin-top: 8px; padding: 12px 10px 10px 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 14px; padding: 0 6px;
            color: #4b8fe8;
        }

        /* 结果 */
        QLabel#resultText {
            font-size: 42px; font-weight: 800; color: #1a2332;
        }
        QLabel#feedbackText {
            font-size: 24px; font-weight: 700; color: #1a2332;
        }

        /* 列表 */
        QListWidget {
            border: 0; background: #f8fafc; border-radius: 8px;
            padding: 4px; font-size: 15px;
        }

        /* ScrollArea */
        QScrollArea { border: 0; background: transparent; }
        QScrollBar:vertical {
            background: transparent; width: 6px;
        }
        QScrollBar::handle:vertical {
            background: #cbd5e1; border-radius: 3px; min-height: 20px;
        }
        QScrollBar::handle:vertical:hover { background: #94a3b8; }
        """)

    # ── mode / dataset / target ────────────────────────

    def mode(self) -> str:
        return ["letters", "words", "free"][self.tabs.currentIndex()]

    def dataset_for_mode(self) -> str:
        return "words" if self.mode() == "words" else "letters"

    def target_text(self) -> str | None:
        m = self.mode()
        if m == "letters": return self.current_letter
        if m == "words": return self.current_word
        return None

    def word_subtitle(self, word: str) -> str:
        return f"{CHINESE_MEANING.get(word, '基础词汇')}  |  请朗读这个单词"

    # ── update_prompt / next_prompt ────────────────────

    def update_prompt(self) -> None:
        idx = self.tabs.currentIndex()
        self.page_stack.setCurrentIndex(idx)
        self.letter_page.prompt.setText(self.current_letter)
        self.word_page.prompt.setText(self.current_word)
        self.word_page.subtitle.setText(self.word_subtitle(self.current_word))

        # 自由朗读模式：隐藏模型选择/导航/识别结果（右边看结果即可）
        is_free = idx == 2
        self.model_section.setVisible(not is_free)
        self.nav_section.setVisible(not is_free)
        self.result_box.setVisible(not is_free)
        self.next_btn.setEnabled(not is_free)
        self.prev_btn.setEnabled(not is_free and len(self.nav_stack) > 0)
        if is_free:
            self.nav_stack.clear()

        # 切换右侧面板
        if is_free:
            self.right_stack.setCurrentIndex(1)  # FreeModeResultPanel
            word = self.free_input.word_input.text().strip()
            if word and self.free_input.word_info_cache:
                info = self.free_input.word_info_cache
                self.free_input.hint_label.setText(
                    f"🎯 单词: {info.get('word', word)}   音标: {info.get('phonetic', 'N/A')}\n"
                    f"请点击录音按钮朗读此单词")
            else:
                self.free_input.hint_label.setText(
                    "请输入单词并点击查询，然后录音朗读")
        else:
            self.right_stack.setCurrentIndex(0)  # ChartPanel

    def on_order_changed(self, idx: int) -> None:
        self.order_mode = "random" if idx == 0 else "sequential"

    def _push_nav_state(self) -> None:
        """保存当前状态到导航历史栈"""
        m = self.mode()
        state = {"mode": m}
        if m == "letters":
            state["letter"] = self.current_letter
            state["letter_idx"] = self.current_letter_idx
        elif m == "words":
            state["word"] = self.current_word
            state["word_idx"] = self.current_word_idx
        self.nav_stack.append(state)

    def _restore_nav_state(self, state: dict) -> None:
        """从导航历史栈恢复状态"""
        m = state["mode"]
        if m == "letters":
            self.current_letter = state["letter"]
            self.current_letter_idx = state["letter_idx"]
        elif m == "words":
            self.current_word = state["word"]
            self.current_word_idx = state["word_idx"]

    def prev_prompt(self) -> None:
        if not self.nav_stack:
            return
        state = self.nav_stack.pop()
        self._restore_nav_state(state)
        self.prev_btn.setEnabled(len(self.nav_stack) > 0)
        self.update_prompt()

    def next_prompt(self) -> None:
        self._push_nav_state()
        if self.mode() == "letters":
            if self.order_mode == "sequential":
                self.current_letter_idx = (self.current_letter_idx + 1) % len(LETTERS)
                self.current_letter = LETTERS[self.current_letter_idx]
            else:
                self.current_letter = random.choice(LETTERS)
                self.current_letter_idx = LETTERS.index(self.current_letter)
        elif self.mode() == "words":
            if self.order_mode == "sequential":
                self.current_word_idx = (self.current_word_idx + 1) % len(DEFAULT_WORDS)
                self.current_word = DEFAULT_WORDS[self.current_word_idx]
            else:
                self.current_word = random.choice(DEFAULT_WORDS)
                self.current_word_idx = DEFAULT_WORDS.index(self.current_word)
        self.prev_btn.setEnabled(len(self.nav_stack) > 0)
        self.update_prompt()

    # ── recording ─────────────────────────────────────

    def toggle_recording(self) -> None:
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        self.frames = []
        self.recording = True
        self.record_btn.setText("⏹  停止录音")
        self.record_btn.setStyleSheet(
            "QPushButton#recordBtn { background: #e85d5d; color: white; "
            "font-size: 20px; font-weight: 700; border-radius: 12px; }")
        self.feedback_label.setText("🔴 正在录音...")
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            callback=self.audio_callback)
        self.stream.start()
        self.timer.start(80)

    def audio_callback(self, indata, frames, time_info, status) -> None:
        del frames, time_info, status
        self.frames.append(indata[:, 0].copy())

    def stop_recording(self) -> None:
        self.recording = False
        self.timer.stop()
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.record_btn.setText("🎤  开始录音")
        self.record_btn.setStyleSheet(
            "QPushButton#recordBtn { background: #3bb273; color: white; "
            "font-size: 20px; font-weight: 700; border-radius: 12px; }")
        if not self.frames:
            self.feedback_label.setText("没有录到声音")
            return
        self.current_audio = np.concatenate(self.frames).astype(np.float32)
        self.run_prediction()

    def refresh_live_wave(self) -> None:
        if not self.frames:
            return
        audio = np.concatenate(self.frames[-20:]).astype(np.float32)
        self.chart_panel.update_live_wave(audio)

    # ── prediction ────────────────────────────────────

    def run_prediction(self) -> None:
        current_mode = self.mode()
        if current_mode == "free":
            self.run_free_mode_scoring()
            return

        # 字母/单词模式
        target = self.target_text()
        dataset = self.dataset_for_mode()
        model_name = self.model_box.currentText()
        try:
            feature = extract_features(self.current_audio, SAMPLE_RATE)
            result = self.model_manager.predict(model_name, dataset, feature)
        except Exception as exc:
            self.pred_label.setText("未识别")
            self.conf_label.setText("置信度：--")
            self.feedback_label.setText(str(exc))
            self.score_widget.animate_to(0)
            self.chart_panel.update_live_wave(self.current_audio)
            return

        pred = display_label(result.label)
        target_for_score = target or pred
        labels = [display_label(l) for l in result.labels]
        score = result.score
        if target:
            matched_index = None
            for i, l in enumerate(labels):
                if l.lower() == target.lower():
                    matched_index = i
                    break
            score = int(round(float(result.probabilities[matched_index]) * 100)) \
                if matched_index is not None else 0

        feedback, stars, color = score_text(score)
        self.pred_label.setText(f"{pred}")
        self.conf_label.setText(f"置信度：{result.confidence * 100:.1f}%")
        self.feedback_label.setText(f"{stars}  {feedback}")
        self.feedback_label.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: 700;")
        self.score_widget.animate_to(score)

        standard_audio = None
        try:
            standard_audio = self.standard.load(target_for_score, self.slow_check.isChecked())
        except Exception:
            standard_audio = None
        self.current_standard = standard_audio
        self.chart_panel.update_result(self.current_audio, result.labels,
                                       result.probabilities, target_for_score,
                                       standard_audio)
        self.add_history(target_for_score, pred, score)

    def run_free_mode_scoring(self) -> None:
        """自由朗读：有道评分 + DeepSeek AI 建议"""
        word = self.free_input.word_input.text().strip().lower()
        if not word:
            self.feedback_label.setText("请先在左侧输入要练习的单词")
            return

        self.feedback_label.setText("⏳ 正在有道智云评分...")
        QApplication.processEvents()

        try:
            youdao_result = score_pronunciation_youdao(
                self.current_audio, word, SAMPLE_RATE)
            self.free_result_panel.set_youdao_result(youdao_result)

            if youdao_result.get("errorCode") == "0":
                overall = youdao_result.get("overall", 0)
                pron = youdao_result.get("pronunciation", 0)
                fluency = youdao_result.get("fluency", 0)

                self.pred_label.setText(f"{word}")
                self.conf_label.setText("有道智云评测")
                fb, stars, color = score_text(int(overall))
                self.feedback_label.setText(
                    f"{stars}  综合{overall:.0f} | 准确度{pron:.0f} | 流利度{fluency:.0f}")
                self.feedback_label.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: 700;")
                self.score_widget.animate_to(int(overall))
                self.add_history(word, word, int(overall))

                # DeepSeek AI
                self.feedback_label.setText("🤖 正在获取 AI 智能建议...")
                QApplication.processEvents()
                info = self.free_input.word_info_cache
                phonetic = info.get('phonetic', 'N/A') if info else 'N/A'
                definitions = info.get('definitions', []) if info else []
                advice = get_deepseek_suggestions(word, phonetic, definitions, youdao_result)
                self.free_result_panel.set_deepseek_advice(advice)
                self.feedback_label.setText(f"{stars}  有道{overall:.0f}分 | AI建议已生成 ✓")
                self.feedback_label.setStyleSheet(
                    f"color: {color}; font-size: 24px; font-weight: 700;")
            else:
                self.pred_label.setText(f"{word}")
                self.conf_label.setText("评分失败")
                self.feedback_label.setText(
                    f"有道API错误: {youdao_result.get('errorCode', '未知')}")
                self.score_widget.animate_to(0)
                self.free_result_panel.set_deepseek_advice("")
        except Exception as exc:
            self.pred_label.setText("出错")
            self.feedback_label.setText(f"评分请求失败：{exc}")
            self.score_widget.animate_to(0)
            self.free_result_panel.set_youdao_result(None)
            self.free_result_panel.set_deepseek_advice("")

        self.chart_panel.update_live_wave(self.current_audio)

    # ── history ───────────────────────────────────────

    def add_history(self, target: str, predicted: str, score: int) -> None:
        item = {
            "target": target, "predicted": predicted,
            "score": score, "time": time.strftime("%H:%M:%S"),
        }
        self.history.insert(0, item)
        self.history = self.history[:10]
        self.history_list.clear()
        for row in self.history:
            text = f"{row['time']}  {row['target']} → {row['predicted']}  {row['score']}分"
            self.history_list.addItem(QListWidgetItem(text))

    # ── playback ──────────────────────────────────────

    def play_standard(self) -> None:
        text = self.target_text()
        if text is None:
            text = self.pred_label.text().strip()
        if not text or text == "--":
            QMessageBox.information(self, "提示", "请先选择题目或完成一次识别。")
            return
        try:
            self.current_standard = self.standard.play(text, self.slow_check.isChecked())
        except Exception as exc:
            QMessageBox.warning(self, "标准发音", f"播放失败：{exc}")

    def play_self_recording(self) -> None:
        """播放用户自己的录音"""
        if self.current_audio.size == 0:
            QMessageBox.information(self, "提示", "还没有录音，请先点击录音按钮。")
            return
        try:
            sd.stop()
            sd.play(self.current_audio, SAMPLE_RATE)
            self.feedback_label.setText("🔊 正在播放您的录音...")
        except Exception as exc:
            QMessageBox.warning(self, "播放录音", f"播放失败：{exc}")

    def play_free_standard(self) -> None:
        """自由朗读模式播放标准发音"""
        word = self.free_input.word_input.text().strip().lower()
        if not word:
            QMessageBox.information(self, "提示", "请先输入要练习的单词。")
            return
        info = self.free_input.word_info_cache
        if info and info.get('audio_url') and info['audio_url'] != 'N/A':
            try:
                import urllib.request
                audio_url = info['audio_url']
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                    urllib.request.urlretrieve(audio_url, tmp.name)
                    audio, sr = load_audio(tmp.name, SAMPLE_RATE)
                    sd.play(audio, SAMPLE_RATE)
                    self.current_standard = audio
                    Path(tmp.name).unlink(missing_ok=True)
                    return
            except Exception:
                pass
        try:
            self.current_standard = self.standard.play(
                word, self.free_result_panel.free_slow_check.isChecked())
        except Exception as exc:
            QMessageBox.warning(self, "标准发音", f"播放失败：{exc}")

    # ── word lookup ───────────────────────────────────

    def do_word_lookup(self) -> None:
        """查询单词 - 结果同时更新左侧提示和右侧面板"""
        word = self.free_input.word_input.text().strip().lower()
        if not word:
            QMessageBox.information(self, "提示", "请输入要练习的英文单词。")
            return

        self.free_input.set_word_info(None)
        self.free_result_panel.set_word_info(None)
        self.free_input.lookup_btn.setText("⏳")
        self.free_input.lookup_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            info = get_word_info(word)
            if info is None:
                QMessageBox.warning(self, "未找到",
                                    f"未能找到单词 \"{word}\" 的信息。\n请检查拼写后重试。")
                self.free_input.set_word_info(None)
                self.free_result_panel.set_word_info(None)
            else:
                self.free_input.set_word_info(info)
                self.free_result_panel.set_word_info(info)
        except Exception as exc:
            QMessageBox.warning(self, "查询失败", f"网络请求出错：{exc}")
        finally:
            self.free_input.lookup_btn.setText("🔍 查询")
            self.free_input.lookup_btn.setEnabled(True)

    # ── dialogs ───────────────────────────────────────

    def show_stats(self) -> None:
        StatsDialog(self.history, self).exec_()

    def show_settings(self) -> None:
        SettingsDialog(self).exec_()

    def closeEvent(self, event) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        sd.stop()
        event.accept()


# ═══════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════

def load_json_config() -> dict:
    path = ROOT / "config.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    load_json_config()
    app = QApplication(sys.argv)
    window = SpeakEasyWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
