"""PyQt5 desktop GUI for SpeakEasy pronunciation training."""

from __future__ import annotations

import json
import math
import random
import sys
import time
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
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from audio_feature import compute_mfcc_frames, extract_features
from audio_preprocess import load_audio, preprocess_audio
from config import DEFAULT_WORDS, LETTERS, PROJECT_ROOT, SAMPLE_RATE, TEMPLATES_DIR

ROOT = PROJECT_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bp_network import BPNetwork
from cnn_network import CNN1D

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]
matplotlib.rcParams["axes.unicode_minus"] = False


CHINESE_MEANING = {
    "apple": "苹果",
    "book": "书",
    "cat": "猫",
    "dog": "狗",
    "egg": "鸡蛋",
    "fish": "鱼",
    "goat": "山羊",
    "hat": "帽子",
    "ice": "冰",
    "jam": "果酱",
    "key": "钥匙",
    "lion": "狮子",
    "map": "地图",
    "net": "网",
    "owl": "猫头鹰",
    "pen": "钢笔",
    "queen": "女王",
    "rat": "老鼠",
    "sun": "太阳",
    "tree": "树",
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


class CircularScoreWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._value = 0
        self._color = QColor("#e85d5d")
        self.setMinimumSize(150, 150)

    def get_value(self) -> int:
        return self._value

    def set_value(self, value: int) -> None:
        self._value = max(0, min(100, int(value)))
        _, _, color = score_text(self._value)
        self._color = QColor(color)
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

        side = min(self.width(), self.height()) - 20
        rect = QRectF(
            (self.width() - side) / 2,
            (self.height() - side) / 2,
            side,
            side,
        )

        base_pen = QPen(QColor("#e8edf3"), 14)
        base_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(base_pen)
        painter.drawArc(rect, 0, 360 * 16)

        value_pen = QPen(self._color, 14)
        value_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(value_pen)
        painter.drawArc(rect, 90 * 16, -int(360 * 16 * self._value / 100))

        painter.setPen(QColor("#253043"))
        painter.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
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

    def update_result(
        self,
        audio: np.ndarray,
        labels: list[str],
        probabilities: np.ndarray,
        target: str | None,
        standard_audio: np.ndarray | None,
    ) -> None:
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
            ax_cmp.plot(ts, standard_audio + offset, color="#3bb273", linewidth=1.0, label="标准")
        ax_cmp.legend(loc="upper right")
        ax_cmp.grid(alpha=0.2)

        self.figure.tight_layout()
        self.canvas.draw_idle()


class ModelManager:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache: dict[tuple[str, str], tuple[torch.nn.Module, list[str], np.ndarray | None, np.ndarray | None]] = {}

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
            model = BPNetwork(
                input_size=input_dim,
                output_size=output_dim,
                dropout_rate=float(data.get("dropout_rate", 0.0)),
                task=str(data.get("task", dataset)),
            ).to(self.device)
        else:
            model = CNN1D(
                input_dim=input_dim,
                num_classes=output_dim,
                dropout_rate=float(data.get("dropout_rate", 0.3)),
            ).to(self.device)

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
            return PredictionResult(bp.labels[idx], float(probs[idx]), int(round(probs[idx] * 100)), bp.labels, probs)
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
        confidence = float(probs[idx])
        return PredictionResult(labels[idx], confidence, int(round(confidence * 100)), labels, probs)


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
    def __init__(self, history: list[dict], parent: QWidget | None = None) -> None:
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
            ax1.axis("off")
            ax2.axis("off")
            ax3.axis("off")
            canvas.draw_idle()
            return

        labels = sorted({h["target"] for h in history})
        means = [np.mean([h["score"] for h in history if h["target"] == label]) for label in labels]
        counts = [sum(1 for h in history if h["target"] == label) for label in labels]
        scores = [h["score"] for h in history]

        ax1.bar(labels, means, color="#4b8fe8")
        ax1.set_title("平均分")
        ax1.tick_params(axis="x", labelrotation=90, labelsize=8)
        ax1.set_ylim(0, 100)

        ax2.bar(labels, counts, color="#3bb273")
        ax2.set_title("练习次数")
        ax2.tick_params(axis="x", labelrotation=90, labelsize=8)

        ax3.plot(range(1, len(scores) + 1), scores, marker="o", color="#f2a541")
        ax3.set_title("进步曲线")
        ax3.set_xlabel("次数")
        ax3.set_ylabel("分数")
        ax3.set_ylim(0, 100)
        ax3.grid(alpha=0.2)
        canvas.draw_idle()


class SettingsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
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
        self.sensitivity.setRange(1, 100)
        self.sensitivity.setValue(35)

        self.voice_box = QComboBox()
        self.voice_box.addItems(["系统默认音色", "美式女声优先", "美式男声优先"])

        layout.addWidget(QLabel("录音设备"))
        layout.addWidget(self.device_box)
        layout.addWidget(QLabel("录音灵敏度"))
        layout.addWidget(self.sensitivity)
        layout.addWidget(QLabel("TTS音色"))
        layout.addWidget(self.voice_box)
        close_btn = QPushButton("确定")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


class TrainingPage(QWidget):
    def __init__(self, title: str, big_text: str, subtitle: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        self.title = QLabel(title)
        self.title.setObjectName("sectionTitle")
        self.prompt = QLabel(big_text)
        self.prompt.setAlignment(Qt.AlignCenter)
        self.prompt.setObjectName("bigPrompt")
        self.subtitle = QLabel(subtitle)
        self.subtitle.setAlignment(Qt.AlignCenter)
        self.subtitle.setObjectName("hintText")
        layout.addWidget(self.title)
        layout.addWidget(self.prompt)
        layout.addWidget(self.subtitle)


class SpeakEasyWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("英语口语发音训练系统")
        self.resize(1280, 820)

        self.model_manager = ModelManager()
        self.standard = StandardSpeech()
        self.history: list[dict] = []
        self.current_audio = np.array([], dtype=np.float32)
        self.current_standard: np.ndarray | None = None
        self.recording = False
        self.frames: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None
        self.current_letter = random.choice(LETTERS)
        self.current_word = random.choice(DEFAULT_WORDS)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_live_wave)

        self.build_ui()
        self.apply_style()
        self.update_prompt()

    def build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(18, 14, 18, 18)
        outer.setSpacing(14)

        header = QLabel("英语口语发音训练系统")
        header.setObjectName("appTitle")
        header.setAlignment(Qt.AlignCenter)
        outer.addWidget(header)

        body = QHBoxLayout()
        body.setSpacing(16)
        outer.addLayout(body, stretch=1)

        left = QFrame()
        left.setObjectName("sidePanel")
        left.setFixedWidth(390)
        side = QVBoxLayout(left)
        side.setSpacing(12)
        body.addWidget(left)

        self.tabs = QTabWidget()
        self.tabs.addTab(QWidget(), "字母训练")
        self.tabs.addTab(QWidget(), "单词训练")
        self.tabs.addTab(QWidget(), "自由朗读")
        self.tabs.currentChanged.connect(self.update_prompt)
        side.addWidget(self.tabs)

        self.page_stack = QStackedWidget()
        self.letter_page = TrainingPage("随机出题", self.current_letter, "请朗读屏幕中的大写字母")
        self.word_page = TrainingPage("单词卡片", self.current_word, self.word_subtitle(self.current_word))
        self.free_page = TrainingPage("自由朗读", "ABC", "朗读任意字母或单词，系统会识别结果")
        self.page_stack.addWidget(self.letter_page)
        self.page_stack.addWidget(self.word_page)
        self.page_stack.addWidget(self.free_page)
        side.addWidget(self.page_stack)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("模型"))
        self.model_box = QComboBox()
        self.model_box.addItems(["BP", "CNN", "融合模型"])
        model_row.addWidget(self.model_box)
        side.addLayout(model_row)

        self.record_btn = QPushButton("开始录音")
        self.record_btn.setObjectName("primaryButton")
        self.record_btn.clicked.connect(self.toggle_recording)
        side.addWidget(self.record_btn)

        button_row = QHBoxLayout()
        self.listen_btn = QPushButton("听标准发音")
        self.listen_btn.clicked.connect(self.play_standard)
        self.slow_check = QCheckBox("慢速")
        button_row.addWidget(self.listen_btn)
        button_row.addWidget(self.slow_check)
        side.addLayout(button_row)

        next_row = QHBoxLayout()
        self.next_btn = QPushButton("下一个")
        self.next_btn.clicked.connect(self.next_prompt)
        self.stats_btn = QPushButton("学习统计")
        self.stats_btn.clicked.connect(self.show_stats)
        self.settings_btn = QPushButton("设置")
        self.settings_btn.clicked.connect(self.show_settings)
        next_row.addWidget(self.next_btn)
        next_row.addWidget(self.stats_btn)
        next_row.addWidget(self.settings_btn)
        side.addLayout(next_row)

        result_box = QGroupBox("识别结果")
        result_layout = QGridLayout(result_box)
        self.pred_label = QLabel("--")
        self.pred_label.setObjectName("resultText")
        self.conf_label = QLabel("置信度：--")
        self.feedback_label = QLabel("等待录音")
        self.feedback_label.setObjectName("feedbackText")
        self.score_widget = CircularScoreWidget()
        result_layout.addWidget(self.pred_label, 0, 0, 1, 2)
        result_layout.addWidget(self.score_widget, 1, 0, 2, 1)
        result_layout.addWidget(self.conf_label, 1, 1)
        result_layout.addWidget(self.feedback_label, 2, 1)
        side.addWidget(result_box)

        history_box = QGroupBox("最近10次练习")
        history_layout = QVBoxLayout(history_box)
        self.history_list = QListWidget()
        history_layout.addWidget(self.history_list)
        side.addWidget(history_box, stretch=1)

        right = QFrame()
        right.setObjectName("chartPanel")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(12, 12, 12, 12)
        self.chart_panel = ChartPanel()
        right_layout.addWidget(self.chart_panel)
        body.addWidget(right, stretch=1)

    def apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #f5f8fc; }
            QLabel#appTitle {
                color: #253043; font-size: 30px; font-weight: 700;
                padding: 10px 0;
            }
            QFrame#sidePanel, QFrame#chartPanel {
                background: white; border: 1px solid #dfe7f1; border-radius: 8px;
            }
            QTabWidget::pane { border: 0; }
            QTabBar::tab {
                background: #e8edf3; color: #253043; border-radius: 8px;
                padding: 8px 14px; margin-right: 5px; font-weight: 600;
            }
            QTabBar::tab:selected { background: #4b8fe8; color: white; }
            QLabel#sectionTitle { font-size: 17px; font-weight: 700; color: #253043; }
            QLabel#bigPrompt {
                font-size: 76px; font-weight: 800; color: #4b8fe8;
                background: #eef5ff; border-radius: 8px; padding: 20px;
            }
            QLabel#hintText { font-size: 16px; color: #64748b; }
            QPushButton {
                background: #edf3fb; color: #253043; border: 1px solid #d7e2ef;
                border-radius: 8px; padding: 10px 12px; font-size: 15px; font-weight: 600;
            }
            QPushButton:hover { background: #e2ecf8; }
            QPushButton#primaryButton {
                background: #3bb273; color: white; border: 0;
                font-size: 18px; padding: 13px 12px;
            }
            QPushButton#primaryButton:checked, QPushButton#primaryButton:hover { background: #319762; }
            QComboBox {
                background: white; border: 1px solid #d7e2ef; border-radius: 8px;
                padding: 8px; font-size: 15px;
            }
            QGroupBox {
                font-size: 15px; font-weight: 700; color: #253043;
                border: 1px solid #dfe7f1; border-radius: 8px; margin-top: 10px; padding: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
            QLabel#resultText { font-size: 30px; font-weight: 800; color: #253043; }
            QLabel#feedbackText { font-size: 18px; font-weight: 700; color: #253043; }
            QListWidget {
                border: 0; background: #f7fafd; border-radius: 8px; padding: 4px;
            }
            """
        )

    def mode(self) -> str:
        return ["letters", "words", "free"][self.tabs.currentIndex()]

    def dataset_for_mode(self) -> str:
        mode = self.mode()
        if mode == "words":
            return "words"
        return "letters"

    def target_text(self) -> str | None:
        mode = self.mode()
        if mode == "letters":
            return self.current_letter
        if mode == "words":
            return self.current_word
        return None

    def word_subtitle(self, word: str) -> str:
        return f"{CHINESE_MEANING.get(word, '基础词汇')}  |  请朗读这个单词"

    def update_prompt(self) -> None:
        idx = self.tabs.currentIndex()
        self.page_stack.setCurrentIndex(idx)
        self.letter_page.prompt.setText(self.current_letter)
        self.word_page.prompt.setText(self.current_word)
        self.word_page.subtitle.setText(self.word_subtitle(self.current_word))
        self.next_btn.setEnabled(idx != 2)

    def next_prompt(self) -> None:
        if self.mode() == "letters":
            self.current_letter = random.choice(LETTERS)
        elif self.mode() == "words":
            self.current_word = random.choice(DEFAULT_WORDS)
        self.update_prompt()

    def toggle_recording(self) -> None:
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        self.frames = []
        self.recording = True
        self.record_btn.setText("停止录音")
        self.record_btn.setStyleSheet("background: #e85d5d; color: white;")
        self.feedback_label.setText("正在录音...")
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self.audio_callback,
        )
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

        self.record_btn.setText("开始录音")
        self.record_btn.setStyleSheet("")
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

    def run_prediction(self) -> None:
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
        labels = [display_label(label) for label in result.labels]
        score = result.score
        if target:
            matched_index = None
            for i, label in enumerate(labels):
                if label.lower() == target.lower():
                    matched_index = i
                    break
            score = int(round(float(result.probabilities[matched_index]) * 100)) if matched_index is not None else 0

        feedback, stars, color = score_text(score)
        self.pred_label.setText(f"{pred}")
        self.conf_label.setText(f"置信度：{result.confidence * 100:.1f}%")
        self.feedback_label.setText(f"{stars}  {feedback}")
        self.feedback_label.setStyleSheet(f"color: {color};")
        self.score_widget.animate_to(score)

        standard_audio = None
        try:
            standard_audio = self.standard.load(target_for_score, self.slow_check.isChecked())
        except Exception:
            standard_audio = None
        self.current_standard = standard_audio
        self.chart_panel.update_result(
            self.current_audio,
            result.labels,
            result.probabilities,
            target_for_score,
            standard_audio,
        )
        self.add_history(target_for_score, pred, score)

    def add_history(self, target: str, predicted: str, score: int) -> None:
        item = {
            "target": target,
            "predicted": predicted,
            "score": score,
            "time": time.strftime("%H:%M:%S"),
        }
        self.history.insert(0, item)
        self.history = self.history[:10]
        self.history_list.clear()
        for row in self.history:
            text = f"{row['time']}  {row['target']} -> {row['predicted']}  {row['score']}分"
            self.history_list.addItem(QListWidgetItem(text))

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
