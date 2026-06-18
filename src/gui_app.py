"""SpeakEasy main window and entry point."""
from __future__ import annotations
import json, random, sys, time, tempfile, traceback
from pathlib import Path
import numpy as np
import sounddevice as sd
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QStackedWidget, QTabWidget,
    QVBoxLayout, QWidget,
)
from config import LETTERS, PROJECT_ROOT, SAMPLE_RATE
ROOT = PROJECT_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from api import get_word_info, score_pronunciation_youdao, get_deepseek_suggestions
from gui_widgets import (
    CircularScoreWidget, ChartPanel, ModelManager, StandardSpeech,
    StatsDialog, SettingsDialog, TrainingPage, FreeReadingInput,
    FreeModeResultPanel, score_text, display_label,
)
from audio_feature import extract_features, extract_mfcc_sequence
from pronunciation_scorer import PronunciationScorer

class SpeakEasyWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SpeakEasy · 英语口语发音训练系统")
        self.setMinimumSize(1000, 680)
        self.resize(1450, 920)

        self.model_manager = ModelManager()
        self.standard = StandardSpeech()
        self.pron_scorer = PronunciationScorer()
        self.history: list[dict] = []
        self.current_audio = np.array([], dtype=np.float32)
        self.current_standard: np.ndarray | None = None
        self.recording = False
        self.frames: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None
        self.current_letter = random.choice(LETTERS)
        self.current_letter_idx = LETTERS.index(self.current_letter)
        self.order_mode = "random"  # "random" or "sequential"
        self.nav_stack: list[dict] = []  # 导航历史：上一步状态
        self._current_scale = 1.0  # 当前缩放比例

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_live_wave)

        self.build_ui()
        self._apply_scaled_style(1.0)
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
        self.left_frame.setMinimumWidth(360)
        left_layout = QVBoxLayout(self.left_frame)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)

        # -- 模式标签 --
        self.tabs = QTabWidget(self.left_frame)
        self._tab_letter = QWidget(self.tabs)
        self._tab_free = QWidget(self.tabs)
        self.tabs.addTab(self._tab_letter, "🔤 字母训练")
        self.tabs.addTab(self._tab_free, "🎯 自由朗读")
        self.tabs.currentChanged.connect(self.update_prompt)
        left_layout.addWidget(self.tabs)

        # -- 题目/输入页栈 --
        self.page_stack = QStackedWidget()
        self.letter_page = TrainingPage("随机出题", self.current_letter,
                                        "请朗读屏幕中的大写字母")
        self.free_input = FreeReadingInput()
        self.page_stack.addWidget(self.letter_page)
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
        self.score_widget = CircularScoreWidget(120)
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
        self.free_input.random_btn.clicked.connect(self.do_random_word)
        self.free_input.prev_word_btn.clicked.connect(self.do_prev_word)
        self.free_input.next_word_btn.clicked.connect(self.do_next_word)
        self.free_input.browse_mode_box.currentIndexChanged.connect(self.on_browse_mode_changed)
        self.free_result_panel.free_standard_btn.clicked.connect(self.play_free_standard)

    # ── 自适应缩放 ────────────────────────────────────

    def _scale(self) -> float:
        """基于窗口宽度计算缩放比例"""
        return max(0.70, min(1.56, self.width() / 1450.0))

    def _s(self, base: int) -> str:
        """返回缩放后的 px 值字符串"""
        return f"{int(base * self._current_scale)}px"

    def _apply_scaled_style(self, scale: float) -> None:
        """根据缩放比例重建样式表"""
        self._current_scale = scale
        s = self._s  # shorthand

        self.setStyleSheet(f"""
        /* 全局 */
        QMainWindow {{ background: #f0f4f8; }}
        QLabel#appTitle {{
            color: #1a2332; font-size: {s(28)}; font-weight: 800;
            padding: {s(6)} 0; letter-spacing: 2px;
        }}

        /* 面板 */
        QFrame#sidePanel {{
            background: #ffffff; border: 1px solid #e2e8f0;
            border-radius: {s(12)};
        }}
        QFrame#rightPanel {{
            background: #ffffff; border: 1px solid #e2e8f0;
            border-radius: {s(12)};
        }}

        /* 标签页 */
        QTabWidget::pane {{ border: 0; }}
        QTabBar::tab {{
            background: #f1f5f9; color: #475569; border-radius: {s(8)};
            padding: {s(10)} {s(16)}; margin-right: {s(4)}; font-size: {s(16)}; font-weight: 600;
        }}
        QTabBar::tab:selected {{ background: #4b8fe8; color: white; }}

        /* 标题 */
        QLabel#sectionTitle {{
            font-size: {s(20)}; font-weight: 700; color: #1a2332;
        }}
        QLabel#bigPrompt {{
            font-size: {s(72)}; font-weight: 800; color: #4b8fe8;
            background: #eef5ff; border-radius: {s(10)}; padding: {s(16)};
        }}
        QLabel#hintText {{
            font-size: {s(15)}; color: #64748b; line-height: 1.5;
        }}

        /* 通用按钮 */
        QPushButton {{
            background: #f1f5f9; color: #334155; border: 1px solid #e2e8f0;
            border-radius: {s(8)}; padding: {s(8)} {s(12)}; font-size: {s(14)}; font-weight: 600;
        }}
        QPushButton:hover {{ background: #e2e8f0; }}

        /* 录音按钮 */
        QPushButton#recordBtn {{
            background: #3bb273; color: white; border: 0;
            font-size: {s(20)}; font-weight: 700; border-radius: {s(12)};
        }}
        QPushButton#recordBtn:hover {{ background: #319762; }}

        /* 强调按钮 */
        QPushButton#accentBtn {{
            background: #4b8fe8; color: white; border: 0;
        }}
        QPushButton#accentBtn:hover {{ background: #3a7bd5; }}

        /* 下拉框 */
        QComboBox {{
            background: white; border: 1px solid #e2e8f0; border-radius: {s(8)};
            padding: {s(6)} {s(10)}; font-size: {s(14)};
        }}
        QComboBox:hover {{ border-color: #4b8fe8; }}

        /* 分组框 */
        QGroupBox {{
            font-size: {s(15)}; font-weight: 700; color: #1a2332;
            border: 1px solid #e2e8f0; border-radius: {s(10)};
            margin-top: {s(6)}; padding: {s(10)} {s(8)} {s(8)} {s(8)};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin; left: {s(14)}; padding: 0 {s(6)};
            color: #4b8fe8;
        }}

        /* 结果 */
        QLabel#resultText {{
            font-size: {s(28)}; font-weight: 800; color: #1a2332;
        }}
        QLabel#feedbackText {{
            font-size: {s(18)}; font-weight: 700; color: #1a2332;
        }}

        /* 列表 */
        QListWidget {{
            border: 0; background: #f8fafc; border-radius: {s(8)};
            padding: {s(3)}; font-size: {s(13)};
        }}

        /* ScrollArea */
        QScrollArea {{ border: 0; background: transparent; }}
        QScrollBar:vertical {{
            background: transparent; width: {s(6)};
        }}
        QScrollBar::handle:vertical {{
            background: #cbd5e1; border-radius: {s(3)}; min-height: {s(20)};
        }}
        QScrollBar::handle:vertical:hover {{ background: #94a3b8; }}
        """)

        # 动态缩放左栏宽度
        self.left_frame.setMinimumWidth(int(360 * scale))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        new_scale = self._scale()
        if abs(new_scale - self._current_scale) > 0.02:
            self._apply_scaled_style(new_scale)
            self._update_dynamic_sizes()

    def _update_dynamic_sizes(self) -> None:
        """更新代码中动态设置的尺寸"""
        sc = self._current_scale
        # 评分圆环
        self.score_widget.set_scale(sc)
        # 自由模式输入框字体
        self.free_input.word_input.setStyleSheet(
            f"QLineEdit {{ font-size: {int(15*sc)}px; padding: {int(4*sc)}px {int(8*sc)}px; "
            f"border: 2px solid #d7e2ef; border-radius: {int(8*sc)}px; "
            f"background: white; }}"
            f"QLineEdit:focus {{ border-color: #4b8fe8; }}")
        self.free_input.word_input.setMinimumHeight(int(32 * sc))
        self.free_input.lookup_btn.setMinimumHeight(int(32 * sc))
        self.free_input.random_btn.setMinimumHeight(int(28 * sc))
        self.free_input.prev_word_btn.setMinimumHeight(int(34 * sc))
        self.free_input.next_word_btn.setMinimumHeight(int(34 * sc))
        self.free_input.browse_mode_box.setMinimumHeight(int(34 * sc))
        self.free_input.hint_label.setStyleSheet(
            f"font-size: {int(14*sc)}px; color: #64748b; padding: {int(8*sc)}px; "
            f"background: #f7fafd; border-radius: {int(8*sc)}px;")
        # 自由模式右侧面板
        self.free_result_panel.set_scale(sc)

    # ── mode / dataset / target ────────────────────────

    def mode(self) -> str:
        return ["letters", "free"][self.tabs.currentIndex()]

    def dataset_for_mode(self) -> str:
        return "letters"

    def target_text(self) -> str | None:
        return self.current_letter if self.mode() == "letters" else None

    # ── update_prompt / next_prompt ────────────────────

    def update_prompt(self) -> None:
        idx = self.tabs.currentIndex()
        self.page_stack.setCurrentIndex(idx)
        self.letter_page.prompt.setText(self.current_letter)

        # 自由朗读模式：隐藏模型选择/导航/识别结果
        is_free = idx == 1
        self.model_section.setVisible(not is_free)
        self.nav_section.setVisible(not is_free)
        self.result_box.setVisible(not is_free)
        self.next_btn.setEnabled(not is_free)
        self.prev_btn.setEnabled(not is_free and len(self.nav_stack) > 0)
        if is_free:
            self.nav_stack.clear()

        # 切换右侧面板
        if is_free:
            self.right_stack.setCurrentIndex(1)
        else:
            self.right_stack.setCurrentIndex(0)

    def on_order_changed(self, idx: int) -> None:
        self.order_mode = "random" if idx == 0 else "sequential"

    def _push_nav_state(self) -> None:
        """保存当前状态到导航历史栈"""
        self.nav_stack.append({
            "letter": self.current_letter,
            "letter_idx": self.current_letter_idx,
        })

    def _restore_nav_state(self, state: dict) -> None:
        """从导航历史栈恢复状态"""
        self.current_letter = state["letter"]
        self.current_letter_idx = state["letter_idx"]

    def prev_prompt(self) -> None:
        if not self.nav_stack:
            return
        state = self.nav_stack.pop()
        self._restore_nav_state(state)
        self.prev_btn.setEnabled(len(self.nav_stack) > 0)
        self.update_prompt()

    def next_prompt(self) -> None:
        self._push_nav_state()
        if self.order_mode == "sequential":
            self.current_letter_idx = (self.current_letter_idx + 1) % len(LETTERS)
            self.current_letter = LETTERS[self.current_letter_idx]
        else:
            self.current_letter = random.choice(LETTERS)
            self.current_letter_idx = LETTERS.index(self.current_letter)
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
        try:
            self.current_audio = np.concatenate(self.frames).astype(np.float32)
            self.run_prediction()
        except Exception:
            import traceback as _tb
            print("[stop_recording 异常]", file=sys.stderr)
            _tb.print_exc(file=sys.stderr)
            self.feedback_label.setText("评分出错，请重试")
            self.score_widget.animate_to(0)

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
        matched_index = None
        if target:
            for i, l in enumerate(labels):
                if l.lower() == target.lower():
                    matched_index = i
                    break
            score = int(round(float(result.probabilities[matched_index]) * 100)) \
                if matched_index is not None else 0

        # ── 三维度融合评分（置信度 + DTW + 声学特征） ──
        standard_audio = None
        pron_detail = None
        try:
            standard_audio = self.standard.load(
                target_for_score, self.slow_check.isChecked())
            if standard_audio is not None and getattr(standard_audio, "size", 0) > 0:
                user_mfcc_seq = extract_mfcc_sequence(
                    self.current_audio, SAMPLE_RATE)
                std_mfcc_seq = extract_mfcc_sequence(
                    standard_audio, SAMPLE_RATE)
                target_idx = matched_index if matched_index is not None \
                    else int(np.argmax(result.probabilities))
                pron_result = self.pron_scorer.score(
                    result.probabilities, target_idx,
                    user_mfcc_seq=user_mfcc_seq,
                    template_mfcc_seq=std_mfcc_seq,
                    user_audio=self.current_audio,
                    template_audio=standard_audio,
                )
                score = int(round(pron_result.total_score))
                pron_detail = pron_result
        except Exception:
            import traceback as _tb
            print("[三维度评分失败，回退到置信度评分]", file=sys.stderr)
            _tb.print_exc(file=sys.stderr)
            # 回退到仅置信度评分，不中断程序
        self.current_standard = standard_audio

        try:
            feedback, stars, color = score_text(score)
            self.pred_label.setText(f"{pred}")
            if pron_detail is not None:
                self.conf_label.setText(
                    f"综合{score}分 | 置信{pron_detail.conf_score:.0f} "
                    f"DTW{pron_detail.dtw_score:.0f} 声学{pron_detail.acoustic_score:.0f}")
            else:
                self.conf_label.setText(f"置信度：{result.confidence * 100:.1f}%")
            self.feedback_label.setText(f"{stars}  {feedback}")
            fs = int(18 * self._current_scale)
            self.feedback_label.setStyleSheet(f"color: {color}; font-size: {fs}px; font-weight: 700;")
            self.score_widget.animate_to(score)

            self.chart_panel.update_result(self.current_audio, result.labels,
                                           result.probabilities, target_for_score,
                                           standard_audio)
            self.add_history(target_for_score, pred, score)
        except Exception:
            import traceback as _tb
            print("[显示/图表更新异常]", file=sys.stderr)
            _tb.print_exc(file=sys.stderr)
            self.feedback_label.setText("结果显示异常，请重试")
            self.score_widget.animate_to(0)

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
                fs = int(18 * self._current_scale)
                self.feedback_label.setText(
                    f"{stars}  综合{overall:.0f} | 准确度{pron:.0f} | 流利度{fluency:.0f}")
                self.feedback_label.setStyleSheet(f"color: {color}; font-size: {fs}px; font-weight: 700;")
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
                    f"color: {color}; font-size: {fs}px; font-weight: 700;")
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

    def do_random_word(self) -> None:
        """随机推荐单词并自动查询"""
        self.free_input.set_random_word()
        self.do_word_lookup()

    def do_prev_word(self) -> None:
        """浏览上一个单词并自动查询"""
        self.free_input.prev_word()
        self.do_word_lookup()

    def do_next_word(self) -> None:
        """浏览下一个单词并自动查询"""
        self.free_input.next_word()
        self.do_word_lookup()

    def on_browse_mode_changed(self, idx: int) -> None:
        """切换随机/顺序浏览时，同步刷新提示"""
        mode = "📋 顺序浏览" if idx == 1 else "🎲 随机浏览"
        self.free_input.hint_label.setText(f"当前模式：{mode}  点击左右箭头浏览单词")

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

    # ── 全局异常钩子：未捕获异常弹窗显示，避免静默崩溃 ──
    def _global_excepthook(exc_type, exc_value, exc_tb):
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(f"[未捕获异常]\n{msg}", file=sys.stderr)
        QMessageBox.critical(None, "程序异常", f"发生未捕获的异常：\n\n{msg}")
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = _global_excepthook

    window = SpeakEasyWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()