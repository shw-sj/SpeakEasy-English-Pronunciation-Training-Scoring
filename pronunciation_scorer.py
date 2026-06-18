"""
发音智能评测打分模块 —— 三维度融合评分（核心创新）
====================================================
综合三种打分策略，加权融合得出最终评分（0-100分）：
  1. 嵌入相似度打分（权重45%）—— 基于神经网络128维嵌入的余弦相似度
  2. DTW对齐距离打分（权重30%）—— 动态时间规整用户MFCC与标准模板对齐
  3. 声学特征相似度打分（权重25%）—— F0基频 + 共振峰 + 能量包络

最终得分 = 0.45 × score_emb + 0.30 × score_dtw + 0.25 × score_acoustic

使用方式：
  from pronunciation_scorer import PronunciationScorer, ScoreResult
  scorer = PronunciationScorer()
  result = scorer.score(user_embedding, target_embedding,
                         user_mfcc_seq=user_seq, template_mfcc_seq=ref_seq,
                         user_audio=user_wav, template_audio=ref_wav)
  print(result.total_score, result.grade, result.star_rating)

v2 变更：
  - 新增 EmbeddingScorer：用神经网络学到的语音嵌入余弦相似度
    替代原来的 softmax 置信度，从根本上解决分类目标
    与发音质量评估任务之间的错配。
  - 保留 ConfidenceScorer 作为向后兼容的回退方案。
  - 支持通过 scorer.use_embedding_scorer 切换模式。
"""

import numpy as np
from scipy.spatial.distance import cdist
from collections import namedtuple
import warnings

# ==================== 评分结果数据结构 ====================
ScoreResult = namedtuple("ScoreResult", [
    "total_score",       # 最终综合评分 [0, 100]
    "conf_score",        # 置信度评分 [0, 100]
    "dtw_score",         # DTW对齐评分 [0, 100]
    "acoustic_score",    # 声学相似度评分 [0, 100]
    "grade",             # 评级：优秀 / 良好 / 一般 / 需努力
    "star_rating",       # 星级：★ x 1~5
    "detail",            # 详细反馈字典
])


# ====================================================================
# 第一部分：置信度打分（权重40%）
# ====================================================================
class ConfidenceScorer:
    """
    置信度打分器
    基于神经网络Softmax输出中目标类的概率值评估发音质量

    公式：score_conf = softmax_prob(target_class) × 100
    """

    def __init__(self, temperature=1.0):
        """
        :param temperature: Softmax温度系数（<1使分布更尖锐，>1更平滑）
        """
        self.temperature = temperature

    def score(self, probs, target_class_idx):
        """
        根据Softmax输出概率计算置信度评分

        :param probs: Softmax概率向量 shape=[类别数] 或 [1, 类别数]
        :param target_class_idx: 目标类别索引 (int)
        :return: 置信度评分 [0, 100]
        """
        probs = np.array(probs).flatten()
        if target_class_idx < 0 or target_class_idx >= len(probs):
            raise ValueError(
                f"目标类别索引 {target_class_idx} 超出范围 [0, {len(probs)-1}]")

        # 温度缩放
        if self.temperature != 1.0:
            log_probs = np.log(np.maximum(probs, 1e-8))
            scaled = np.exp(log_probs / self.temperature)
            probs = scaled / np.sum(scaled)

        score = float(probs[target_class_idx] * 100.0)
        return float(np.clip(score, 0.0, 100.0))

    def score_batch(self, probs_batch, target_indices):
        """批量评分"""
        return np.array([self.score(p, i)
                         for p, i in zip(probs_batch, target_indices)])

    def get_topk_info(self, probs, label_map=None, k=5):
        """
        获取Top-K预测信息（用于详细反馈）

        :param probs: Softmax概率向量
        :param label_map: {idx: label_name}
        :param k: 返回前K个
        :return: [(label, prob), ...]
        """
        probs = np.array(probs).flatten()
        topk_idx = np.argsort(probs)[::-1][:k]
        results = []
        for idx in topk_idx:
            label = label_map.get(idx, str(idx)) if label_map else str(idx)
            results.append((label, float(probs[idx])))
        return results


# ====================================================================
# 第二部分：嵌入相似度打分（权重40%） ★v2 核心创新★
# ====================================================================
class EmbeddingScorer:
    """嵌入相似度打分器 —— 解决分类目标与发音质量评估的错配。

    核心思路：
      不用 softmax 置信度（"模型有多确定这是字母X"），
      而是用神经网络倒数第二层的 128 维嵌入向量，
      计算用户发音嵌入与标准发音嵌入的余弦相似度。

    直觉：
      - 发音越标准，嵌入向量越靠近该字母的"理想嵌入"
      - 发音走样，嵌入会偏离理想位置
      - 这直接度量了"发音有多接近标准"，而非"分类有多确定"

    Attributes
    ----------
    standard_embeddings : dict
        {letter: np.ndarray(128,)} 每个字母的标准发音嵌入。
        通过将 TTS 标准音频送入模型提取嵌入得到。
    similarity_mode : str
        "cosine" — 余弦相似度
        "euclidean" — 欧氏距离映射
    """

    def __init__(
        self,
        standard_embeddings: dict | None = None,
        similarity_mode: str = "cosine",
        euclidean_scale: float = 0.05,
    ):
        """
        Parameters
        ----------
        standard_embeddings : dict or None
            预计算的标准嵌入 {label: np.array(128,)}。
            如果为 None，需要后续调用 ``set_standards()``。
        similarity_mode : str
            "cosine" 或 "euclidean"。
        euclidean_scale : float
            欧氏距离映射到 [0,100] 的缩放系数。
            score = max(0, 100 - euclidean_scale * dist)。
        """
        self.standard_embeddings = standard_embeddings or {}
        self.similarity_mode = similarity_mode
        self.euclidean_scale = euclidean_scale

    def set_standards(self, standard_embeddings: dict) -> None:
        """设置或更新标准嵌入字典。

        Parameters
        ----------
        standard_embeddings : dict
            {label: np.array(128,)} 映射。
        """
        self.standard_embeddings = standard_embeddings

    def score(
        self,
        user_embedding: np.ndarray,
        target_label,
    ) -> float:
        """计算嵌入相似度评分。

        Parameters
        ----------
        user_embedding : np.ndarray
            用户音频的 128 维嵌入向量，shape=(128,) 或 (1, 128)。
        target_label : str or int
            目标字母（如 "A" 或 0），用于查找标准嵌入。

        Returns
        -------
        score : float  [0, 100]
        """
        user_emb = np.array(user_embedding, dtype=np.float64).flatten()

        if target_label not in self.standard_embeddings:
            # 回退：如果没有标准嵌入，返回 50 分（中性）
            return 50.0

        std_emb = np.array(
            self.standard_embeddings[target_label], dtype=np.float64
        ).flatten()

        if self.similarity_mode == "cosine":
            dot = np.dot(user_emb, std_emb)
            norm = (
                np.linalg.norm(user_emb) * np.linalg.norm(std_emb) + 1e-10
            )
            cosine_sim = dot / norm
            # 余弦相似度 [-1, 1] → [0, 100]
            # cos=1 → 100分, cos=0 → 50分, cos=-1 → 0分
            score = (cosine_sim + 1.0) / 2.0 * 100.0
        elif self.similarity_mode == "euclidean":
            dist = np.linalg.norm(user_emb - std_emb)
            score = max(0.0, 100.0 - self.euclidean_scale * dist)
        else:
            raise ValueError(f"未知相似度模式: {self.similarity_mode}")

        return float(np.clip(score, 0.0, 100.0))

    def score_batch(
        self,
        user_embeddings: np.ndarray,
        target_labels: list,
    ) -> np.ndarray:
        """批量评分。

        Parameters
        ----------
        user_embeddings : (N, 128)
        target_labels : list of str/int, length N

        Returns
        -------
        scores : (N,) float
        """
        return np.array([
            self.score(user_embeddings[i], target_labels[i])
            for i in range(len(target_labels))
        ])

    def get_topk_similar(
        self,
        user_embedding: np.ndarray,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """找到与用户嵌入最相似的 k 个标准嵌入（用于详细反馈）。

        Returns
        -------
        [(label, similarity_score), ...] 按相似度降序排列。
        """
        user_emb = np.array(user_embedding, dtype=np.float64).flatten()
        results = []
        for label, std_emb in self.standard_embeddings.items():
            std = np.array(std_emb, dtype=np.float64).flatten()
            dot = np.dot(user_emb, std)
            norm = (
                np.linalg.norm(user_emb) * np.linalg.norm(std) + 1e-10
            )
            sim = float((dot / norm + 1.0) / 2.0 * 100.0)
            results.append((str(label), sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


# ====================================================================
# 第二部分：DTW对齐距离打分（权重35%） ★创新★
# ====================================================================
class DTWScorer:
    """
    DTW发音评分器

    动态时间规整将用户MFCC序列与标准模板对齐后评分。
    即使语速不同（快/慢），DTW也能找到最优时间对齐路径。

    公式：score_dtw = max(0, 100 - α × dtw_avg_distance)

    注：如已有 dtw_matcher.py，优先使用其中的 DTWMatcher；
        本类提供自包含的DTW实现，无需额外依赖。
    """

    def __init__(self, alpha=1.0, window=None, distance_metric='euclidean'):
        """
        :param alpha: 评分缩放系数（通过实验校准，默认1.0）
        :param window: Sakoe-Chiba窗口宽度（None=无约束）
        :param distance_metric: 'euclidean' / 'cosine' / 'cityblock'
        """
        self.alpha = alpha
        self.window = window
        self.distance_metric = distance_metric

    def compute(self, user_mfcc, template_mfcc):
        """
        对用户MFCC序列与标准模板进行DTW对齐并评分

        :param user_mfcc: shape=[n_frames_user, n_features]
        :param template_mfcc: shape=[n_frames_template, n_features]
        :return: dict {dtw_distance, avg_distance, score, warping_path}
        """
        user_mfcc = np.atleast_2d(user_mfcc).astype(np.float64)
        template_mfcc = np.atleast_2d(template_mfcc).astype(np.float64)

        if user_mfcc.shape[1] != template_mfcc.shape[1]:
            raise ValueError(
                f"MFCC维度不一致：用户={user_mfcc.shape[1]}, 模板={template_mfcc.shape[1]}")

        M, N = user_mfcc.shape[0], template_mfcc.shape[0]
        if M == 0 or N == 0:
            return {"dtw_distance": float('inf'), "avg_distance": float('inf'),
                    "score": 0.0, "warping_path": []}

        # 局部距离矩阵
        dist_mat = cdist(user_mfcc, template_mfcc, metric=self.distance_metric)
        # 累计代价矩阵
        cost = self._build_cost_matrix(dist_mat, M, N)
        # 回溯路径
        path = self._backtrack(cost, M, N)
        # 评分
        dtw_dist = cost[-1, -1]
        avg_dist = dtw_dist / len(path)
        score = max(0.0, 100.0 - self.alpha * avg_dist)

        return {
            "dtw_distance": dtw_dist,
            "avg_distance": avg_dist,
            "score": float(score),
            "warping_path": path,
        }

    def _build_cost_matrix(self, dist_mat, M, N):
        """动态规划构建累计代价矩阵"""
        cost = np.full((M, N), np.inf)
        cost[0, 0] = dist_mat[0, 0]
        for i in range(1, M):
            cost[i, 0] = cost[i - 1, 0] + dist_mat[i, 0]
        for j in range(1, N):
            cost[0, j] = cost[0, j - 1] + dist_mat[0, j]

        for i in range(1, M):
            if self.window is not None:
                j_start = max(1, i - self.window)
                j_end = min(N, i + self.window + 1)
            else:
                j_start, j_end = 1, N

            for j in range(j_start, j_end):
                cost[i, j] = dist_mat[i, j] + min(
                    cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

        return cost

    def _backtrack(self, cost_matrix, M, N):
        """从右下角回溯到左上角，得到最优规整路径"""
        path = [(M - 1, N - 1)]
        i, j = M - 1, N - 1
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                neighbors = [(i - 1, j), (i, j - 1), (i - 1, j - 1)]
                costs = [cost_matrix[ni, nj] for ni, nj in neighbors]
                i, j = neighbors[int(np.argmin(costs))]
            path.append((i, j))
        path.reverse()
        return path

    def score(self, user_mfcc, template_mfcc):
        """快捷评分接口，只返回分数"""
        return self.compute(user_mfcc, template_mfcc)["score"]

    def calibrate_alpha(self, good_samples, bad_samples, templates,
                        target_good=85.0, target_bad=40.0):
        """
        通过实验数据自动校准alpha系数

        :param good_samples: 好发音MFCC列表
        :param bad_samples: 差发音MFCC列表
        :param templates: 对应标准模板列表
        :param target_good: 好发音目标分数
        :param target_bad: 差发音目标分数
        :return: 校准后的alpha值
        """
        temp_scorer = DTWScorer(alpha=1.0)
        good_dists = [temp_scorer.compute(s, t)["avg_distance"]
                      for s, t in zip(good_samples, templates)]
        bad_dists = [temp_scorer.compute(s, t)["avg_distance"]
                     for s, t in zip(bad_samples, templates)]

        alpha_good = (100.0 - target_good) / (np.mean(good_dists) + 1e-8)
        alpha_bad = (100.0 - target_bad) / (np.mean(bad_dists) + 1e-8)
        alpha = max((alpha_good + alpha_bad) / 2.0, 1e-6)

        print(f"[Alpha校准] 好发音平均距离={np.mean(good_dists):.4f}, "
              f"差发音平均距离={np.mean(bad_dists):.4f} → alpha={alpha:.4f}")
        self.alpha = alpha
        return alpha


# ====================================================================
# 第三部分：声学特征相似度打分（权重25%） ★创新★
# ====================================================================
class AcousticScorer:
    """
    声学特征相似度打分器
    比较用户发音与标准发音的三个声学维度：
      1. 基频（F0）轮廓相似度
      2. 共振峰（Formant）频率差异
      3. 短时能量包络相似度

    综合三维度余弦相似度 → 0~100分
    """

    def __init__(self, sample_rate=16000, frame_length=0.025, frame_shift=0.010):
        """
        :param sample_rate: 采样率（Hz）
        :param frame_length: 帧长（秒）
        :param frame_shift: 帧移（秒）
        """
        self.sample_rate = sample_rate
        self.frame_length = int(frame_length * sample_rate)
        self.frame_shift = int(frame_shift * sample_rate)

    # ---------- F0提取（自相关法） ----------
    def extract_f0(self, audio_signal):
        """
        基于自相关法提取基频(F0)轮廓

        :param audio_signal: 一维音频信号
        :return: F0序列 [n_frames]，单位Hz
        """
        signal = np.array(audio_signal, dtype=np.float64)
        if len(signal) == 0:
            return np.array([0.0])

        f0_min, f0_max = 50.0, 400.0  # 儿童基频范围
        n_frames = max(1, (len(signal) - self.frame_length) // self.frame_shift + 1)
        f0_values = []

        for i in range(n_frames):
            start = i * self.frame_shift
            frame = signal[start:start + self.frame_length]
            if len(frame) < self.frame_length:
                frame = np.pad(frame, (0, self.frame_length - len(frame)))
            frame = frame - np.mean(frame)

            # 自相关
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr) // 2:]
            min_lag = int(self.sample_rate / f0_max)
            max_lag = min(int(self.sample_rate / f0_min), len(corr) - 1)

            if min_lag >= max_lag or np.max(corr[min_lag:max_lag]) < 0.01:
                f0_values.append(0.0)
            else:
                peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
                f0_values.append(self.sample_rate / peak_lag if peak_lag > 0 else 0.0)

        return np.array(f0_values)

    # ---------- 共振峰提取（简化LPC法） ----------
    def extract_formants(self, audio_signal, n_formants=3):
        """
        基于LPC提取共振峰频率

        :return: 共振峰序列 [n_frames, n_formants]
        """
        signal = np.array(audio_signal, dtype=np.float64)
        if len(signal) == 0:
            return np.zeros((1, n_formants))

        n_frames = max(1, (len(signal) - self.frame_length) // self.frame_shift + 1)
        lpc_order = 12
        formants_seq = []

        for i in range(n_frames):
            start = i * self.frame_shift
            frame = signal[start:start + self.frame_length]
            if len(frame) < self.frame_length:
                frame = np.pad(frame, (0, self.frame_length - len(frame)))
            frame = frame - np.mean(frame)

            try:
                r = np.correlate(frame, frame, mode='full')
                r = r[len(r) // 2:len(r) // 2 + lpc_order + 1]
                a, e = self._levinson_durbin(r, lpc_order)

                if e < 1e-10:
                    formants_seq.append([0.0] * n_formants)
                    continue

                roots = np.roots(np.append(1, a))
                roots = roots[np.abs(roots) < 1]
                angles = np.abs(np.angle(roots))
                freqs = np.sort(angles * self.sample_rate / (2 * np.pi))
                freqs = freqs[freqs > 50]

                if len(freqs) >= n_formants:
                    formants_seq.append(freqs[:n_formants].tolist())
                else:
                    formants_seq.append(
                        list(freqs) + [0.0] * (n_formants - len(freqs)))
            except Exception:
                formants_seq.append([0.0] * n_formants)

        return np.array(formants_seq)

    def _levinson_durbin(self, r, order):
        """Levinson-Durbin递推求解LPC系数"""
        a = np.zeros(order)
        e = r[0]
        for i in range(order):
            k = r[i + 1] - sum(a[j] * r[i - j] for j in range(i))
            k /= e if e > 1e-10 else 1e-10
            a[i] = k
            for j in range(i):
                a[j] = a[j] - k * a[i - 1 - j]
            e *= (1 - k * k)
            if e < 1e-10:
                break
        return a, e

    # ---------- 短时能量包络 ----------
    def extract_energy_envelope(self, audio_signal):
        """提取短时能量包络（对数RMS）"""
        signal = np.array(audio_signal, dtype=np.float64)
        if len(signal) == 0:
            return np.array([0.0])
        n_frames = max(1, (len(signal) - self.frame_length) // self.frame_shift + 1)
        energy = []
        for i in range(n_frames):
            start = i * self.frame_shift
            frame = signal[start:start + self.frame_length]
            if len(frame) < self.frame_length:
                frame = np.pad(frame, (0, self.frame_length - len(frame)))
            rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
            energy.append(np.log10(rms + 1e-10))
        return np.array(energy)

    # ---------- 综合声学特征向量 ----------
    def extract_acoustic_features(self, audio_signal):
        """
        提取综合声学特征向量

        :return: 拼接特征向量
          [f0_mean, f0_std, f0_p25, f0_p75,
           f1_mean, f1_std, f2_mean, f2_std, f3_mean, f3_std,
           energy_mean, energy_std, energy_range]
        """
        f0 = self.extract_f0(audio_signal)
        # 仅保留浊音帧（F0 > 50 Hz），避免清音帧的 0 值干扰统计
        f0_voiced = f0[f0 > 50]
        if len(f0_voiced) < 3:
            f0_voiced = np.array([120.0])  # 默认典型F0值
        f0_feat_raw = np.array([np.mean(f0_voiced), np.std(f0_voiced),
                                np.percentile(f0_voiced, 25), np.percentile(f0_voiced, 75)])

        formants = self.extract_formants(audio_signal, n_formants=3)
        fmt_feat_raw = []
        for d in range(formants.shape[1]):
            fd = formants[:, d]
            # 仅保留有效共振峰（> 80 Hz），过滤无效帧
            fd_v = fd[fd > 80]
            if len(fd_v) < 3:
                fd_v = np.array([500.0 * (d + 1)])  # 默认典型共振峰
            fmt_feat_raw.extend([np.mean(fd_v), np.std(fd_v)])

        energy = self.extract_energy_envelope(audio_signal)
        energy_feat_raw = np.array([np.mean(energy), np.std(energy),
                                    np.max(energy) - np.min(energy)])

        # ── 对各子特征组独立 L2 归一化，避免共振峰量纲主导余弦相似度 ──
        def _norm_group(vec):
            v = np.array(vec, dtype=np.float64)
            norm = np.linalg.norm(v) + 1e-10
            return v / norm

        f0_feat = _norm_group(f0_feat_raw)
        fmt_feat = _norm_group(fmt_feat_raw)
        energy_feat = _norm_group(energy_feat_raw)

        return np.concatenate([f0_feat, fmt_feat, energy_feat])

    # ---------- 相似度计算 ----------
    def compute_similarity(self, user_audio, template_audio):
        """
        计算用户发音与标准发音的声学相似度

        :param user_audio: 用户音频信号 (1D array)
        :param template_audio: 标准音频信号 (1D array)
        :return: 相似度评分 [0, 100]
        """
        uf = self.extract_acoustic_features(user_audio)
        tf = self.extract_acoustic_features(template_audio)
        dot = np.dot(uf, tf)
        norm = np.linalg.norm(uf) * np.linalg.norm(tf) + 1e-10
        cosine_sim = dot / norm
        return float(np.clip(cosine_sim * 100.0, 0.0, 100.0))

    def compute_detailed_similarity(self, user_audio, template_audio):
        """
        详细声学对比（返回各子维度分数）

        :return: dict
        """
        # F0相似度
        uf0 = self.extract_f0(user_audio)
        tf0 = self.extract_f0(template_audio)
        if len(uf0) > 1 and len(tf0) > 1:
            interp = np.interp(np.linspace(0, 1, len(tf0)),
                               np.linspace(0, 1, len(uf0)), uf0)
            f0_corr = np.corrcoef(interp, tf0)[0, 1]
            f0_sim = max(0.0, (f0_corr + 1) / 2 * 100)
        else:
            f0_sim = 50.0

        # 共振峰相似度
        ufmt = self.extract_formants(user_audio, n_formants=3)
        tfmt = self.extract_formants(template_audio, n_formants=3)
        fmt_sims = []
        for d in range(min(ufmt.shape[1], tfmt.shape[1])):
            um, tm = np.mean(ufmt[:, d]), np.mean(tfmt[:, d])
            if um > 0 and tm > 0:
                fmt_sims.append(max(0.0, 100 - abs(um - tm) / (tm + 1e-10) * 100))
            else:
                fmt_sims.append(50.0)
        fmt_sim = np.mean(fmt_sims) if fmt_sims else 50.0

        # 能量相似度
        ue = self.extract_energy_envelope(user_audio)
        te = self.extract_energy_envelope(template_audio)
        if len(ue) > 1 and len(te) > 1:
            interp_e = np.interp(np.linspace(0, 1, len(te)),
                                 np.linspace(0, 1, len(ue)), ue)
            e_corr = np.corrcoef(interp_e, te)[0, 1]
            e_sim = max(0.0, (e_corr + 1) / 2 * 100)
        else:
            e_sim = 50.0

        overall = f0_sim * 0.35 + fmt_sim * 0.35 + e_sim * 0.30
        return {
            "f0_similarity": round(f0_sim, 2),
            "formant_similarity": round(fmt_sim, 2),
            "energy_similarity": round(e_sim, 2),
            "overall_acoustic_score": round(overall, 2),
        }


# ====================================================================
# 第四部分：三维度融合发音评分器 ★核心★
# ====================================================================
class PronunciationScorer:
    """
    综合发音评分器
    ================
    融合三种打分策略，加权得出最终评分（0-100分）：

      - 置信度打分（权重40%）：BP网络Softmax目标类概率
      - DTW对齐打分（权重35%）：MFCC序列DTW规整距离
      - 声学特征打分（权重25%）：F0+共振峰+能量包络相似度

    最终得分 = 0.45 × first + 0.30 × dtw + 0.25 × acoustic

    评分等级：
      90-100分：优秀 ★★★★★
      75-89分 ：良好 ★★★★☆
      60-74分 ：一般 ★★★☆☆
      0-59分  ：需努力 ★★☆☆☆
    """

    def __init__(self, emb_weight=0.45, dtw_weight=0.30, acoustic_weight=0.25,
                 sample_rate=16000, dtw_alpha=1.0,
                 use_embedding=True, standard_embeddings=None):
        total = emb_weight + dtw_weight + acoustic_weight
        self.emb_weight = emb_weight / total
        self.dtw_weight = dtw_weight / total
        self.acoustic_weight = acoustic_weight / total
        self.sample_rate = sample_rate
        self.use_embedding = use_embedding

        # v2: 嵌入打分器（默认，解决分类目标与质量评估的错配）
        self.emb_scorer = EmbeddingScorer(standard_embeddings=standard_embeddings)
        # v1: 置信度打分器（回退）
        self.conf_scorer = ConfidenceScorer()
        self.dtw_scorer = DTWScorer(alpha=dtw_alpha)
        self.acoustic_scorer = AcousticScorer(sample_rate=sample_rate)

    # ==================== 综合评分主接口 (v2) ====================
    def score(self, softmax_probs=None, target_class_idx=None,
              user_embedding=None, target_label=None,
              user_mfcc_seq=None, template_mfcc_seq=None,
              user_audio=None, template_audio=None):
        """
        综合评分（主入口，v2：支持嵌入相似度）。

        两种调用方式（向后兼容）：

        1. **嵌入模式** (v2 推荐，默认):
           scorer.score(user_embedding=emb, target_label="A",
                        user_mfcc_seq=..., template_mfcc_seq=...,
                        user_audio=..., template_audio=...)

        2. **置信度模式** (v1 回退):
           scorer.score(softmax_probs=probs, target_class_idx=0,
                        user_mfcc_seq=..., template_mfcc_seq=...,
                        user_audio=..., template_audio=...)

        :return: ScoreResult 命名元组
        """
        # 1. 嵌入/置信度评分
        if self.use_embedding and user_embedding is not None and target_label is not None:
            score_first = self.emb_scorer.score(user_embedding, target_label)
            first_name = "emb"
        elif softmax_probs is not None and target_class_idx is not None:
            score_first = self.conf_scorer.score(softmax_probs, target_class_idx)
            first_name = "conf"
        else:
            score_first = 50.0
            first_name = "conf"
            if self.use_embedding:
                warnings.warn(
                    "嵌入模式已启用但未提供 user_embedding/target_label，"
                    "第一维度使用默认值50分")

        # 2. DTW评分
        if user_mfcc_seq is not None and template_mfcc_seq is not None:
            score_dtw = self.dtw_scorer.score(user_mfcc_seq, template_mfcc_seq)
        else:
            score_dtw = 50.0
            warnings.warn("未提供MFCC序列，DTW评分使用默认值50分")

        # 3. 声学相似度评分
        if user_audio is not None and template_audio is not None:
            score_acoustic = self.acoustic_scorer.compute_similarity(
                user_audio, template_audio)
        else:
            score_acoustic = 50.0
            warnings.warn("未提供音频信号，声学评分使用默认值50分")

        # 4. 加权融合
        total = (self.emb_weight * score_first +
                 self.dtw_weight * score_dtw +
                 self.acoustic_weight * score_acoustic)
        total = np.clip(total, 0.0, 100.0)

        # 5. 评级
        grade, star = self._get_grade(total)

        return ScoreResult(
            total_score=round(total, 2),
            conf_score=round(score_first, 2),
            dtw_score=round(score_dtw, 2),
            acoustic_score=round(score_acoustic, 2),
            grade=grade,
            star_rating=star,
            detail={
                f"{first_name}_score": round(score_first, 2),
                "dtw_score": round(score_dtw, 2),
                "acoustic_score": round(score_acoustic, 2),
                "weights": {first_name: round(self.emb_weight, 3),
                            "dtw": round(self.dtw_weight, 3),
                            "acoustic": round(self.acoustic_weight, 3)},
                "scoring_mode": "embedding" if first_name == "emb" else "confidence",
                "grade": grade,
            },
        )

    # ==================== 仅嵌入/置信度评分（快速模式） ====================
    def score_embedding_only(self, user_embedding, target_label):
        """快速模式：仅使用嵌入相似度评分 (v2)"""
        sc = self.emb_scorer.score(user_embedding, target_label)
        grade, star = self._get_grade(sc)
        return ScoreResult(total_score=round(sc, 2), conf_score=round(sc, 2),
                           dtw_score=0.0, acoustic_score=0.0,
                           grade=grade, star_rating=star,
                           detail={"mode": "embedding_only"})

    def score_confidence_only(self, softmax_probs, target_class_idx):
        """快速模式：仅使用置信度评分 (v1 回退)"""
        sc = self.conf_scorer.score(softmax_probs, target_class_idx)
        grade, star = self._get_grade(sc)
        return ScoreResult(total_score=round(sc, 2), conf_score=round(sc, 2),
                           dtw_score=0.0, acoustic_score=0.0,
                           grade=grade, star_rating=star,
                           detail={"mode": "confidence_only"})

    # ==================== 完整详细评分 (v2) ====================
    def score_detailed(self, softmax_probs=None, target_class_idx=None,
                       user_embedding=None, target_label=None,
                       user_mfcc_seq=None, template_mfcc_seq=None,
                       user_audio=None, template_audio=None):
        """完整评分（含所有子维度详细信息）"""
        result = self.score(softmax_probs=softmax_probs,
                            target_class_idx=target_class_idx,
                            user_embedding=user_embedding,
                            target_label=target_label,
                            user_mfcc_seq=user_mfcc_seq,
                            template_mfcc_seq=template_mfcc_seq,
                            user_audio=user_audio,
                            template_audio=template_audio)
        detail = dict(result.detail)

        # 嵌入Top-5 或 置信度Top-5
        if detail.get("scoring_mode") == "embedding" and user_embedding is not None:
            detail["emb_top5"] = self.emb_scorer.get_topk_similar(user_embedding, k=5)
        elif softmax_probs is not None:
            detail["conf_top5"] = self.conf_scorer.get_topk_info(softmax_probs, k=5)

        # DTW详细信息
        if user_mfcc_seq is not None and template_mfcc_seq is not None:
            dtw_result = self.dtw_scorer.compute(user_mfcc_seq, template_mfcc_seq)
            detail["dtw_avg_distance"] = round(dtw_result["avg_distance"], 6)
            detail["dtw_path_length"] = len(dtw_result["warping_path"])

        # 声学详细信息
        if user_audio is not None and template_audio is not None:
            acoustic_detail = self.acoustic_scorer.compute_detailed_similarity(
                user_audio, template_audio)
            detail.update(acoustic_detail)

        return ScoreResult(
            total_score=result.total_score, conf_score=result.conf_score,
            dtw_score=result.dtw_score, acoustic_score=result.acoustic_score,
            grade=result.grade, star_rating=result.star_rating, detail=detail)

    # ==================== 评分等级判定 ====================
    @staticmethod
    def _get_grade(score):
        if score >= 90:
            return "优秀 (Excellent!)", "★★★★★"
        elif score >= 75:
            return "良好 (Good)", "★★★★☆"
        elif score >= 60:
            return "一般 (Fair)", "★★★☆☆"
        else:
            return "需努力 (Try Again!)", "★★☆☆☆"

    # ==================== 权重调整 ====================
    def set_weights(self, first_w, dtw_w, acoustic_w):
        """动态调整三维度权重"""
        total = first_w + dtw_w + acoustic_w
        self.emb_weight = first_w / total
        self.dtw_weight = dtw_w / total
        self.acoustic_weight = acoustic_w / total

    # ==================== 标准嵌入管理 (v2) ====================
    def set_standard_embeddings(self, standard_embeddings: dict) -> None:
        """设置标准发音嵌入字典。{label: np.array(128,)} """
        self.emb_scorer.set_standards(standard_embeddings)

    # ==================== 配置信息 ====================
    def get_info(self):
        """返回当前评分器配置"""
        return {
            "scoring_mode": "embedding" if self.use_embedding else "confidence",
            "weights": {"first_dim": round(self.emb_weight, 3),
                        "dtw": round(self.dtw_weight, 3),
                        "acoustic": round(self.acoustic_weight, 3)},
            "dtw_alpha": self.dtw_scorer.alpha,
            "sample_rate": self.sample_rate,
            "formula": "total = {:.0%}*first + {:.0%}*dtw + {:.0%}*acoustic".format(
                self.emb_weight, self.dtw_weight, self.acoustic_weight),
        }


# ====================================================================
# 第五部分：权重寻优工具
# ====================================================================
def optimize_weights(human_scores, conf_scores, dtw_scores, acoustic_scores,
                     grid_points=21):
    """
    网格搜索最优权重组合（使机器评分与人工评分Pearson相关最大）

    :param human_scores: 人工专家评分 [n_samples]
    :param conf_scores: 置信度评分 [n_samples]
    :param dtw_scores: DTW评分 [n_samples]
    :param acoustic_scores: 声学评分 [n_samples]
    :param grid_points: 网格采样点数
    :return: (best_weights, best_corr, all_results)
    """
    from scipy.stats import pearsonr
    hs = np.array(human_scores)
    cs = np.array(conf_scores)
    ds = np.array(dtw_scores)
    as_ = np.array(acoustic_scores)

    best_corr = -1
    best_weights = (0.40, 0.35, 0.25)
    all_results = []

    for w1 in np.linspace(0, 1, grid_points):
        for w2 in np.linspace(0, 1 - w1, grid_points):
            w3 = 1 - w1 - w2
            if w3 < 0:
                continue
            total = w1 * cs + w2 * ds + w3 * as_
            corr, _ = pearsonr(hs, total)
            all_results.append({"weights": (round(w1, 3), round(w2, 3), round(w3, 3)),
                                "pearson_r": round(corr, 4)})
            if corr > best_corr:
                best_corr = corr
                best_weights = (w1, w2, w3)

    print(f"[权重优化] 最优权重: conf={best_weights[0]:.3f}, "
          f"dtw={best_weights[1]:.3f}, acoustic={best_weights[2]:.3f}, "
          f"Pearson r={best_corr:.4f}")
    return best_weights, best_corr, all_results


# ====================================================================
# 第六部分：快速测试
# ====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("发音智能评测打分模块 —— 三维度融合评分测试 (v2)")
    print("=" * 60)
    np.random.seed(42)

    # 模拟数据
    n_classes = 26
    target_idx = 0
    target_label = "A"
    probs = np.ones(n_classes) * 0.01
    probs[target_idx] = 0.75
    probs = probs / probs.sum()

    # 模拟128维嵌入
    user_embedding = np.random.randn(128).astype(np.float64)
    user_embedding /= np.linalg.norm(user_embedding)
    standard_embedding = user_embedding + np.random.randn(128) * 0.3
    standard_embedding /= np.linalg.norm(standard_embedding)
    standard_embeddings = {"A": standard_embedding}

    user_mfcc = np.random.randn(35, 13) * 0.3
    template_mfcc = np.random.randn(30, 13) * 0.3

    sr = 16000
    user_audio = np.sin(2 * np.pi * 200 * np.arange(0, 1.5, 1/sr))
    template_audio = np.sin(2 * np.pi * 200 * np.arange(0, 1.5, 1/sr))

    # 测试1：嵌入模式完整评分 (v2 默认)
    print("\n[测试1] 嵌入相似度 + DTW + 声学 三维度融合评分 (v2)")
    scorer = PronunciationScorer(use_embedding=True,
                                 standard_embeddings=standard_embeddings)
    result = scorer.score(user_embedding=user_embedding, target_label=target_label,
                          user_mfcc_seq=user_mfcc, template_mfcc_seq=template_mfcc,
                          user_audio=user_audio, template_audio=template_audio)
    print(f"  综合评分: {result.total_score:.2f} / 100")
    print(f"  嵌入相似度: {result.conf_score:.2f} | DTW: {result.dtw_score:.2f} | "
          f"声学: {result.acoustic_score:.2f}")
    print(f"  评分模式: {result.detail.get('scoring_mode', 'N/A')}")
    print(f"  评级: {result.grade} | 星级: {result.star_rating}")

    # 测试2：置信度回退模式 (v1)
    print("\n[测试2] 置信度 + DTW + 声学 三维度融合评分 (v1 回退)")
    scorer_v1 = PronunciationScorer(use_embedding=False)
    result_v1 = scorer_v1.score(softmax_probs=probs, target_class_idx=target_idx,
                                user_mfcc_seq=user_mfcc, template_mfcc_seq=template_mfcc,
                                user_audio=user_audio, template_audio=template_audio)
    print(f"  综合评分: {result_v1.total_score:.2f} / 100")
    print(f"  置信度: {result_v1.conf_score:.2f} | DTW: {result_v1.dtw_score:.2f} | "
          f"声学: {result_v1.acoustic_score:.2f}")

    # 测试3：仅嵌入评分（快速模式）
    print("\n[测试3] 仅嵌入相似度评分（快速模式 v2）")
    r3 = scorer.score_embedding_only(user_embedding, target_label)
    print(f"  评分: {r3.total_score:.2f} → {r3.grade}")

    # 测试4：嵌入相似度 Top-K
    print("\n[测试4] 嵌入相似度 Top-K")
    topk = scorer.emb_scorer.get_topk_similar(user_embedding, k=3)
    for label, sim in topk:
        print(f"  {label}: {sim:.2f}")

    # 测试5：权重优化（与维度无关）
    print("\n[测试5] 权重网格搜索优化")
    n = 50
    hs = np.random.uniform(40, 95, n)
    fs = hs + np.random.randn(n) * 8  # first-dim scores (emb or conf)
    ds = hs + np.random.randn(n) * 10
    as_ = hs + np.random.randn(n) * 12
    bw, bc, _ = optimize_weights(hs, fs, ds, as_)
    print(f"  最优权重: first={bw[0]:.3f}, dtw={bw[1]:.3f}, acoustic={bw[2]:.3f}")
    print(f"  Pearson r = {bc:.4f}")

    # 测试6：DTW alpha校准
    print("\n[测试6] DTW alpha校准")
    good = [template_mfcc + np.random.randn(30, 13) * 0.1 for _ in range(10)]
    bad = [template_mfcc + np.random.randn(30, 13) * 2.0 for _ in range(10)]
    tmps = [template_mfcc for _ in range(10)]
    dtw_scorer = DTWScorer(alpha=1.0)
    dtw_scorer.calibrate_alpha(good, bad, tmps)

    print("\n[OK] 所有测试完成！")
