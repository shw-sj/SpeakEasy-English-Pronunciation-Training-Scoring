import numpy as np
from scipy.spatial.distance import cdist
from collections import namedtuple

# ====================== DTW 对齐结果数据结构 ======================
DTWResult = namedtuple("DTWResult", [
    "dtw_distance",      # 规整路径上的累计欧氏距离
    "avg_distance",      # 规整路径上的平均欧氏距离（总距离 / 路径长度）
    "score",             # 发音评分：max(0, 100 - α × avg_distance)
    "warping_path",      # 规整路径：[(i1,j1), (i2,j2), ...]
    "cost_matrix",       # 累计代价矩阵（可选保存，用于可视化）
])

# ====================== 1. DTW 核心算法（带窗口约束） ======================
class DTWMatcher:
    """
    动态时间规整(DTW)发音评分器

    核心思路：
        - 将用户语音的 MFCC 序列与标准模板的 MFCC 序列对齐
        - 即使语速不同（快/慢），DTW 也能找到最优的时间对齐路径
        - 对齐后计算平均欧氏距离 → 距离越小发音越接近标准 → 分数越高

    公式：
        score_dtw = max(0, 100 - α × avg_distance)

    参数：
        alpha:        缩放系数，通过实验数据校准（默认 1.0）
        window:       规整窗口宽度（Sakoe-Chiba 约束），None 表示无约束
                      窗口限制后：|i - j| ≤ window，加速计算并防止病态对齐
        distance_metric: 局部距离度量方式，默认 'euclidean'
    """

    def __init__(self, alpha=1.0, window=None, distance_metric='euclidean'):
        """
        初始化 DTW 匹配器

        :param alpha: 评分缩放系数，alpha 越大，同样距离下分数越低
        :param window: Sakoe-Chiba 窗口宽度（int），None 表示全局对齐无约束
        :param distance_metric: 局部距离度量 'euclidean' / 'cosine' / 'cityblock'
        """
        self.alpha = alpha
        self.window = window
        self.distance_metric = distance_metric

    # ==================== DTW 核心：代价矩阵 + 规整路径 ====================
    def compute(self, user_mfcc, template_mfcc, return_cost_matrix=False):
        """
        对用户 MFCC 序列与标准模板 MFCC 序列进行 DTW 对齐并评分

        :param user_mfcc:     用户语音 MFCC 特征，shape = [n_frames_user, n_features]
        :param template_mfcc: 标准模板 MFCC 特征，shape = [n_frames_template, n_features]
        :param return_cost_matrix: 是否返回累计代价矩阵（用于可视化）
        :return: DTWResult 命名元组
        """
        # 1. 输入校验
        user_mfcc = np.atleast_2d(user_mfcc).astype(np.float64)
        template_mfcc = np.atleast_2d(template_mfcc).astype(np.float64)

        if user_mfcc.shape[1] != template_mfcc.shape[1]:
            raise ValueError(
                f"MFCC 特征维度不一致：用户={user_mfcc.shape[1]}, 模板={template_mfcc.shape[1]}"
            )

        n_user = user_mfcc.shape[0]          # 用户语音帧数 M
        n_template = template_mfcc.shape[0]  # 模板语音帧数 N

        # 边缘情况：空序列
        if n_user == 0 or n_template == 0:
            return DTWResult(
                dtw_distance=float('inf'),
                avg_distance=float('inf'),
                score=0.0,
                warping_path=[],
                cost_matrix=None
            )

        # 2. 计算局部距离矩阵 dist_mat[i, j] = ||user[i] - template[j]||
        dist_mat = cdist(user_mfcc, template_mfcc, metric=self.distance_metric)
        # shape = [M, N]

        # 3. 构建累计代价矩阵（动态规划）
        cost_matrix = self._build_cost_matrix(dist_mat, n_user, n_template)

        # 4. 反向回溯最优规整路径
        warping_path = self._backtrack(cost_matrix, n_user, n_template)

        # 5. 沿规整路径计算评分
        dtw_dist = cost_matrix[-1, -1]                         # 累计总距离
        avg_dist = dtw_dist / len(warping_path)                # 平均每步距离
        score = self._distance_to_score(avg_dist)              # 映射为 0~100 分

        return DTWResult(
            dtw_distance=dtw_dist,
            avg_distance=avg_dist,
            score=score,
            warping_path=warping_path,
            cost_matrix=cost_matrix if return_cost_matrix else None
        )

    # ==================== 构建累计代价矩阵 ====================
    def _build_cost_matrix(self, dist_mat, n_user, n_template):
        """
        动态规划构建累计代价矩阵

        DP 递推式：
            cost[i, j] = dist[i, j] + min(
                cost[i-1, j],      # 插入（用户多一帧）
                cost[i, j-1],      # 删除（模板多一帧）
                cost[i-1, j-1]     # 匹配（同步对齐）
            )

        边界条件：
            cost[0, 0] = dist[0, 0]
            cost[i, 0] = Σ_{k=0}^{i} dist[k, 0]   (只能从上方来)
            cost[0, j] = Σ_{k=0}^{j} dist[0, k]   (只能从左方来)
        """
        cost = np.full((n_user, n_template), np.inf)

        # 边界初始化
        cost[0, 0] = dist_mat[0, 0]

        # 第一列（只能从上方来：用户帧依次对齐到模板第一帧）
        for i in range(1, n_user):
            cost[i, 0] = cost[i - 1, 0] + dist_mat[i, 0]

        # 第一行（只能从左方来：模板帧依次对齐到用户第一帧）
        for j in range(1, n_template):
            cost[0, j] = cost[0, j - 1] + dist_mat[0, j]

        # 主体：逐行填充（按列填充亦可）
        for i in range(1, n_user):
            # Sakoe-Chiba 窗口约束：只计算窗口内的点，其余保持 inf
            if self.window is not None:
                j_start = max(1, i - self.window)
                j_end = min(n_template, i + self.window + 1)
            else:
                j_start = 1
                j_end = n_template

            for j in range(j_start, j_end):
                cost[i, j] = dist_mat[i, j] + min(
                    cost[i - 1, j],      # 上方：用户多一帧（插入）
                    cost[i, j - 1],      # 左方：模板多一帧（删除）
                    cost[i - 1, j - 1]   # 左上：帧同步（匹配）
                )

        return cost

    # ==================== 回溯最优规整路径 ====================
    def _backtrack(self, cost_matrix, n_user, n_template):
        """
        从 cost_matrix 右下角 (M-1, N-1) 回溯到左上角 (0, 0)

        每一步选择局部最优方向（min 回溯），得到对齐路径：
            warping_path = [(0,0), ..., (M-1, N-1)]
        """
        path = []
        i, j = n_user - 1, n_template - 1
        path.append((i, j))

        while i > 0 or j > 0:
            if i == 0:
                # 只能向左走
                j -= 1
            elif j == 0:
                # 只能向上走
                i -= 1
            else:
                # 选择三个方向中代价最小的
                directions = [
                    (i - 1, j),      # 上
                    (i, j - 1),      # 左
                    (i - 1, j - 1)   # 左上
                ]
                costs = [cost_matrix[d[0], d[1]] for d in directions]
                best_idx = int(np.argmin(costs))
                i, j = directions[best_idx]

            path.append((i, j))

        path.reverse()  # 从 (0,0) → (M-1, N-1)
        return path

    # ==================== 距离 → 评分映射 ====================
    def _distance_to_score(self, avg_distance):
        """
        公式：score = max(0, 100 - α × avg_distance)

        :param avg_distance: 规整路径上的平均欧氏距离
        :return: 评分 [0, 100]
        """
        score = 100.0 - self.alpha * avg_distance
        return max(0.0, score)

    # ==================== 便捷接口：直接评分 ====================
    def score(self, user_mfcc, template_mfcc):
        """
        快捷评分接口，只返回分数

        :param user_mfcc:     用户语音 MFCC，shape = [n_frames_user, n_features]
        :param template_mfcc: 标准模板 MFCC，shape = [n_frames_template, n_features]
        :return: 评分 float ∈ [0, 100]
        """
        result = self.compute(user_mfcc, template_mfcc)
        return result.score


# ====================== 2. 批量评分工具 ======================
def batch_score(matcher, user_mfcc_list, template_mfcc_list):
    """
    批量对多个用户发音进行 DTW 评分

    :param matcher:            DTWMatcher 实例
    :param user_mfcc_list:    用户 MFCC 列表，每个 shape = [n_frames, n_features]
    :param template_mfcc_list: 标准模板 MFCC 列表（一一对应），每个 shape = [n_frames, n_features]
    :return: scores 列表
    """
    scores = []
    for user_mfcc, template_mfcc in zip(user_mfcc_list, template_mfcc_list):
        scores.append(matcher.score(user_mfcc, template_mfcc))
    return scores


# ====================== 3. alpha 系数自动校准 ======================
def calibrate_alpha(good_samples, bad_samples, templates, target_good_score=85.0, target_bad_score=40.0):
    """
    通过实验数据自动校准 alpha 缩放系数

    思路：
        好发音 → 期望高分（如 85 分）
        差发音 → 期望低分（如 40 分）
        求解 alpha 使得两端都能得到合理分数

    :param good_samples:      好发音的 MFCC 列表
    :param bad_samples:       差发音的 MFCC 列表
    :param templates:         对应的标准模板 MFCC 列表
    :param target_good_score: 好发音的目标分数（默认 85）
    :param target_bad_score:  差发音的目标分数（默认 40）
    :return: 校准后的 alpha 值
    """
    # 先用 alpha=1.0 计算原始距离
    temp_matcher = DTWMatcher(alpha=1.0, window=None)
    good_distances = []
    bad_distances = []

    for sample, template in zip(good_samples, templates):
        result = temp_matcher.compute(sample, template)
        good_distances.append(result.avg_distance)

    for sample, template in zip(bad_samples, templates):
        result = temp_matcher.compute(sample, template)
        bad_distances.append(result.avg_distance)

    avg_good_dist = np.mean(good_distances)
    avg_bad_dist = np.mean(bad_distances)

    # 从两组数据解 alpha
    # good: 100 - alpha * avg_good_dist = target_good_score  →  alpha_good = (100 - target_good) / avg_good_dist
    # bad:  100 - alpha * avg_bad_dist  = target_bad_score   →  alpha_bad  = (100 - target_bad)  / avg_bad_dist
    alpha_good = (100.0 - target_good_score) / (avg_good_dist + 1e-8)
    alpha_bad = (100.0 - target_bad_score) / (avg_bad_dist + 1e-8)

    # 取平均作为最终 alpha
    alpha_calibrated = (alpha_good + alpha_bad) / 2.0
    alpha_calibrated = max(alpha_calibrated, 1e-6)  # 防止为 0 或负

    print(f"=== Alpha 校准结果 ===")
    print(f"好发音平均距离：{avg_good_dist:.4f} → alpha_good = {alpha_good:.4f}")
    print(f"差发音平均距离：{avg_bad_dist:.4f} → alpha_bad  = {alpha_bad:.4f}")
    print(f"校准后 alpha = {alpha_calibrated:.4f}")
    print(f"验证 — 好发音评分：{100 - alpha_calibrated * avg_good_dist:.2f}")
    print(f"验证 — 差发音评分：{100 - alpha_calibrated * avg_bad_dist:.2f}")

    return alpha_calibrated


# ====================== 4. 规整路径可视化（可选调试用） ======================
def plot_warping_path(result, user_label="用户语音", template_label="标准模板",
                      title="DTW 规整路径", save_path=None):
    """
    可视化 DTW 对齐路径（调试/分析用）

    :param result:         DTWResult（需 compute 时设置 return_cost_matrix=True）
    :param user_label:     用户语音标签
    :param template_label: 标准模板标签
    :param title:          图表标题
    :param save_path:      保存路径，None 则 plt.show()
    """
    import matplotlib.pyplot as plt

    if result.cost_matrix is None:
        raise ValueError("需要 DTWResult 包含 cost_matrix，请在 compute() 中设置 return_cost_matrix=True")
    if len(result.warping_path) == 0:
        print("规整路径为空，跳过绘图")
        return

    path_i, path_j = zip(*result.warping_path)

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：代价矩阵 + 规整路径
    ax1 = axes[0]
    im = ax1.imshow(result.cost_matrix, origin='lower', aspect='auto', cmap='viridis')
    ax1.plot(path_j, path_i, 'r-', linewidth=2, label='规整路径')
    ax1.set_xlabel(template_label + " (帧)")
    ax1.set_ylabel(user_label + " (帧)")
    ax1.set_title(title)
    ax1.legend()
    plt.colorbar(im, ax=ax1, label='累计代价')

    # 右图：帧对齐关系（斜率 = 1 表示语速完全一致）
    ax2 = axes[1]
    ax2.plot(path_j, path_i, 'b.-', markersize=2, linewidth=1)
    ax2.plot([0, max(path_j)], [0, max(path_j)], 'k--', alpha=0.3, label='语速一致参考线')
    ax2.set_xlabel(template_label + " (帧)")
    ax2.set_ylabel(user_label + " (帧)")
    ax2.set_title("帧对齐关系图")
    ax2.legend()
    ax2.set_aspect('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"规整路径图已保存至：{save_path}")
    else:
        plt.show()


# ====================== 5. 快速测试 / 示例 ======================
if __name__ == "__main__":
    print("=" * 60)
    print("DTW 发音评分匹配器 — 测试")
    print("=" * 60)

    # ---- 模拟数据：MFCC 特征（13维 × 不定长帧） ----
    np.random.seed(42)

    # 标准模板：30 帧 × 13 维
    template = np.random.randn(30, 13)

    # 好发音：和模板很接近（加入少量噪声）
    good_pronunciation = template + np.random.randn(30, 13) * 0.1

    # 差发音：和模板差异较大（加入大量噪声）
    bad_pronunciation = template + np.random.randn(30, 13) * 1.5

    # 语速不同但发音标准：同样是模板，但拉伸到 45 帧（模拟慢速朗读）
    # 通过线性插值模拟帧数不一致
    indices = np.linspace(0, 29, 45)
    slow_pronunciation = np.array([np.interp(indices, np.arange(30), template[:, d])
                                    for d in range(13)]).T
    slow_pronunciation += np.random.randn(45, 13) * 0.1  # 加少量噪声

    # ---- 测试 1：默认 alpha=1.0 ----
    print("\n[测试 1] 默认 alpha=1.0")
    matcher = DTWMatcher(alpha=1.0)

    r_good = matcher.compute(good_pronunciation, template)
    r_bad = matcher.compute(bad_pronunciation, template)
    r_slow = matcher.compute(slow_pronunciation, template)

    print(f"  好发音评分：{r_good.score:.2f}  (avg_dist={r_good.avg_distance:.4f})")
    print(f"  差发音评分：{r_bad.score:.2f}  (avg_dist={r_bad.avg_distance:.4f})")
    print(f"  慢速好发音：{r_slow.score:.2f}  (avg_dist={r_slow.avg_distance:.4f})")

    # ---- 测试 2：alpha 自动校准 ----
    print("\n[测试 2] alpha 自动校准")
    # 构造多组好/差发音样本
    good_samples = [template + np.random.randn(30, 13) * 0.1 for _ in range(10)]
    bad_samples = [template + np.random.randn(30, 13) * 2.0 for _ in range(10)]
    templates_list = [template for _ in range(10)]

    alpha_cal = calibrate_alpha(good_samples, bad_samples, templates_list,
                                 target_good_score=85.0, target_bad_score=40.0)

    # ---- 测试 3：使用校准后的 alpha 重新评分 ----
    print("\n[测试 3] 使用校准后 alpha 评分")
    matcher_cal = DTWMatcher(alpha=alpha_cal)

    r_good_cal = matcher_cal.compute(good_pronunciation, template)
    r_bad_cal = matcher_cal.compute(bad_pronunciation, template)
    r_slow_cal = matcher_cal.compute(slow_pronunciation, template)

    print(f"  好发音评分：{r_good_cal.score:.2f}")
    print(f"  差发音评分：{r_bad_cal.score:.2f}")
    print(f"  慢速好发音：{r_slow_cal.score:.2f}")

    # ---- 测试 4：批量评分 ----
    print("\n[测试 4] 批量评分")
    user_list = [good_pronunciation, bad_pronunciation, slow_pronunciation]
    tmpl_list = [template, template, template]
    scores = batch_score(matcher_cal, user_list, tmpl_list)
    for i, s in enumerate(scores):
        print(f"  样本 {i+1} 评分：{s:.2f}")

    # ---- 测试 5：带窗口约束的 DTW ----
    print("\n[测试 5] 带 Sakoe-Chiba 窗口约束（window=10）")
    matcher_win = DTWMatcher(alpha=alpha_cal, window=10)
    r_win = matcher_win.compute(slow_pronunciation, template, return_cost_matrix=True)
    print(f"  慢速好发音（窗口约束）：{r_win.score:.2f}  (avg_dist={r_win.avg_distance:.4f})")
    print(f"  规整路径长度：{len(r_win.warping_path)}")

    print("\n✅ 所有测试完成！")
