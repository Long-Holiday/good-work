"""DiffPeak-SAM 的提示奖励与图像级二值指标。

奖励定义（与设计文档一致）
--------------------------
.. code-block:: text

    R = w_iou * IoU + w_cldice * clDice + w_path * R_path
        - point_cost * (K+ + K-) / point_reference

其中 ``K+``、``K-`` 分别是正、负提示点数量；``point_reference`` 只是归一化
常数（默认 32），不是点数上限。质量项的默认权重之和为 1
（``0.6 + 0.4 + 0.0``）。

空图规则（训练、奖励与推理保持一致）
------------------------------------
* 预测与标注都为空：IoU = F1 = clDice = 1，奖励为 ``1 - 提示成本``；
* 仅一方为空：IoU = F1 = clDice = 0，质量奖励为 0；
* 空的标注/预测不参与路径项。

路径项（可选，默认关闭）
------------------------
``path_weight > 0`` 时才计算。实现要点：

1. 在两侧二值图各自的骨架上建立 8 邻接无向像素图，边权为 1 或 ``sqrt(2)``；
2. 在源图（target 或 prediction）的**同一连通组件**内随机抽取端点对，
   避免抽到图上本来就不连通的无效对；
3. 用 KDTree 把两个端点分别匹配到另一侧骨架点，匹配半径由
   ``path_match_radius`` 控制，并保证同一次匹配中目标点不被复用；
4. 用 Dijkstra 测地距离作为路径长度代理，按
   ``max(0, 1 - |d_pred - d_target| / (d_target + eps))`` 计分；
5. 双向计算（target→prediction、prediction→target），对请求的采样对取平均；
   端点无法匹配或对应路径断开时按零分计入。无法请求任何采样对时，路径项
   无效，剩余质量权重重新归一化。

该路径项只是**抽样代理奖励**，不是正式 APLS。它衡量的是“标注所表达的
道路连通性”，不能区分立交跨越与平面交叉；论文/最终评估中的 APLS 仍应
使用正式实现。

所有随机采样使用 ``RewardConfig.seed`` 初始化的局部 ``numpy.random.Generator``，
因此同一输入与配置下的结果可复现。路径图与 Dijkstra 只在
``path_weight > 0`` 时构建，且采样数受 ``path_samples`` 限制，以控制 CPU 开销。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

__all__ = ["RewardConfig", "compute_reward", "image_confusion_metrics"]


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardConfig:
    """奖励函数配置。

    Attributes
    ----------
    iou_weight:
        区域 IoU 奖励权重，默认 0.6。
    cldice_weight:
        中心线 Dice（clDice）奖励权重，默认 0.4。
    path_weight:
        路径代理奖励权重，默认 0.0（关闭路径项以避免额外 CPU 开销）。
    point_cost:
        提示成本系数，默认 0.05。
    point_reference:
        提示数量归一化常数 ``K_ref``，默认 32；不是点数上限。
    path_samples:
        每个方向的路径采样对数，默认 32。
    path_match_radius:
        端点跨图 KDTree 匹配的最大像素距离，默认 8。
    seed:
        路径采样随机种子，默认 0。固定种子保证奖励可复现。
    """

    iou_weight: float = 0.6
    cldice_weight: float = 0.4
    path_weight: float = 0.0
    point_cost: float = 0.05
    point_reference: int = 32
    path_samples: int = 32
    path_match_radius: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.iou_weight < 0:
            raise ValueError("iou_weight 必须非负")
        if self.cldice_weight < 0:
            raise ValueError("cldice_weight 必须非负")
        if self.path_weight < 0:
            raise ValueError("path_weight 必须非负")
        if self.point_cost < 0:
            raise ValueError("point_cost 必须非负")
        if self.point_reference <= 0:
            raise ValueError("point_reference 必须为正整数")
        if self.path_samples <= 0:
            raise ValueError("path_samples 必须为正整数")
        if self.path_match_radius < 0:
            raise ValueError("path_match_radius 必须非负")


# ---------------------------------------------------------------------------
# 基础校验与图像级混淆指标
# ---------------------------------------------------------------------------


def _as_binary_image(array: np.ndarray, name: str) -> np.ndarray:
    """校验输入为 ``H x W`` 的二值图并返回 bool 数组（不复制原数组）。"""
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{name} 必须是二维 H x W，实际形状为 {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        # 拒绝把 logits/概率图误当作二值掩码传入，避免“所有非零像素都是道路”。
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} 包含 NaN 或 Inf")
        if not np.all((array == 0) | (array == 1)):
            raise ValueError(f"{name} 是浮点数组时只能包含 0/1")
    return array.astype(bool, copy=False)


def _confusion_counts(
    prediction: np.ndarray,
    target: np.ndarray,
) -> Tuple[int, int, int, int]:
    """返回整张图的 (TP, FP, FN, TN) 像素计数。"""
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    tn = int(prediction.size - tp - fp - fn)
    return tp, fp, fn, tn


def image_confusion_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:
    """计算单张二值图的全局（global / micro）混淆指标。

    指标直接由整张图的 TP/FP/FN 汇总得到，因此与“每个像素等权”的
    global 口径一致。具体实现为：先累加混淆计数，再计算 IoU 与 F1。

    Parameters
    ----------
    prediction:
        预测二值图，``H x W``，bool 或 0/1 数值数组。
    target:
        标注二值图，形状与 prediction 相同。

    Returns
    -------
    dict
        包含 ``tp``、``fp``、``fn``、``tn``、``iou``、``f1``。

    Notes
    -----
    关于 micro 与 macro
        - **global / micro**（本函数）：把一张图（或多张图）的 TP/FP/FN
          先汇总，再算一次 IoU/F1。像素多的图像自然占更大权重。
        - **macro**：先对每张图（或每个类别）分别计算 IoU/F1，再做**算术
          平均**。每张图权重相同，与图像大小无关；因此少量道路的小图、
          甚至空图，都可能显著影响 macro 结果。
        - 若需要数据集级 macro，请对每张图调用本函数，再平均各图的
          ``iou`` / ``f1``；若需要数据集级 micro，请先累加所有图的
          TP/FP/FN，再计算指标。两者不可混用。

    空图约定
        两者都为空时 ``iou = f1 = 1``；仅一方为空时 ``iou = f1 = 0``。
        该约定在 ``compute_reward`` 中保持一致，也会影响 macro 平均。
    """
    prediction = _as_binary_image(prediction, "prediction")
    target = _as_binary_image(target, "target")
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction 与 target 形状必须一致："
            f"{prediction.shape} vs {target.shape}"
        )

    tp, fp, fn, tn = _confusion_counts(prediction, target)
    union = tp + fp + fn
    f1_denominator = 2 * tp + fp + fn

    iou = 1.0 if union == 0 else tp / union
    f1 = 1.0 if f1_denominator == 0 else (2 * tp) / f1_denominator
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "iou": float(iou),
        "f1": float(f1),
    }


def _binary_cldice(prediction: np.ndarray, target: np.ndarray) -> float:
    """计算二值 clDice；调用前请保证两者均非空。"""
    pred_skeleton = skeletonize(prediction)
    target_skeleton = skeletonize(target)

    pred_skeleton_size = int(np.count_nonzero(pred_skeleton))
    target_skeleton_size = int(np.count_nonzero(target_skeleton))
    if pred_skeleton_size == 0 or target_skeleton_size == 0:
        return 0.0

    # Tprec：预测骨架中有多少落在目标区域内。
    tprec = np.count_nonzero(pred_skeleton & target) / pred_skeleton_size
    # Tsens：目标骨架中有多少被预测区域覆盖。
    tsens = np.count_nonzero(target_skeleton & prediction) / target_skeleton_size
    if tprec + tsens == 0.0:
        return 0.0
    return float(2.0 * tprec * tsens / (tprec + tsens))


# ---------------------------------------------------------------------------
# 路径代理奖励
# ---------------------------------------------------------------------------

# 只枚举 4 个方向即可覆盖无向 8 邻接的全部边：东、南、东南、西南。
_NEIGHBOR_OFFSETS: Tuple[Tuple[int, int, float], ...] = (
    (0, 1, 1.0),
    (1, 0, 1.0),
    (1, 1, float(np.sqrt(2.0))),
    (1, -1, float(np.sqrt(2.0))),
)


def _build_skeleton_graph(
    skeleton: np.ndarray,
) -> Tuple[np.ndarray, Optional[csr_matrix]]:
    """在骨架像素上构建 8 邻接无向图。

    Returns
    -------
    coords:
        ``(N, 2)`` 的 ``(row, col)`` 坐标，顺序与图节点编号一致。
    graph:
        ``N x N`` 的 CSR 稀疏图；骨架为空时返回 ``None``。
        每条无向边只存一次，Dijkstra/连通组件均以 ``directed=False`` 使用。
    """
    coords = np.argwhere(skeleton)
    num_nodes = int(coords.shape[0])
    if num_nodes == 0:
        return coords, None

    height, width = skeleton.shape
    node_index = np.full(skeleton.shape, -1, dtype=np.int64)
    node_index[coords[:, 0], coords[:, 1]] = np.arange(num_nodes, dtype=np.int64)

    rows = []
    cols = []
    weights = []
    for delta_row, delta_col, weight in _NEIGHBOR_OFFSETS:
        neighbor_row = coords[:, 0] + delta_row
        neighbor_col = coords[:, 1] + delta_col
        in_bounds = (
            (neighbor_row >= 0)
            & (neighbor_row < height)
            & (neighbor_col >= 0)
            & (neighbor_col < width)
        )

        neighbor_index = np.full(num_nodes, -1, dtype=np.int64)
        neighbor_index[in_bounds] = node_index[
            neighbor_row[in_bounds], neighbor_col[in_bounds]
        ]
        edge_mask = neighbor_index >= 0
        if bool(edge_mask.any()):
            rows.append(np.flatnonzero(edge_mask).astype(np.int64))
            cols.append(neighbor_index[edge_mask])
            weights.append(
                np.full(int(edge_mask.sum()), weight, dtype=np.float64)
            )

    if not rows:
        # 全部骨架像素都是孤立点。
        return coords, csr_matrix((num_nodes, num_nodes), dtype=np.float64)

    graph = csr_matrix(
        (
            np.concatenate(weights),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(num_nodes, num_nodes),
    )
    return coords, graph


def _sample_endpoint_pairs(
    labels: np.ndarray,
    num_components: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """在同一连通组件内随机抽取端点对。

    只在节点数 ``>= 2`` 的组件内采样，因此抽到的两个端点在图上是连通的，
    不会产生“本来就没有路径”的无效对。组件按均匀概率选择，组件内节点
    也均匀选择，避免超长主干完全主导采样。
    """
    if count <= 0:
        return np.empty((0, 2), dtype=np.int64)

    sizes = np.bincount(labels, minlength=num_components)
    valid_components = np.flatnonzero(sizes >= 2)
    if valid_components.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    # 一次性把节点按组件分组，避免每次采样都做全量扫描。
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    starts = np.searchsorted(sorted_labels, np.arange(num_components), side="left")
    ends = np.searchsorted(sorted_labels, np.arange(num_components), side="right")

    pairs = np.empty((count, 2), dtype=np.int64)
    for i in range(count):
        component = int(valid_components[rng.integers(valid_components.size)])
        component_nodes = order[starts[component] : ends[component]]
        sampled = rng.choice(component_nodes, size=2, replace=False)
        pairs[i, 0] = int(sampled[0])
        pairs[i, 1] = int(sampled[1])
    return pairs


def _match_endpoints(
    points: np.ndarray,
    tree: cKDTree,
    radius: float,
    used: np.ndarray,
) -> np.ndarray:
    """把端点匹配到另一侧骨架点，距离超过半径或不唯一时标记为 -1。

    每个端点最多查看最近的 8 个候选点，并跳过已被本次匹配占用的点，
    从而保证同一次方向计算中匹配目标不被复用。
    """
    num_points = int(points.shape[0])
    matched = np.full(num_points, -1, dtype=np.int64)
    if num_points == 0 or tree.n == 0:
        return matched

    k = min(8, int(tree.n))
    distances, indices = tree.query(points, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    for i in range(num_points):
        for j in range(k):
            distance = float(distances[i, j])
            candidate = int(indices[i, j])
            if distance <= radius and not used[candidate]:
                matched[i] = candidate
                used[candidate] = True
                break
    return matched


def _directed_path_score(
    coords_a: np.ndarray,
    graph_a: csr_matrix,
    labels_a: np.ndarray,
    num_components_a: int,
    coords_b: np.ndarray,
    graph_b: csr_matrix,
    labels_b: np.ndarray,
    config: RewardConfig,
    rng: np.random.Generator,
    target_is_source: bool,
) -> Tuple[float, int, int]:
    """计算一个方向的抽样路径得分。

    Parameters
    ----------
    target_is_source:
        ``True`` 表示 ``a`` 侧是真实标注（target→prediction），
        ``False`` 表示 ``b`` 侧是真实标注（prediction→target）。设计文档的
        相对误差始终用真实标注一侧的路径长度 ``d_target`` 归一化。

    Returns
    -------
    score_sum:
        该方向所有采样对的得分之和（每个采样对得分在 [0, 1]）。
    matched_count:
        端点成功匹配、且两侧路径均存在的采样对数量。
    requested_count:
        该方向请求的采样对数量（作为分母，未匹配对按 0 分计）。
    """
    pairs = _sample_endpoint_pairs(
        labels_a, num_components_a, config.path_samples, rng
    )
    requested = int(pairs.shape[0])
    if requested == 0:
        return 0.0, 0, 0

    # 相同源节点跨路径复用同一个匹配，不能每次消耗新的目标节点，
    # 否则完全相同的短路网也会因重复抽样而得不到满分。
    unique_nodes, inverse = np.unique(pairs.reshape(-1), return_inverse=True)
    endpoints_a = coords_a[unique_nodes]
    tree_b = cKDTree(coords_b)
    used_b = np.zeros(coords_b.shape[0], dtype=bool)
    unique_matches = _match_endpoints(
        endpoints_a,
        tree_b,
        radius=float(config.path_match_radius),
        used=used_b,
    )
    matched = unique_matches[inverse]

    matched_a_source = matched[0::2]
    matched_b_source = matched[1::2]
    valid = (matched_a_source >= 0) & (matched_b_source >= 0)

    # 两个端点还必须落在预测图的同一个连通组件中，才存在可比较的路径。
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return 0.0, 0, requested
    same_component = (
        labels_b[matched_a_source[valid_indices]]
        == labels_b[matched_b_source[valid_indices]]
    )
    valid_indices = valid_indices[same_component]
    if valid_indices.size == 0:
        return 0.0, 0, requested

    valid_pairs = pairs[valid_indices]
    target_sources = matched_a_source[valid_indices]
    target_targets = matched_b_source[valid_indices]

    # 源图测地距离：端点来自同一组件，理论上有限。
    distance_a_matrix = dijkstra(
        graph_a, directed=False, indices=valid_pairs[:, 0]
    )
    distance_a = distance_a_matrix[
        np.arange(valid_indices.size), valid_pairs[:, 1]
    ]

    # 另一侧图测地距离：匹配后的两个端点同属一个组件。
    distance_b_matrix = dijkstra(
        graph_b, directed=False, indices=target_sources
    )
    distance_b = distance_b_matrix[
        np.arange(valid_indices.size), target_targets
    ]

    usable = (
        np.isfinite(distance_a)
        & np.isfinite(distance_b)
        & (distance_a > 0.0)
        & (distance_b > 0.0)
    )
    score_sum = 0.0
    matched_count = int(np.count_nonzero(usable))
    if matched_count > 0:
        # 设计文档规定用真实标注一侧的路径长度归一化误差：
        # target→prediction 时它是 distance_a，prediction→target 时它是 distance_b。
        target_distance = (
            distance_a[usable] if target_is_source else distance_b[usable]
        )
        relative_error = np.abs(distance_b[usable] - distance_a[usable]) / (
            target_distance + 1e-6
        )
        scores = np.clip(1.0 - relative_error, 0.0, None)
        score_sum = float(scores.sum())

    return score_sum, matched_count, requested


def _bidirectional_path_reward(
    prediction: np.ndarray,
    target: np.ndarray,
    config: RewardConfig,
) -> Tuple[float, int, int]:
    """双向骨架路径代理奖励。

    Returns
    -------
    path_score:
        所有有效方向/采样对的平均得分；无有效路径对时为 0。
    matched_count:
        有效路径对总数。
    requested_count:
        两个方向请求的采样对总数。
    """
    prediction_skeleton = skeletonize(prediction)
    target_skeleton = skeletonize(target)

    coords_pred, graph_pred = _build_skeleton_graph(prediction_skeleton)
    coords_target, graph_target = _build_skeleton_graph(target_skeleton)
    if graph_pred is None or graph_target is None:
        return 0.0, 0, 0

    num_components_pred, labels_pred = connected_components(
        graph_pred, directed=False
    )
    num_components_target, labels_target = connected_components(
        graph_target, directed=False
    )

    # 每个奖励调用使用独立的固定种子生成器，保证可复现且不污染全局随机状态。
    rng = np.random.default_rng(config.seed)

    score_sum = 0.0
    matched_count = 0
    requested_count = 0

    # 方向 1：在标注骨架上采样，匹配到预测骨架。
    directional = _directed_path_score(
        coords_target,
        graph_target,
        labels_target,
        int(num_components_target),
        coords_pred,
        graph_pred,
        labels_pred,
        config,
        rng,
        target_is_source=True,
    )
    score_sum += directional[0]
    matched_count += directional[1]
    requested_count += directional[2]

    # 方向 2：在预测骨架上采样，匹配到标注骨架。
    directional = _directed_path_score(
        coords_pred,
        graph_pred,
        labels_pred,
        int(num_components_pred),
        coords_target,
        graph_target,
        labels_target,
        config,
        rng,
        target_is_source=False,
    )
    score_sum += directional[0]
    matched_count += directional[1]
    requested_count += directional[2]

    if requested_count == 0 or matched_count == 0:
        return 0.0, 0, requested_count
    return score_sum / requested_count, matched_count, requested_count


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------


def compute_reward(
    prediction: np.ndarray,
    target: np.ndarray,
    point_count: int,
    config: RewardConfig,
) -> Dict[str, float]:
    """计算一条提示布局对应的分割/拓扑/提示成本奖励。

    Parameters
    ----------
    prediction:
        SAM 输出的二值道路掩码，``H x W``，bool 或 0/1 数组。
    target:
        训练集二值道路标注，形状与 prediction 相同。
    point_count:
        提示点总数 ``K+ + K-``。只用于提示成本项，不限制点数。
    config:
        奖励权重与路径采样配置。

    Returns
    -------
    dict
        至少包含以下键：

        ``reward``
            标量奖励。
        ``iou`` / ``f1``
            图像级 global 混淆指标（见 :func:`image_confusion_metrics`）。
        ``cldice``
            二值骨架 Dice。
        ``path``
            路径代理得分；未启用或无效时为 0.0。

        另含诊断字段 ``path_valid``、``path_matched``、``path_requested``、
        ``point_penalty``，便于记录训练日志。

    Notes
    -----
    权重规则：

    * 路径项有效时：按 ``iou_weight``、``cldice_weight``、``path_weight``
      直接加权；
    * 路径项未启用或没有可请求的采样对时：把剩余的 ``iou_weight`` 与
      ``cldice_weight`` 归一到总和 1（默认配置下原本就是 1），避免因为
      路径项缺失而整体缩小质量奖励；
    * 提示成本始终扣除。

    路径项是抽样代理而非正式 APLS；详见模块文档。
    """
    if not isinstance(config, RewardConfig):
        raise TypeError("config 必须是 RewardConfig 实例")
    if isinstance(point_count, bool) or not isinstance(
        point_count, (int, np.integer)
    ):
        raise TypeError("point_count 必须是整数")
    point_count = int(point_count)
    if point_count < 0:
        raise ValueError("point_count 必须非负")

    prediction = _as_binary_image(prediction, "prediction")
    target = _as_binary_image(target, "target")
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction 与 target 形状必须一致："
            f"{prediction.shape} vs {target.shape}"
        )

    tp, fp, fn, _ = _confusion_counts(prediction, target)
    union = tp + fp + fn
    f1_denominator = 2 * tp + fp + fn

    # ---------- 空图约定 ----------
    if union == 0:
        # 两者都为空：正确空预测。
        iou = 1.0
        f1 = 1.0
        cldice = 1.0
    elif tp == 0 and (fp == 0 or fn == 0):
        # 仅一方为空（prediction 全空或 target 全空）。
        iou = 0.0
        f1 = 0.0
        cldice = 0.0
    else:
        iou = tp / union
        f1 = (2 * tp) / f1_denominator
        cldice = _binary_cldice(prediction, target)

    point_penalty = (
        float(config.point_cost) * point_count / float(config.point_reference)
    )

    # ---------- 路径项 ----------
    path_valid = False
    path_score = 0.0
    path_matched = 0
    path_requested = 0

    if config.path_weight > 0.0 and prediction.any() and target.any():
        path_score, path_matched, path_requested = _bidirectional_path_reward(
            prediction, target, config
        )
        # 源图存在可采样路径即应计分；匹配失败/断路是零分而非“无法评估”。
        path_valid = path_requested > 0

    # ---------- 加权 ----------
    if path_valid:
        reward = (
            float(config.iou_weight) * iou
            + float(config.cldice_weight) * cldice
            + float(config.path_weight) * path_score
            - point_penalty
        )
    else:
        # 路径项缺失：剩余质量权重重新归一化到总和 1。
        remaining_weight = float(config.iou_weight) + float(
            config.cldice_weight
        )
        if remaining_weight > 0.0:
            reward = (
                float(config.iou_weight) / remaining_weight * iou
                + float(config.cldice_weight) / remaining_weight * cldice
                - point_penalty
            )
        else:
            reward = -point_penalty
        path_score = 0.0

    return {
        "reward": float(reward),
        "iou": float(iou),
        "f1": float(f1),
        "cldice": float(cldice),
        "path": float(path_score),
        "path_valid": bool(path_valid),
        "path_matched": int(path_matched),
        "path_requested": int(path_requested),
        "point_penalty": float(point_penalty),
    }
