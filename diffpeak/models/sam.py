"""SAM 2.1 分割分支。

约定：
- 图像编码器与提示编码器始终冻结，仅掩码解码器（含其高分辨率投影）可训练；
- 输入图像为 RGB、取值范围 ``[0, 1]``、尺寸 ``B x 3 x 1024 x 1024``；
- 提示点坐标已经位于 SAM2 的 1024 输入坐标系，正点标签为 1，负点为 0。

与 SAM1 的关键差异：
- 图像编码器为 Hiera + FPN neck，输出低分辨率 ``B x 256 x 64 x 64``；
- SAM2.1 解码器默认还需要两级高分辨率特征
  ``B x 32 x 256 x 256`` 与 ``B x 64 x 128 x 128``；
- 归一化使用 ImageNet ``[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]``
  （在 ``[0, 1]`` 像素尺度上），与 SAM1 的 0~255 均值/方差数值等价；
- 官方图像预测器会把冻结的 ``no_mem_embed`` 加到最低分辨率特征上，
  这里保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# SAM2.1 Hiera-B+ 的官方 Hydra 配置名（相对于已安装的 sam2 包）。
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DEFAULT_SAM2_ARCHITECTURE = "sam2.1_hiera_base_plus"


@dataclass(frozen=True)
class SAM2ImageFeatures:
    """一张（或一批）图像的低分辨率与高分辨率 SAM2 特征。

    Attributes:
        low_res: ``B x 256 x 64 x 64``，供条件编码器与掩码解码器使用。
        high_res: ``(B x 256 x 256 x 256, B x 256 x 128 x 128)`` 的原始 FPN
            特征。其通道仍为 neck 的 ``d_model``；1x1 投影 ``conv_s0`` /
            ``conv_s1`` 在解码时执行，使这两个解码器参数可训练。
        """

    low_res: Tensor
    high_res: Optional[Tuple[Tensor, Tensor]] = None

    def select(self, index: int) -> "SAM2ImageFeatures":
        """返回 batch 中第 ``index`` 个样本的视图（保持 batch 维）。"""
        high = (
            None
            if self.high_res is None
            else (self.high_res[0][index : index + 1], self.high_res[1][index : index + 1])
        )
        return SAM2ImageFeatures(self.low_res[index : index + 1], high)


def low_res_features(embeddings: object) -> Tensor:
    """从特征容器或裸张量中取出低分辨率特征。"""
    low = getattr(embeddings, "low_res", None)
    if isinstance(low, Tensor):
        return low
    if isinstance(embeddings, Tensor):
        return embeddings
    raise TypeError(
        "SAM 特征应为 Tensor 或 SAM2ImageFeatures，"
        f"实际为 {type(embeddings)!r}"
    )


def _is_size_pair(value: object) -> bool:
    """判断 ``value`` 是否是可解释为 ``(height, width)`` 的标量对。"""
    if isinstance(value, (str, bytes)):
        return False
    try:
        return len(value) == 2 and all(  # type: ignore[arg-type]
            isinstance(item, Real) for item in value
        )
    except TypeError:
        return False


def _build_sam2(
    config_name: str,
    checkpoint: Optional[str],
    *,
    apply_postprocessing: bool = False,
) -> nn.Module:
    """按需导入官方 SAM2 并构建模型，避免无 SAM2 环境下的强依赖。

    ``checkpoint=None`` 时使用官方随机初始化路径，便于无权重架构测试。
    """
    try:
        from sam2.build_sam import build_sam2
    except ImportError as exc:  # pragma: no cover - 仅用于给出更清晰的安装提示
        raise ImportError(
            "未找到官方 SAM2，请安装 "
            "`SAM2_BUILD_CUDA=0 pip install "
            "'SAM-2 @ git+https://github.com/facebookresearch/sam2.git"
            "@2b90b9f5ceec907a1c18123530e92e794ad901a4'`"
        ) from exc

    # 统一在 CPU 上构建，由外层 ``.to(device)`` 搬运；避免构建时假设 CUDA。
    # apply_postprocessing 默认关闭，让单掩码输出固定使用 token 0，
    # 训练/验证/奖励口径一致。
    try:
        return build_sam2(
            config_file=config_name,
            ckpt_path=checkpoint,
            device="cpu",
            mode="eval",
            apply_postprocessing=apply_postprocessing,
        )
    except (RuntimeError, KeyError) as exc:
        if checkpoint is None:
            raise
        # 官方 ``_load_checkpoint`` 对 missing/unexpected key 只抛出无信息的
        # RuntimeError；若文件根本不含 ``"model"`` 字段（例如直接保存的
        # SAM1 state dict），则抛出 KeyError。两者都补上明确诊断，拒绝把
        # SAM1 ViT-B 权重误用于 SAM2。
        raise ValueError(
            f"无法按 SAM2 配置 {config_name!r} 加载权重：{checkpoint}。"
            "请确认权重与所选 Hiera 尺寸一致。SAM1 ViT-B（如 "
            "sam_vit_b_01ec64.pth）的 state dict 与 SAM2 不兼容；其他尺寸的"
            "SAM2.1 权重也需要对应的 sam_config/sam_architecture。"
        ) from exc


class RoadSAM(nn.Module):
    """基于官方 SAM2.1 Hiera-B+ 的道路分割分支。

    图像编码器、提示编码器、记忆模块与位置参数全部冻结；仅
    ``sam_mask_decoder``（含 ``conv_s0`` / ``conv_s1``）可训练。
    ``encode`` 在 ``torch.no_grad()`` 下运行冻结编码器，返回不带梯度的
    特征；高分辨率投影在 ``decode`` 中执行，以保留了解码器参数的梯度。

    Args:
        checkpoint: 官方 SAM2.1 权重路径（例如
            ``sam2.1_hiera_base_plus.pt``）。
        config_name: 官方 Hydra 配置名，默认 Hiera-B+。
        no_positive_logit: 无正点时的常量负 logit，默认 -20。
        apply_postprocessing: 是否启用官方解码器的动态多掩码回退。默认
            ``False``，保证训练、验证与奖励使用同一种单掩码语义。
        architecture: 写入快照的骨干标识；默认使用 ``config_name``。
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        config_name: str = DEFAULT_SAM2_CONFIG,
        no_positive_logit: float = -20.0,
        apply_postprocessing: bool = False,
        architecture: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.sam = _build_sam2(
            str(config_name),
            None if checkpoint is None else str(checkpoint),
            apply_postprocessing=bool(apply_postprocessing),
        )
        self.config_name = str(config_name)
        self.architecture = str(architecture or config_name)
        self.apply_postprocessing = bool(apply_postprocessing)
        self.no_positive_logit = float(no_positive_logit)

        # SAM2 官方变换在 [0,1] 像素尺度上做 ImageNet 标准化。
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        self._freeze_encoder_and_prompt()

    @property
    def mask_decoder(self) -> nn.Module:
        """可训练的解码器（SAM2 命名为 ``sam_mask_decoder``）。"""
        return self.sam.sam_mask_decoder

    def _freeze_encoder_and_prompt(self) -> None:
        """冻结除掩码解码器外的全部 SAM2 参数。"""
        for parameter in self.sam.parameters():
            parameter.requires_grad_(False)
        for parameter in self.sam.sam_mask_decoder.parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> "RoadSAM":
        """保持冻结模块处于 eval，仅切换掩码解码器的训练状态。"""
        super().train(mode)
        for name, child in self.sam.named_children():
            if name != "sam_mask_decoder":
                child.eval()
        self.sam.sam_mask_decoder.train(mode)
        return self

    @staticmethod
    def _normalise_valid_sizes(
        valid_sizes: Union[Sequence[Tuple[int, int]], Tensor],
        batch_size: int,
        image_size: int,
    ) -> List[Tuple[int, int]]:
        """把显式 ``valid_sizes`` 校验为长度为 B 的 ``(height, width)`` 列表。

        支持 ``B x 2`` 张量、``B`` 个 ``(h, w)`` 的序列，以及 batch=1 时
        单个 ``(h, w)``。这里只接受调用方显式给出的几何信息，不会从像素
        内容推断填充区域。
        """
        raw = (
            valid_sizes.detach().cpu().tolist()
            if isinstance(valid_sizes, Tensor)
            else valid_sizes
        )
        if _is_size_pair(raw):
            if batch_size != 1:
                raise ValueError("单个 (height, width) 只适用于 batch_size=1")
            items = [raw]
        else:
            try:
                items = list(raw)
            except TypeError as exc:
                raise TypeError("valid_sizes 必须是序列或单个 (height, width)") from exc
            if len(items) != batch_size:
                raise ValueError(
                    f"valid_sizes 长度 {len(items)} 与 batch_size {batch_size} 不一致"
                )

        normalised: List[Tuple[int, int]] = []
        for item in items:
            if item is None:
                normalised.append((image_size, image_size))
                continue
            if isinstance(item, Tensor):
                item = item.detach().cpu().tolist()
            if not _is_size_pair(item):
                raise ValueError("valid_sizes 的每个元素必须是 (height, width)")
            height, width = int(item[0]), int(item[1])
            if not (0 < height <= image_size and 0 < width <= image_size):
                raise ValueError(
                    f"valid_size 必须在 (0, {image_size}] 内，实际为 {(height, width)}"
                )
            normalised.append((height, width))
        return normalised

    def encode(
        self,
        images: Tensor,
        valid_sizes: Optional[Union[Sequence[Tuple[int, int]], Tensor]] = None,
    ) -> SAM2ImageFeatures:
        """编码 RGB 图像，返回 SAM2 低分辨率与高分辨率特征。

        Args:
            images: ``[0, 1]`` 范围的 ``B x 3 x 1024 x 1024`` 张量。非方形
                影像应由外层先按最长边缩放到 1024，并在右下补零到 1024²。
            valid_sizes: 可选的每样本有效区域 ``(height, width)``。提供时
                会在归一化后把右下填充区显式置零。

        Returns:
            :class:`SAM2ImageFeatures`，其中 ``low_res`` 为
            ``B x 256 x 64 x 64``，``high_res`` 为原始 FPN 的
            ``B x 256 x 256 x 256`` 与 ``B x 256 x 128 x 128`` 特征。
            所有特征都不带梯度。
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images 应为 B x 3 x H x W，实际为 {tuple(images.shape)}")
        if tuple(images.shape[-2:]) != (1024, 1024):
            raise ValueError(
                f"SAM2 Hiera 要求 1024 x 1024 输入，实际为 {tuple(images.shape[-2:])}"
            )
        if not images.is_floating_point():
            raise TypeError(f"images 必须是浮点张量，实际 dtype 为 {images.dtype}")

        height, width = int(images.shape[-2]), int(images.shape[-1])

        with torch.no_grad():
            x = (images - self.pixel_mean.to(device=images.device, dtype=images.dtype)) / (
                self.pixel_std.to(device=images.device, dtype=images.dtype)
            )
            if valid_sizes is not None:
                sizes = self._normalise_valid_sizes(
                    valid_sizes, images.shape[0], image_size=height
                )
                if any(h < height or w < width for h, w in sizes):
                    mask = torch.zeros(
                        (images.shape[0], 1, height, width),
                        dtype=torch.bool,
                        device=images.device,
                    )
                    for index, (valid_h, valid_w) in enumerate(sizes):
                        mask[index, 0, :valid_h, :valid_w] = True
                    x = x.masked_fill(~mask, 0.0)

            # 直接调用 image_encoder，而不是 forward_image：后者会在 no_grad
            # 内执行解码器的 conv_s0/conv_s1，导致这两个参数拿不到梯度。
            backbone_out = self.sam.image_encoder(x)
            low_res = backbone_out["vision_features"]
            if bool(getattr(self.sam, "directly_add_no_mem_embed", False)):
                # ``no_mem_embed`` 的官方形状是 ``1 x 1 x C``（对应
                # ``HW x B x C``），加到 BCHW 特征前必须 reshape 为
                # ``1 x C x 1 x 1``，否则会沿错误维度广播或直接报错。
                no_mem = self.sam.no_mem_embed.view(
                    1, -1, 1, 1
                ).to(device=low_res.device, dtype=low_res.dtype)
                low_res = low_res + no_mem

            high_res: Optional[Tuple[Tensor, Tensor]] = None
            if bool(getattr(self.sam, "use_high_res_features_in_sam", False)):
                fpn = backbone_out["backbone_fpn"]
                high_res = (fpn[0].detach(), fpn[1].detach())

            return SAM2ImageFeatures(low_res.detach(), high_res)

    @staticmethod
    def _per_sample_points(
        points: Sequence[Tuple[Tensor, Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> List[Tuple[Tensor, Tensor]]:
        """校验并转换每个样本的点集，**不**做跨样本补齐。

        官方 SAM2 提示编码器在 ``boxes=None`` 时会自行追加一个
        ``not_a_point`` 填充 token。如果在 batch 内按最长点集补齐为 -1，
        提示编码器仍会再追加一个，导致同一样本在批量与单图解码时 token
        数不同、结果不一致。因此 ``decode`` 必须使用未补齐的逐样本点集。
        """
        if len(points) != batch_size:
            raise ValueError(f"points 数量 {len(points)} 与 batch size {batch_size} 不一致")

        prepared: List[Tuple[Tensor, Tensor]] = []
        for sample_points in points:
            if not isinstance(sample_points, (tuple, list)) or len(sample_points) != 2:
                raise ValueError("每个样本的 points 应为 (coords[K, 2], labels[K])")
            coords, labels = sample_points
            coords = coords.to(device=device, dtype=torch.float32).reshape(-1, 2)
            labels = labels.to(device=device, dtype=torch.long).reshape(-1)
            if coords.shape[0] != labels.shape[0]:
                raise ValueError("点坐标与标签数量不一致")
            prepared.append((coords, labels))
        return prepared

    def _prepare_points(
        self,
        points: Sequence[Tuple[Tensor, Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """将变长点集补齐为 ``B x K x 2`` 与 ``B x K``。

        这是批量张量接口的辅助函数；``decode`` 逐图解码，使用
        :meth:`_per_sample_points`，避免跨样本补齐引入多余提示 token。
        """
        prepared = self._per_sample_points(points, batch_size, device)
        max_points = max((coords.shape[0] for coords, _ in prepared), default=0)

        padded_coords = torch.zeros(batch_size, max_points, 2, device=device, dtype=torch.float32)
        padded_labels = torch.full(
            (batch_size, max_points), -1, device=device, dtype=torch.long
        )
        for index, (coords, labels) in enumerate(prepared):
            count = coords.shape[0]
            if count == 0:
                continue
            padded_coords[index, :count] = coords
            padded_labels[index, :count] = labels
        return padded_coords, padded_labels

    def _empty_logits(
        self,
        embeddings: Union[Tensor, SAM2ImageFeatures],
        output_size: Tuple[int, int],
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """生成不带梯度的常量负 logits，用于无正点样本。"""
        low_res = low_res_features(embeddings)
        batch_size = low_res.shape[0]
        height, width = int(output_size[0]), int(output_size[1])
        return torch.full(
            (batch_size, 1, height, width),
            self.no_positive_logit,
            device=low_res.device,
            dtype=low_res.dtype if dtype is None else dtype,
        )

    def decode(
        self,
        embeddings: Union[Tensor, SAM2ImageFeatures],
        points: Sequence[Tuple[Tensor, Tensor]],
        output_size: Tuple[int, int] = (1024, 1024),
    ) -> Tensor:
        """根据 SAM2 图像特征与正负点提示输出道路分割 logits。

        Args:
            embeddings: ``encode`` 返回的 :class:`SAM2ImageFeatures`。为兼容
                旧测试，也接受裸低分辨率 ``B x 256 x 64 x 64`` 张量；此时若
                模型需要高分辨率特征会明确报错。
            points: 长度为 B 的列表，每项为
                ``(coords[K, 2], labels[K])``；坐标使用 1024 坐标系，
                标签 1 为正点、0 为负点。
            output_size: 输出空间尺寸，默认 ``(1024, 1024)``。

        Returns:
            ``B x 1 x H x W`` 的分割 logits。
        """
        low_res = low_res_features(embeddings)
        if low_res.ndim != 4:
            raise ValueError(
                f"embeddings 应为 B x 256 x 64 x 64，实际为 {tuple(low_res.shape)}"
            )
        batch_size = low_res.shape[0]
        if len(output_size) != 2:
            raise ValueError("output_size 应为 (H, W)")

        high_res = (
            embeddings.high_res if isinstance(embeddings, SAM2ImageFeatures) else None
        )

        prepared_points = self._per_sample_points(points, batch_size, low_res.device)
        positive_counts = [
            int((labels == 1).sum()) for _, labels in prepared_points
        ]

        # 整个 batch 都没有正点：直接返回常量，避免无意义的解码与空梯度。
        if not any(count > 0 for count in positive_counts):
            return self._empty_logits(embeddings, output_size)

        if bool(getattr(self.sam, "use_high_res_features_in_sam", False)) and high_res is None:
            raise ValueError(
                "SAM2.1 解码器需要高分辨率特征；请传入 RoadSAM.encode 的 "
                "SAM2ImageFeatures，而不是裸低分辨率张量"
            )

        with torch.no_grad():
            image_pe = self.sam.sam_prompt_encoder.get_dense_pe()

        decoded: List[Optional[Tensor]] = []
        for index, positive_count in enumerate(positive_counts):
            if positive_count == 0:
                decoded.append(None)
                continue

            sample_features = (
                embeddings.select(index)
                if isinstance(embeddings, SAM2ImageFeatures)
                else embeddings[index : index + 1]
            )
            sample_low = low_res_features(sample_features)
            # 不补齐：提示编码器会自行追加一个 not_a_point token，与官方
            # 单图预测路径保持一致。
            coords, labels = prepared_points[index]
            sample_coords = coords.unsqueeze(0)
            sample_labels = labels.unsqueeze(0)
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.sam.sam_prompt_encoder(
                    points=(sample_coords, sample_labels), boxes=None, masks=None
                )

            high_res_features = None
            if high_res is not None:
                # 1x1 投影属于掩码解码器，必须在 no_grad 之外执行，保证
                # conv_s0 / conv_s1 与解码器其他参数一样可训练。
                sample_high = high_res[0][index : index + 1], high_res[1][index : index + 1]
                high_res_features = [
                    self.sam.sam_mask_decoder.conv_s0(sample_high[0]),
                    self.sam.sam_mask_decoder.conv_s1(sample_high[1]),
                ]

            low_res_logits, _, _, _ = self.sam.sam_mask_decoder(
                image_embeddings=sample_low,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )
            decoded.append(
                F.interpolate(
                    low_res_logits,
                    size=(int(output_size[0]), int(output_size[1])),
                    mode="bilinear",
                    align_corners=False,
                )
            )

        reference = next(logits for logits in decoded if logits is not None)
        outputs = [
            (
                self._empty_logits(
                    embeddings.select(index)
                    if isinstance(embeddings, SAM2ImageFeatures)
                    else embeddings[index : index + 1],
                    output_size,
                    dtype=reference.dtype,
                )
                if logits is None
                else logits
            )
            for index, logits in enumerate(decoded)
        ]
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _normalise_prompt_valid_sizes(
        valid_sizes: Union[Sequence[Tuple[int, int]], Tensor],
        batch_size: int,
        output_size: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """把 ``valid_sizes`` 校验为 ``output_size`` 坐标系下的 ``(h, w)``。

        与 :meth:`encode` 的 ``valid_sizes`` 口径一致：数值单位是 SAM 输入
        画布像素，``None`` 表示该样本满幅。密集热图提示会把这些尺寸按比例
        缩放到热图分辨率后再屏蔽右下填充区。
        """
        out_height, out_width = int(output_size[0]), int(output_size[1])
        raw = (
            valid_sizes.detach().cpu().tolist()
            if isinstance(valid_sizes, Tensor)
            else valid_sizes
        )
        if _is_size_pair(raw):
            if batch_size != 1:
                raise ValueError("单个 (height, width) 只适用于 batch_size=1")
            items: Sequence[object] = [raw]
        else:
            try:
                items = list(raw)  # type: ignore[arg-type]
            except TypeError as exc:
                raise TypeError("valid_sizes 必须是序列或单个 (height, width)") from exc
            if len(items) != batch_size:
                raise ValueError(
                    f"valid_sizes 长度 {len(items)} 与 batch_size {batch_size} 不一致"
                )

        normalised: List[Tuple[int, int]] = []
        for item in items:
            if item is None:
                normalised.append((out_height, out_width))
                continue
            if isinstance(item, Tensor):
                item = item.detach().cpu().tolist()
            if not _is_size_pair(item):
                raise ValueError("valid_sizes 的每个元素必须是 (height, width)")
            height, width = int(item[0]), int(item[1])  # type: ignore[index]
            if not (0 < height <= out_height and 0 < width <= out_width):
                raise ValueError(
                    "valid_size 必须在 "
                    f"(0, {out_height}] x (0, {out_width}] 内，"
                    f"实际为 {(height, width)}"
                )
            normalised.append((height, width))
        return normalised

    def decode_heatmaps(
        self,
        embeddings: Union[Tensor, SAM2ImageFeatures],
        heatmaps: Tensor,
        output_size: Tuple[int, int] = (1024, 1024),
        valid_sizes: Optional[Union[Sequence[Tuple[int, int]], Tensor]] = None,
        decoder: Optional[nn.Module] = None,
    ) -> Tensor:
        """用可微密集热图提示解码全部样本。

        该路径用于联合 ABC 端到端训练：把扩散热图的**正负通道差**
        ``heatmaps[:, 0] - heatmaps[:, 1]``（一通道输入则直接使用）作为 SAM
        掩码提示，经冻结的提示编码器得到 dense embedding，再交给掩码解码器。
        与 :meth:`decode` 不同：

        * 不使用任何稀疏点提示（``points=None`` / ``boxes=None``）；
        * 不跳过空热图样本，整批一次解码，因此每张图都有解码器梯度路径；
        * 提示编码器在**不包裹** ``torch.no_grad()`` 的情况下执行。其权重仍然
          冻结，但因输入 ``heatmaps`` 要求梯度，反向传播可以穿过冻结的掩码
          下采样卷积回到热图，进而回到去噪器/条件编码器；
        * 高分辨率 FPN 投影 ``conv_s0`` / ``conv_s1`` 取自 ``decoder``（默认
          当前模型的掩码解码器），保证联合训练时这些参数仍可训练。

        Args:
            embeddings: ``encode`` 返回的 :class:`SAM2ImageFeatures`（含高分辨率
                特征时最完整）；也接受裸低分辨率 ``B x 256 x 64 x 64`` 张量。
            heatmaps: ``B x 2 x h x w`` 的正/负提示热图（通道 0 为正、1 为负），
                或 ``B x 1 x h x w`` 的已签名掩码。范围约定为 ``[0, 1]``。
            output_size: 输出 logits 的 ``(height, width)``，默认 1024。
            valid_sizes: 可选的每样本有效区域 ``(height, width)``，单位为
                ``output_size`` 画布像素；右下填充区会在送入提示编码器前清零。
            decoder: 可选的掩码解码器覆盖模块（需提供与 SAM2 相同的
                ``forward`` 签名以及 ``conv_s0`` / ``conv_s1``，例如强化学习
                阶段冻结的参考解码器）。默认使用 ``self.mask_decoder``。

        Returns:
            ``B x 1 x H x W`` 的分割 logits（``H, W`` 来自 ``output_size``）。
            当 ``heatmaps.requires_grad`` 时输出保留完整计算图。
        """
        low_res = low_res_features(embeddings)
        if low_res.ndim != 4:
            raise ValueError(
                f"embeddings 应为 B x 256 x 64 x 64，实际为 {tuple(low_res.shape)}"
            )
        if len(output_size) != 2:
            raise ValueError("output_size 应为 (H, W)")
        if not torch.is_tensor(heatmaps):
            raise TypeError(f"heatmaps 必须是张量，实际为 {type(heatmaps)!r}")
        if heatmaps.ndim != 4 or heatmaps.shape[1] not in (1, 2):
            raise ValueError(
                "heatmaps 应为 B x 2 x H x W（或 B x 1 的已签名掩码），"
                f"实际为 {tuple(heatmaps.shape)}"
            )
        if not heatmaps.is_floating_point():
            raise TypeError(f"heatmaps 必须是浮点张量，实际 dtype 为 {heatmaps.dtype}")
        batch_size = int(low_res.shape[0])
        if int(heatmaps.shape[0]) != batch_size:
            raise ValueError(
                f"heatmaps batch {int(heatmaps.shape[0])} 与 embeddings "
                f"batch {batch_size} 不一致"
            )
        if heatmaps.device != low_res.device:
            # 移动是可微操作，不切断到原始 heatmaps 的梯度。
            heatmaps = heatmaps.to(device=low_res.device)

        # 官方约定通道 0 为正提示、通道 1 为负提示；有符号差值即掩码 logits。
        if heatmaps.shape[1] == 2:
            dense_mask = heatmaps[:, 0:1] - heatmaps[:, 1:2]
        else:
            dense_mask = heatmaps[:, 0:1]

        out_height, out_width = int(output_size[0]), int(output_size[1])
        heat_height, heat_width = int(dense_mask.shape[-2]), int(dense_mask.shape[-1])
        sizes: Optional[List[Tuple[int, int]]] = None
        if valid_sizes is not None:
            sizes = self._normalise_prompt_valid_sizes(
                valid_sizes, batch_size, (out_height, out_width)
            )
            valid = torch.zeros_like(dense_mask)
            for index, (valid_height, valid_width) in enumerate(sizes):
                scaled_height = max(
                    1, min(heat_height, int(round(valid_height * heat_height / out_height)))
                )
                scaled_width = max(
                    1, min(heat_width, int(round(valid_width * heat_width / out_width)))
                )
                valid[index, :, :scaled_height, :scaled_width] = 1.0
            # 乘法保持可微，并让填充区的提示恰为 0（背景）。
            dense_mask = dense_mask * valid

        prompt_encoder = getattr(self.sam, "sam_prompt_encoder", None)
        if prompt_encoder is None:
            raise AttributeError("RoadSAM 缺少 sam_prompt_encoder，无法编码密集热图提示")
        mask_input_size = getattr(prompt_encoder, "mask_input_size", (256, 256))
        mask_input_size = (int(mask_input_size[0]), int(mask_input_size[1]))
        mask_prompt = F.interpolate(
            dense_mask,
            size=mask_input_size,
            mode="bilinear",
            align_corners=False,
        )
        if sizes is not None:
            # 双线性上采样会把有效区边界外的零向内侧扩散；在提示编码器的
            # 输入分辨率上再次硬屏蔽，保证 padding 提示严格为 0。
            prompt_valid = torch.zeros_like(mask_prompt)
            for index, (valid_height, valid_width) in enumerate(sizes):
                scaled_height = max(
                    1,
                    min(
                        mask_input_size[0],
                        int(
                            round(
                                valid_height * mask_input_size[0] / out_height
                            )
                        ),
                    ),
                )
                scaled_width = max(
                    1,
                    min(
                        mask_input_size[1],
                        int(round(valid_width * mask_input_size[1] / out_width)),
                    ),
                )
                prompt_valid[index, :, :scaled_height, :scaled_width] = 1.0
            mask_prompt = mask_prompt * prompt_valid
        # 与提示编码器参数 dtype 对齐；``.to`` 可微，不切断到热图的梯度。
        try:
            prompt_dtype = next(prompt_encoder.parameters()).dtype
        except StopIteration:  # pragma: no cover - 无参数的测试替身
            prompt_dtype = mask_prompt.dtype
        if mask_prompt.dtype != prompt_dtype:
            mask_prompt = mask_prompt.to(dtype=prompt_dtype)
        # 关键：此处不能使用 torch.no_grad()。提示编码器参数已冻结，但保留
        # 计算图后，梯度可以穿过冻结权重回传到 mask_prompt / heatmaps。
        sparse_embeddings, dense_embeddings = prompt_encoder(
            points=None, boxes=None, masks=mask_prompt
        )

        decoder_module = self.mask_decoder if decoder is None else decoder
        if not callable(decoder_module):
            raise TypeError("decoder 必须是可调用的掩码解码器模块")

        high_res = (
            embeddings.high_res if isinstance(embeddings, SAM2ImageFeatures) else None
        )
        if bool(getattr(self.sam, "use_high_res_features_in_sam", False)) and high_res is None:
            raise ValueError(
                "SAM2.1 解码器需要高分辨率特征；请传入 RoadSAM.encode 的 "
                "SAM2ImageFeatures，而不是裸低分辨率张量"
            )

        high_res_features = None
        if high_res is not None:
            conv_s0 = getattr(decoder_module, "conv_s0", None)
            conv_s1 = getattr(decoder_module, "conv_s1", None)
            if conv_s0 is None or conv_s1 is None:
                raise ValueError(
                    "decoder 缺少 conv_s0 / conv_s1，无法复用高分辨率 FPN 投影"
                )
            high_res_features = [
                conv_s0(high_res[0]),
                conv_s1(high_res[1]),
            ]

        image_pe = prompt_encoder.get_dense_pe()
        low_res_logits, _, _, _ = decoder_module(
            image_embeddings=low_res,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        return F.interpolate(
            low_res_logits,
            size=(out_height, out_width),
            mode="bilinear",
            align_corners=False,
        )
