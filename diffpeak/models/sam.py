"""SAM 分割分支。

约定：
- 图像编码器与提示编码器始终冻结，仅掩码解码器可训练；
- 输入图像为 RGB、取值范围 ``[0, 1]``、尺寸 ``B x 3 x 1024 x 1024``；
- 提示点坐标已经位于 SAM 的 1024 输入坐标系，正点标签为 1，负点为 0。
"""

from __future__ import annotations

from numbers import Real
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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


def _build_sam(checkpoint: str, model_type: str) -> nn.Module:
    """按需导入官方实现并构建 SAM，避免无 SAM 环境下的模块级强依赖。"""
    try:
        from segment_anything import sam_model_registry
    except ImportError as exc:  # pragma: no cover - 仅用于给出更清晰的安装提示
        raise ImportError(
            "未找到官方 segment-anything，请安装 "
            "`pip install git+https://github.com/facebookresearch/segment-anything.git`"
        ) from exc
    if model_type not in sam_model_registry:
        raise ValueError(f"未知的 SAM 模型类型: {model_type}")
    return sam_model_registry[model_type](checkpoint=checkpoint)


class RoadSAM(nn.Module):
    """基于官方 SAM ViT-B 的道路分割分支。

    图像编码器和提示编码器冻结，掩码解码器可训练。``encode`` 与 SAM 的
    预处理一起完成，内部使用 ``torch.no_grad()``，因此返回的特征不带梯度。

    Args:
        checkpoint: 官方 SAM 权重路径（例如 ``sam_vit_b_01ec64.pth``）。
        model_type: ``segment_anything`` 注册表中的模型类型，默认 ``vit_b``。
        no_positive_logit: 无正点时的常量负 logit，默认 -20（sigmoid 后约为 0）。
    """

    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_b",
        no_positive_logit: float = -20.0,
    ) -> None:
        super().__init__()
        self.sam = _build_sam(checkpoint, model_type)
        self.model_type = model_type
        self.no_positive_logit = float(no_positive_logit)

        # 官方 SAM 使用 0~255 像素尺度下的均值/方差做归一化。
        self.register_buffer(
            "pixel_mean",
            torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1),
            persistent=False,
        )

        self._freeze_encoder_and_prompt()

    def _freeze_encoder_and_prompt(self) -> None:
        """冻结图像编码器与提示编码器，仅保留掩码解码器可训练。"""
        for module in (self.sam.image_encoder, self.sam.prompt_encoder):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad_(True)

    def train(self, mode: bool = True) -> "RoadSAM":
        """保持冻结模块处于 eval，避免训练脚本整体 ``train()`` 改变其行为。"""
        super().train(mode)
        self.sam.image_encoder.eval()
        self.sam.prompt_encoder.eval()
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
    ) -> Tensor:
        """编码 RGB 图像，返回 SAM 图像特征。

        Args:
            images: ``[0, 1]`` 范围的 ``B x 3 x 1024 x 1024`` 张量。非方形
                影像应由外层先按最长边缩放到 1024，并在右下补零到 1024²。
            valid_sizes: 可选的每样本有效区域 ``(height, width)``，表示缩放
                后内容在 1024 画布左上角占据的区域。提供时会把右下填充区在
                归一化后显式置零，与官方 SAM ``preprocess`` 的
                ``F.pad(..., value=0)`` 一致。必须由调用方显式给出；本方法
                不会根据黑色像素猜测填充区域。

        Returns:
            ``B x 256 x 64 x 64`` 图像特征，不带梯度。
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images 应为 B x 3 x H x W，实际为 {tuple(images.shape)}")
        if tuple(images.shape[-2:]) != (1024, 1024):
            raise ValueError(f"SAM ViT-B 要求 1024 x 1024 输入，实际为 {tuple(images.shape[-2:])}")
        if not images.is_floating_point():
            raise TypeError(f"images 必须是浮点张量，实际 dtype 为 {images.dtype}")

        height, width = int(images.shape[-2]), int(images.shape[-1])

        with torch.no_grad():
            x = images * 255.0

            # 官方 Sam.preprocess 先标准化有效图像，再以 0 填充。
            # 外层已填充的 RGB 画布须在标准化之后重置无效区域。
            x = (x - self.pixel_mean.to(device=x.device, dtype=x.dtype)) / self.pixel_std.to(
                device=x.device, dtype=x.dtype
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

            return self.sam.image_encoder(x)

    def _prepare_points(
        self,
        points: Sequence[Tuple[Tensor, Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """将变长点集补齐为 ``B x K x 2`` 与 ``B x K``。

        标签为 -1 的填充位置会被官方提示编码器视为 ``not_a_point``。
        """
        if len(points) != batch_size:
            raise ValueError(f"points 数量 {len(points)} 与 batch size {batch_size} 不一致")

        coords_list: List[Tensor] = []
        labels_list: List[Tensor] = []
        max_points = 0
        for sample_points in points:
            if not isinstance(sample_points, (tuple, list)) or len(sample_points) != 2:
                raise ValueError("每个样本的 points 应为 (coords[K, 2], labels[K])")
            coords, labels = sample_points
            coords = coords.to(device=device, dtype=torch.float32).reshape(-1, 2)
            labels = labels.to(device=device, dtype=torch.long).reshape(-1)
            if coords.shape[0] != labels.shape[0]:
                raise ValueError("点坐标与标签数量不一致")
            coords_list.append(coords)
            labels_list.append(labels)
            max_points = max(max_points, coords.shape[0])

        padded_coords = torch.zeros(batch_size, max_points, 2, device=device, dtype=torch.float32)
        padded_labels = torch.full(
            (batch_size, max_points), -1, device=device, dtype=torch.long
        )
        for index, (coords, labels) in enumerate(zip(coords_list, labels_list)):
            count = coords.shape[0]
            if count == 0:
                continue
            padded_coords[index, :count] = coords
            padded_labels[index, :count] = labels
        return padded_coords, padded_labels

    def _empty_logits(
        self,
        embeddings: Tensor,
        output_size: Tuple[int, int],
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """生成不带梯度的常量负 logits，用于无正点样本。

        ``dtype`` 可显式指定，以便混合 batch 中与掩码解码器的输出对齐
        （例如 AMP 下解码器输出与图像特征 dtype 不同的情况）。
        """
        batch_size = embeddings.shape[0]
        height, width = int(output_size[0]), int(output_size[1])
        return torch.full(
            (batch_size, 1, height, width),
            self.no_positive_logit,
            device=embeddings.device,
            dtype=embeddings.dtype if dtype is None else dtype,
        )

    def decode(
        self,
        embeddings: Tensor,
        points: Sequence[Tuple[Tensor, Tensor]],
        output_size: Tuple[int, int] = (1024, 1024),
    ) -> Tensor:
        """根据 SAM 图像特征与正负点提示输出道路分割 logits。

        Args:
            embeddings: ``encode`` 返回的 ``B x 256 x 64 x 64`` 特征。
            points: 长度为 B 的列表，每项为
                ``(coords[K, 2], labels[K])``；坐标使用 SAM 的 1024 坐标系，
                标签 1 为正点、0 为负点。
            output_size: 输出空间尺寸，默认 ``(1024, 1024)``。

        Returns:
            ``B x 1 x H x W`` 的分割 logits。若某个样本没有正点，则该样本输出
            常量负 logits，并且不会为其构造计算图。

        Notes:
            官方 ``mask_decoder`` 内部会对 ``image_embeddings`` 执行
            ``repeat_interleave(tokens.shape[0], dim=0)``。如果一次传入 B 张
            图像和 B 组提示，会得到 B*B 个 mask；因此这里逐图解码，确保
            batch>1 时语义正确。掩码解码器参数保持可训练，梯度仍可回传。
        """
        if embeddings.ndim != 4:
            raise ValueError(f"embeddings 应为 B x 256 x 64 x 64，实际为 {tuple(embeddings.shape)}")
        batch_size = embeddings.shape[0]
        if len(output_size) != 2:
            raise ValueError("output_size 应为 (H, W)")

        coords, labels = self._prepare_points(points, batch_size, embeddings.device)
        positive_counts = (labels == 1).sum(dim=1).tolist()

        # 整个 batch 都没有正点：直接返回常量，避免无意义的解码与空梯度。
        if not any(count > 0 for count in positive_counts):
            return self._empty_logits(embeddings, output_size)

        with torch.no_grad():
            image_pe = self.sam.prompt_encoder.get_dense_pe()

        decoded: List[Optional[Tensor]] = []
        for index, positive_count in enumerate(positive_counts):
            if positive_count == 0:
                # 无正点样本统一返回空道路掩码，不强行补全局最大点。
                # 先占位，待确定解码器输出的 dtype 后再生成常量。
                decoded.append(None)
                continue

            # 每张图像单独调用一次解码器，避免官方 repeat_interleave 造成
            # B 张图像 × B 组提示的 B² 组合。
            sample_coords = coords[index : index + 1]
            sample_labels = labels[index : index + 1]
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=(sample_coords, sample_labels), boxes=None, masks=None
                )
            low_res_logits, _ = self.sam.mask_decoder(
                image_embeddings=embeddings[index : index + 1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            decoded.append(
                F.interpolate(
                    low_res_logits,
                    size=(int(output_size[0]), int(output_size[1])),
                    mode="bilinear",
                    align_corners=False,
                )
            )

        # 用真实解码结果的 dtype/device 生成空样本常量，兼容 AMP 下
        # 图像特征与 mask decoder 输出精度不同的情况。
        reference = next(logits for logits in decoded if logits is not None)
        outputs = [
            (
                self._empty_logits(
                    embeddings[index : index + 1],
                    output_size,
                    dtype=reference.dtype,
                )
                if logits is None
                else logits
            )
            for index, logits in enumerate(decoded)
        ]
        return torch.cat(outputs, dim=0)
