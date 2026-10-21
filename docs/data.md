# 数据与提示目标模块说明

本文说明 `diffpeak/data/` 与 `diffpeak/prompts.py` 的实现约定，以及
DeepGlobe_Road 的实际组织方式。模型结构、扩散过程和奖励设计见
[`docs/model_grpo.md`](model_grpo.md)。本文只讨论数据、提示目标与固定峰值提取。

## 1. DeepGlobe_Road 实际内容

当前数据目录为：

```text
DeepGlobe_Road/
├── metadata.csv          # 8570 行，含 train / valid / test
├── class_dict.csv        # road = (255,255,255)，background = (0,0,0)
├── train/                # 6226 张 *_sat.jpg + 6226 张 *_mask.png
├── valid/                # 1243 张 *_sat.jpg，无 mask
└── test/                 # 1101 张 *_sat.jpg，无 mask
```

影像与掩码均为 `1024 x 1024`；掩码是 8-bit RGB PNG，道路为白色。
`metadata.csv` 同时列出了没有标注的 valid/test，因此**不能**按 metadata
的 `split` 字段直接把 valid 当作监督集。

`scripts/prepare_splits.py` 只从 `train/` 中有 mask 的影像生成划分清单：

```text
splits/train.json
splits/val.json
splits/test.json         # 仅当 --test-fraction > 0
splits/unlabeled.json    # valid + test 的影像，仅用于推理
splits/protocol.json
```

传入 `--val-fraction 0.1 --test-fraction 0.1 --seed 42` 可生成三分清单；上游环境
已据此生成 `splits/protocol_v1/`，记录为 train/val/test = 4,980/623/623，本
仓库不附带生成文件。`train.json` / `val.json` / `test.json` 可以传给
`DeepGlobeDataset`；`unlabeled.json` 中的影像没有 GT，**不能**传给监督数据集，
只适合单独的无标注推理流程。测试集不参与提示阈值选择、奖励优化或模型选择。
旧的两分清单和诊断样本与新 test 的交叉使用历史见
[实验协议](experimental_protocol.md) 第 2 节，不能仅凭清单文件名声称测试集从未
被查看。

> 注意：按影像随机划分不保证地理独立性。若后续能获得区域/城市分组，
> 应替换为区域隔离的训练/验证划分，再比较不同实验。

## 2. `split_file` 格式

`split_file` 是 JSON 字符串数组，项为相对数据根目录的**影像**路径：

```json
[
  "train/100034_sat.jpg",
  "train/100081_sat.jpg"
]
```

掩码由同目录文件名派生：

```text
train/100034_sat.jpg  ->  train/100034_mask.png
```

规则实现于 `diffpeak/data/splits.py`。绝对路径、`..` 逃逸和缺少掩码都会
在数据集初始化时直接报错，避免把无标注影像静默混入训练。

## 3. `DeepGlobeDataset`

```python
DeepGlobeDataset(
    root: str,
    split_file: str,
    augment: bool = False,
    image_size: int = 1024,
    heatmap_size: int = 256,
    seed: int = 42,
    *,
    sampling_config=None,
    heatmap_sigma=1.0,
    target_min_distance=4.0,
    hard_negative_provider=None,
)
```

`split_file` 也可以直接传字符串列表，便于测试和临时实验。

### 3.1 返回字段

| 键 | 形状 / 类型 | 说明 |
|---|---|---|
| `image` | `float32 [3, image_size, image_size]` | RGB，范围 `[0,1]` |
| `mask` | `float32 [1, image_size, image_size]` | 二值道路（0/1） |
| `skeleton` | `float32 [1, image_size, image_size]` | 二值中心线 |
| `heatmaps` | `float32 [2, heatmap_size, heatmap_size]` | 通道 0 正、通道 1 负 |
| `id` | `str` | 如 `"100034"`（去掉 `_sat`） |
| `original_size` | `(H, W)` | 当前标注分辨率（几何增广后） |
| `resized_size` | `(H, W)` | 填充前的缩放尺寸，可直接作为 `valid_sizes` |

骨架输出与 `mask` 一样位于 `image_size` 坐标系；**骨架的构造和提示点采样
都发生在原始标注分辨率**，之后才做缩放/映射，避免细道路在 256 热图上
先断开。

### 3.2 几何变换

* 缩放比例：`min(image_size / H, image_size / W)`；
* 保持纵横比，内容放在左上角，右侧与底部补零（bottom-right padding）；
* 影像用 bilinear、掩码/骨架用 nearest；
* 点映射使用像素中心对齐，热图与 SAM2 坐标互为逆变换：
  `x_image = (x_heat + 0.5) * image_size / heatmap_size - 0.5`。

例如一张 `100 x 200` 的影像在 `image_size=1024` 下缩放为
`(512, 1024)`，下方 `512` 行为 padding；`valid_sizes=(512, 1024)`。
`PeakExtractor` 会先检测峰、再按该尺寸过滤 padding，padding 不参与 NMS，
因此不会压掉有效边界峰。

### 3.3 正点、负点与无道路样本

* 正点：在原始分辨率骨架上拆分支，沿分支按 `[min_spacing, max_spacing]`
  的随机间距采样；路口/端点按配置概率额外加入。结构点在热图空间筛选中
  优先保留。
* 负点：道路边界外侧的带状区域 + 普通背景；`boundary_margin` 避开标注
  不确定的紧贴边界，`boundary_band` 控制外侧带宽。
* 无道路样本：正、负点都为空，**双通道目标热图全部为零**，不会用背景负点
  填满负通道。
* 点转热图：每个点渲染高斯后逐点取 **max**，不是 sum；默认
  `heatmap_sigma=1.0` 个热图像素。

可选的困难负点接口：

```python
def provider(image, road_mask, rng):
    # image: HxWx3 uint8；road_mask: HxW bool；rng: numpy Generator
    # 返回 (N, 2) 的 (y, x)，必须位于图像内且不在道路上
    return hard_coords
```

`hard_negative_provider` 只接收调用方（例如阶段性分割模型）提供的坐标。
本模块**不自动挖掘**困难负点，也不声称具备这种能力。

### 3.4 随机状态与 persistent workers

每个样本使用独立的 `numpy.random.Generator`，种子由
`(seed, worker_id, epoch, index, occurrence)` 组成：

* `worker_id` 来自 `torch.utils.data.get_worker_info()`，不同 worker 的
  随机流互不共享；
* `augment=True` 时 `occurrence` 是同一 worker 内同索引被访问的次数，
  每次访问递增；persistent workers 即便不重建数据集，也会因 occurrence
  或 epoch 变化得到新布局；
* `epoch` 始终参与种子，由 `set_epoch(epoch)` 显式写入。训练时每个 epoch
  必须调用一次（`diffpeak/engine.py` 已这样做），这样即使 `augment=False`
  也会得到逐 epoch 不同的提示布局；
* `augment=False` 时同一 epoch 内不推进 occurrence，重复读取同一索引会
  得到相同布局，便于评估与离线复现。

验证集若需要跨 epoch 固定布局，应每轮重建数据集、调用 `set_epoch(0)`，
并保持 `augment=False`（`diffpeak/engine.py` 的验证 loader 即如此）。相同
seed、worker 划分、epoch 和访问顺序可复现；不同 `num_workers` 或不同样本
访问顺序不保证逐样本完全一致。

## 4. `PeakExtractor`

```python
from diffpeak.prompts import PeakExtractor

extractor = PeakExtractor(config["peaks"])
# 也支持 PeakExtractor(**config["peaks"]) 与逐通道覆盖
coords_and_labels = extractor(
    heatmaps,                 # B x 2 x h x w（或单张 2 x h x w）
    image_size=1024,
    valid_sizes=[(1024, 1024)],  # 每项为 (height, width)
)
coords, labels = coords_and_labels[0]  # K x 2 (x, y), K
```

协议使用的 `sigma / h / threshold / min_distance` 数值统一见
`configs/3090.yaml` 的 `peaks`，本文不复制参数表。四个参数的含义和单位保持
如下：`sigma` 是提取前高斯平滑标准差（热图像素），`h` 是 h-maxima 相对高度，
`threshold` 是绝对响应阈值，`min_distance` 是同通道 NMS 最小间距（热图像素）。
`h=0` 时退化为普通局部极大。

每个通道可以独立覆盖：

```python
PeakExtractor({
    "sigma": 0.5,
    "channels": {
        "positive": {"threshold": 0.30},
        "negative": {"threshold": 0.25, "min_distance": 5},
    },
})
```

也接受 `positive` / `negative`、`pos` / `neg`、`0` / `1` 作为通道键。
顶层参数还可以写成 `{"positive": ..., "negative": ...}` 或
`(positive, negative)` 二元组。

提取顺序：

1. 每通道独立平滑；
2. `h_maxima`（`h=0` 时使用 `local_maxima`）得到峰区域；
3. 删除绝对响应低于阈值的峰；
4. 峰区域连通域标记，每个区域取一个代表点；平顶时选距区域中心最近者，
   再按 `(y, x)` 字典序固定平局；
5. 映射到 SAM2 坐标，按 `valid_sizes` 排除 padding；
6. 按分数降序做同通道距离 NMS（无固定点数上限）；
7. 正负坐标完全重合时，分数高者胜；分数相同正点优先。

注意：第 5 步在峰检测之后、NMS 之前过滤，因此 padding 峰不会参与 NMS
压掉有效峰。提取器是确定性的，不含随机数。

## 5. 回提诊断

设计文档明确要求：构造目标热图后，必须用**最终峰值提取器**回提一次，
检查目标峰是否在构造阶段就被合并或过滤。

### 5.1 `extract_numpy`

```python
from diffpeak.prompts import extract_numpy

points = extract_numpy(heatmaps, image_size=1024, valid_sizes=None,
                       config=None)
# points[i] = (coords float32 K x 2, labels int64 K)
```

纯 Numpy 输入/输出，便于离线评估。

### 5.2 `target_recovery_report`

```python
from diffpeak.prompts import target_recovery_report

report = target_recovery_report(
    heatmaps,                       # B x 2 x h x w
    expected_coords,                # 逐样本列表 / B x K x 2 / 单样本 K x 2
    expected_labels,                # 对应 0/1
    extractor=extractor,
    image_size=1024,
    valid_sizes=[(1024, 1024)],
    match_radius=2.0,
    expected_space="image",         # 或 "heatmap"
)
```

报告包含整体与逐通道的 `expected / predicted / matched / missed / extra`、
`precision / recall / f1`、`mean_match_distance`、逐样本明细，以及
`recovered`（所有期望点都在半径内找回且无多余峰）。空样本同时预测为空时
会计入 `empty_correct_count`。

### 5.3 数据集级诊断

```python
sample = dataset[0]
report = dataset.recovery_report(0, extractor=extractor)
points = dataset.get_target_points(0)
```

`recovery_report` 只做诊断，不参与训练损失，也不会改变提取器参数。
`check_targets.py` 已提供一个批量检查脚本（依赖安装后由用户手动运行）。
若报告显示大量 `missed`，优先检查：

* `target_min_distance` 是否与提取器 `min_distance` 一致；
* 目标高斯是否被叠加/合并（本实现使用 max，不是 sum）；
* 提取阈值是否高于目标峰的实际响应；
* `valid_sizes` 是否按 `(height, width)` 传入。

## 6. 依赖与配置来源

`requirements.txt` 的固定依赖；SAM2 安装与测试记录见 `docs/testing.md` 和 `docs/probe_3090.md`：

```text
Ubuntu 22.04 / Python 3.12
torch==2.8.0 / torchvision==0.23.0（CUDA 12.8）
numpy==1.26.4
scipy==1.13.1
scikit-image==0.24.0
Pillow==10.4.0
PyYAML==6.0.2
tqdm==4.66.5
SAM-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4
```

SAM2 从官方源码安装，安装时设置 `SAM2_BUILD_CUDA=0` 关闭可选 CUDA 扩展；权重使用
`sam2.1_hiera_base_plus.pt`，官方配置为 `configs/sam2.1/sam2.1_hiera_b+.yaml`。
SAM1 的 `segment-anything==1.0` 与 `sam_vit_b_01ec64.pth` 不再使用，旧 checkpoint
不能加载到 SAM2。

`h_maxima`、`distance_transform_edt`、`skeletonize` 分别来自
scikit-image / SciPy；数据 worker 会使用它们。数据侧所有数值起点（输入/热图
几何、`heatmap_sigma`、`target_min_distance`、`peaks`、正负点采样范围和 worker
数量）统一见 [`configs/3090.yaml`](../configs/3090.yaml)，本文不复制推荐参数表。
worker 数量不是纯性能开关：worker ID 参与现有数据 RNG，改变数量会改变增强和
提示随机流，正式实验必须冻结。CPU 预处理吞吐与 SAM2 B 阶段的 worker 比较见
`docs/probe_3090.md`；探针脚本使用 `--cpu-threads 2 --interop-threads 1` 限制线程。

模型侧的 batch size、梯度累积和 AMP 同样属于训练配置，不在数据模块内强制。

## 7. 已知边界与实验注意

* 骨架来自二值掩码，分叉处可能有短毛刺；`min_branch_length` 只抑制极短
  分支，不代表已做完整拓扑清理。
* 正点按随机间距采样，是**一组合理提示布局的采样**，不保证覆盖每条分支
  的每个端点；多 epoch 访问会产生不同布局（训练模式）。
* 256 热图对应 1024 输入时存在固有的量化误差；`512 x 512` 热图可作为
  单独实验评估这一项。
* 无道路样本的监督是“双通道全零”；有道路但未生成正峰时，模型应通过
  分割奖励得到低分，而不是由数据模块强行补全局最大点。
* 官方 valid/test 没有 mask，不能作为监督集；`unlabeled.json` 仅用于
  推理。测试集不得参与提示选择或奖励优化。
* 困难负点必须由外部提供；本模块没有自动困难负点挖掘。
