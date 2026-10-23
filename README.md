# DiffPeak-SAM

**单通道有符号山脊场 → 全冻结 SAM → 道路分割**：ABC 监督预训练与 D 扩散轨迹 GRPO 只更新条件编码器和 DiT。SAM2.1 的图像、提示、掩码模块与高分辨率投影全阶段冻结；不再提取离散正负点。当前协议为 `signed_ridge_v1`。当前协议见 [`docs/joint_training.md`](docs/joint_training.md)；旧四阶段实验与 `docs/model_grpo.md` 作为历史记录。

> 旧四阶段的 GPU 探针与测试结果不代表新联合训练已收敛；新协议需要重新测量完整采样反传的显存、吞吐及验证集质量。历史实测见 [`docs/probe_3090.md`](docs/probe_3090.md)，日志使用见 [`docs/experiments.md`](docs/experiments.md)。

## 1. 环境与权重（自行安装）

3090 实测环境：**Python 3.12 / PyTorch 2.8.0+cu128 / torchvision 0.23.0+cu128 / RTX 3090 24GB / 14 vCPU / 90 GiB 内存**。科学计算依赖按 `requirements.txt` 安装在隔离环境；本地历史测试环境见 `requirements-pytorch-tested.txt`。

PyTorch CUDA wheel 包含运行时，常规使用不需要另装 CUDA toolkit。服务器镜像已提供 Python 3.12 和 PyTorch 2.8.0+cu128 时，使用该环境即可，不必再创建环境。若需要隔离环境，可用已有的 conda 创建（Ubuntu 22.04 的系统 Python 默认不是 3.12）：

```bash
# 仅在需要新环境时执行
conda create -n diffpeak python=3.12 -y
conda activate diffpeak
```

在项目根目录执行安装；PyTorch 使用官方 cu128 索引，其余依赖使用默认 PyPI 索引：

```bash
nvidia-smi
python --version
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
# 复用已装的 torch，避免构建隔离环境重新下载整套 CUDA/PyTorch。
python -m pip install 'setuptools>=61' wheel
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
```

如可选驱动，建议使用 **570.26 或更新版本**（CUDA 12.8 GA 对应 Linux 驱动）。CUDA 12.x 的 minor compatibility 最低 Linux 驱动为 525.60.13，但旧驱动存在新特性和 PTX 限制，不能仅凭版本下限保证所有算子可运行；已有服务器驱动先用下面的检查确认。`nvidia-smi` 中的 CUDA Version 表示驱动支持的 CUDA 版本，PyTorch 实际使用的运行时看 `torch.version.cuda`。WSL2 应使用 Windows 侧驱动。

安装后检查版本、GPU 运算与 torchvision CUDA 算子：

```bash
python - <<'PY'
import sys
import torch
import torchvision
from torchvision.ops import nms

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
assert sys.version_info[:2] == (3, 12)
assert torch.__version__.split("+")[0] == "2.8.0"
assert torchvision.__version__.split("+")[0] == "0.23.0"
assert torch.version.cuda == "12.8"
assert torch.cuda.is_available(), "CUDA 不可用，请检查 GPU 分配与 NVIDIA 驱动"
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(256, 256, device="cuda", requires_grad=True)
(x @ x).sum().backward()
assert x.grad is not None and torch.isfinite(x.grad).all().item()
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device="cuda")
scores = torch.tensor([0.9, 0.8], device="cuda")
assert nms(boxes, scores, 0.5).tolist() == [0]
torch.cuda.synchronize()
print("CUDA / torchvision 检查通过")
PY

# 不需要 SAM2 权重：验证 GRPO、AMP 与梯度更新的小规模 GPU smoke
python scripts/gpu_smoke.py
```

固定依赖见 `requirements.txt`，NumPy 1.26.4 等版本支持 Python 3.12。其中 `tensorboard>=2.18,<3` 用于实验日志与 TensorBoard 可视化。不需要 diffusers、transformers、xformers、OpenCV 或自定义 CUDA 编译扩展。

SAM2 以官方源码安装，版本固定在 commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`；`requirements.txt` 已包含该依赖。上面的 `SAM2_BUILD_CUDA=0` 用于关闭可选 CUDA 扩展，避免额外编译；本实现不需要该扩展。模型使用官方 **SAM2.1 Hiera-B+** 配置 `configs/sam2.1/sam2.1_hiera_b+.yaml`；从 [SAM2 官方仓库](https://github.com/facebookresearch/sam2) 获取权重 `sam2.1_hiera_base_plus.pt`，放到 `checkpoints/`，或同时修改 YAML 的 `model.sam_checkpoint`、`model.sam_config` 与 `model.sam_architecture`（三者必须对应同一 Hiera 尺寸，构建时会校验）。官方权重地址：

```text
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

**SAM1 与 SAM2 的 checkpoint 不兼容**：`sam_vit_b_01ec64.pth` 等 SAM1 权重不能加载到 SAM2 结构，已有的 SAM1 训练快照也不能用于继续 SAM2 的 ABC/D。切换后必须用 SAM2.1 权重从 ABC 重新开始，并重新测量运行时与显存。代码不会自动下载权重。只加载可信的 checkpoint，训练快照使用 PyTorch 序列化，不能把陌生来源文件当成安全输入。

## 2. 数据与划分

本地数据结构：

```text
DeepGlobe_Road/
  train/100034_sat.jpg
  train/100034_mask.png
  valid/*_sat.jpg
  test/*_sat.jpg
  metadata.csv
  class_dict.csv
```

本次目录检查发现 `train` 有 **6,226 对**影像/标注，`valid`、`test` 无掩码。后两者只能做无标注推理，不能直接用于验证指标、提示监督或 RL 奖励。

在项目根目录执行：

```bash
python scripts/prepare_splits.py --root DeepGlobe_Road --output splits/protocol_v1 \
  --seed 42 --val-fraction 0.1 --test-fraction 0.1
```

生成 train/val/test **4,980 / 623 / 623** 的有标注清单，另有 `unlabeled.json`、`protocol.json`。默认 `--test-fraction 0` 仍兼容旧两分；正式比较使用上面的三分配置，训练不读取 test。脚本拒绝覆盖已有划分。按原影像划分后才做增强；随机划分不保证地理独立，有分组信息时优先区域隔离。旧诊断与新 test 的样本使用历史须按实验协议登记；内部结果不等于官方隐藏测试集成绩。

影像等比例缩放到最长边 1024、右下填充；SAM2 输入保持 RGB `[0,1]`，标准化在封装内部完成。同步增强后构造单通道 256² 山脊场，中心线为 +1、有效背景为 -1。独立有效掩码排除 padding，细道路投影保留率由目标检查报告。

## 3. 代码组织

```text
configs/3090.yaml           单卡实验配置
scripts/prepare_splits.py  数据清单与复现划分
diffpeak/
  data/                    数据读取、增强、有符号山脊场与有效区域
  data/ridge.py            山脊场构造、分区与细道路投影
  models/
    sam.py                 SAM2.1 封装：全冻结模型、高分辨率特征与可微密集提示
    conditioning.py        细节 CNN、多尺度条件与辅助头
    dit.py                 条件 DiT 去噪器（默认）
    diffusion.py           噪声监督、随机采样、轨迹对数概率
  grpo.py                  组内优势、裁剪策略损失与固定方差高斯 KL
  schedules.py             线性 warmup + cosine 退火
  losses.py                BCE / Dice / soft-clDice 与辅助损失
  rewards.py               IoU、clDice、可选双向路径代理奖励
  engine.py                阶段冻结、训练、验证与 checkpoint
  experiment_logging.py    实验会话、JSONL/文本日志、TensorBoard
  inference.py             无标注推理与几何恢复
train.py / predict.py / evaluate.py
```

## 4. 训练阶段

| 阶段 | 更新参数 | 目标 |
|---|---|---|
| ABC | 条件编码器、DiT | 扩散、四区域场重建、辅助道路/骨架、一步重建与纯噪声完整采样分割 |
| D | 条件编码器、DiT | 扩散轨迹 GRPO、固定参考 KL、可配置完整 ABC 锚定 |

SAM 始终冻结且 eval，分割监督仍保留场输入的梯度。验证与推理直接将单通道场按固定尺度送入 SAM，再对概率阈值化。D 使用相同的确定性分割评分，随机性来自扩散轨迹；不采样终端 Bernoulli 掩码。

```bash
# 无需 SAM 权重：检查场、padding 和细道路保留
python check_targets.py --config configs/3090.yaml --samples 16

# 先准备官方 SAM2.1 权重，再检查标注场提示是否有效
python scripts/diagnose_ridge_sam.py --help

# ABC → D，使用独立实验输出目录
python train.py --config configs/3090.yaml --stage all --experiment-name ridge-v1
python train.py --config configs/3090.yaml --stage D --init runs/diffpeak/stage_ABC.pth
python train.py --resume runs/diffpeak/last.pt

# 显式迁移旧权重兼容部分，重新开始训练；不恢复 SAM 或输入输出头
python train.py --config configs/3090.yaml --stage ABC --init old.pt --migrate-legacy

python evaluate.py --checkpoint runs/diffpeak/stage_D.pth --split-file splits/protocol_v1/val.json --output runs/evaluation
python predict.py --checkpoint runs/diffpeak/stage_D.pth --input DeepGlobe_Road/valid --output predictions/valid
```

新快照为版本 5，严格记录 `signed_ridge_v1` 表示、单通道、SAM 全冻结与提示映射。旧双通道或 SAM 微调快照不能普通 `--init`/`--resume`；只允许上述显式迁移并报告跳过的参数，不迁移优化器、参考或训练进度。阶段内恢复以完整 epoch 为单位，恢复优化器、调度器、GradScaler、参考模型与 RNG；保持数据划分、吞吐及训练预算一致。

`last.pt` 保存最近完整 epoch，`stage_ABC.pth`/`stage_D.pth` 保存阶段结束快照，`best.pt` 按当前阶段验证 IoU 选择。`--init` 开始新实验，新 D 的参考从当前权重建立；`--resume` 恢复原 D 参考。旧双通道吞吐配置只作为起点，新协议显存、吞吐和质量需重新实测。详细语义见 [训练说明](docs/joint_training.md)。

### 实验日志与 TensorBoard

实验日志已通过文件/事件回读、异常关闭、RNG 隔离与微型 ABC/D/resume 集成测试；TensorBoard 2.21.0 本地 HTTP 页面及标量接口检查通过。日志已在 CPU 测试与真实 SAM2 有界探针中运行，但尚未验证完整多轮训练的日志轮转与长期曲线；SAM1 时期结果不能外推为 SAM2 结果。

- `--output-dir PATH` 覆盖 YAML `output_dir`，作为输出根目录；`--experiment-name NAME` 在其下建立独立子目录，有效输出目录为 `<output_dir>/NAME`（省略时沿用原 `output_dir`；不与 `--resume` 同用）；`--no-tensorboard` 关闭本次事件写入，其余日志照常。
- 每次启动（包括每次 `--resume`）都在有效输出目录下新建 `<有效输出目录>/experiments/<UTC时间戳>_<随机ID>/`（例如 `20261021T121500.123456Z_ab12cd34`），其中写入 `config.json`、`metadata.json`、`train.log`（stdout/stderr tee）、`metrics.jsonl`（epoch 与 batch/group 结构化指标）和 `tensorboard/` 事件。`experiments/` 之外的既有 `metrics.json`、`last.pt`、`best.pt`、`stage_X.pth` 保持原位置和用途。
- YAML 增加：

  ```yaml
  logging:
    tensorboard: true
    log_every: 10
    flush_secs: 30
  ```

  `log_every` 控制 batch/group 记录间隔（每轮首批、末批也记录）；训练 epoch 汇总每轮记录，验证按 `train.val_every` 及末轮记录；`flush_secs` 控制 TensorBoard 事件刷新周期（文本与 JSONL 按写入刷新），缺省协议即上例。
- TensorBoard tag 按阶段独立，例如 `ABC/train_batch/loss`、`D/train_group/reward`、`ABC/train_epoch/loss`、`ABC/val_epoch/iou`、`ABC/system/epoch_seconds`。epoch 级轴为 1-based；batch/group 轴使用 `epoch_index * len(loader) + index`，其中 `epoch_index` 为该轮从 0 开始的内部序号，`index` 为该 epoch 内从 1 开始的 batch 序号，D 阶段改用 `group_index`。`--resume` 另起 session，`metadata.json` 记录 resume 快照路径、阶段与 epoch，避免把未保存 epoch 的点追加到旧曲线。

```bash
# 独立实验子目录；有效输出目录为 runs/diffpeak/grpo-baseline
python train.py --config configs/3090.yaml --stage all --experiment-name grpo-baseline

# 覆盖输出根目录后仍可用实验名隔离
python train.py --config configs/3090.yaml --stage all --output-dir runs/ablation --experiment-name seed43

# 只关闭 TensorBoard，config/metadata/train.log/metrics.jsonl 仍照常写入
python train.py --config configs/3090.yaml --stage all --no-tensorboard

# 查看事件
conda activate pytorch
tensorboard --logdir runs/diffpeak --port 6006
```

浏览器打开 `http://localhost:6006`。若使用 `--output-dir` 或 `--experiment-name`，将 `--logdir` 指向对应的有效输出目录（或其 `experiments/`）即可。详细字段、tag 与 resume 隔离口径见 [`docs/experiments.md`](docs/experiments.md)。

## 5. 场与 GRPO 协议

- 条件金字塔和 DiT 主体保持原尺寸：输入 1024²、场 256²，四尺度条件通道 64/128/256/256，DiT patch8/hidden256/depth6/heads8。
- 扩散直接处理有符号场，不做逐图 min-max、通道 Softmax 或旧仿射变换；不加入旧背景均值补偿。带噪状态无界，终端场裁剪到 [-1,1] 后以 `U=prompt.scale*H` 提示 SAM，padding 固定为0。
- 随机转移包括末步均保持正标准差；记录原始高斯动作后再裁剪终端输出，轨迹概率按原始动作重放。
- 每图 G 条轨迹独立形成奖励组，优势为 `(R-mean(R))/max(std(R),delta)`；常量组优势为0，低方差非零组仍学习。高斯联合 log-prob 与 KL 对动作维求和，再按样本、轨迹和时间步平均。
- 默认奖励 `0.6*IoU+0.4*clDice`，路径代理默认关闭，候选之间固定奖励随机种子。不奖励点数、峰谷、高频或幅度对比。
- D 固定参考条件编码器与 DiT，不创建参考 SAM。`anchor_weight=0.01` 保留完整 ABC 监督锚定，设0消融。单通道改变概率尺度，裁剪、KL 与更新幅度需重新校准。

## 6. 实验边界

正式训练前先验证标注山脊场和二值 mask 提示在真实冻结 SAM 上的可用性及固定尺度，再比较确定性同场预测基线、监督扩散、监督加 GRPO、关闭锚定。主报告采用固定协议单次生成与跨种子波动。多候选标注最优只作潜力诊断，路径奖励不是正式 APLS。完整实验顺序及风险见 [方案](docs/signed_ridge_diffusion_plan.md)。

持续记录 IoU/F1/clDice、边界与细路表现、空预测率、场分布、提示干预响应、奖励方差、KL 和裁剪比例。仅扩散损失下降不能证明有效收敛。旧双通道和旧四阶段实测保留作历史记录，不代表当前协议性能。

## 7. 测试入口

```bash
# 完整 CPU 测试；包括官方 SAM2 模块的随机权重集成
conda run -n pytorch python -m unittest discover -s tests -v

# GRPO / 调度 / 引擎子集
conda run -n pytorch python -m unittest tests.test_grpo tests.test_schedules tests.test_engine_grpo -v

# RTX 3050 小规模 GPU 检查（模拟 SAM2 特征，不需要真实 SAM2 权重）
conda run -n pytorch python scripts/gpu_smoke.py

# 语法编译检查
conda run -n pytorch python -m compileall -q diffpeak tests scripts train.py predict.py evaluate.py check_targets.py
```

GPU smoke 使用微型条件 DiT 和模拟 SAM 特征验证 CUDA/AMP、单通道采样、概率重放、GRPO 更新和冻结状态，不加载真实 SAM2.1 预训练权重，不能替代正式训练、提示有效性或分割质量评估。当前验收与历史记录见 [测试说明](docs/testing.md)。
