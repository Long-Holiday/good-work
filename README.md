# DiffPeak-SAM

依据 [`docs/model_grpo.md`](docs/model_grpo.md) 的 GRPO 后训练设计实现：**条件扩散生成双通道提示热图 → h-maxima + 距离 NMS → 道路适配 SAM2.1 → 同图组内 GRPO 后训练**。分割骨干为官方 **SAM2.1 Hiera-B+（base_plus）**。旧 SAM1/DDPO 猜想稿已移除，历史版本可从 Git 查看。

> 验证状态：SAM2.1 接入后完整回归测试 **293 项、169 项子测试通过**，包括已安装官方 SAM2 模块的梯度集成测试；3090 上真实 SAM2 权重的 A/B 全尺寸探针已完成实际更新、显存和吞吐检查。学习率与 C/D 验收继续进行，短程结果不代表完整训练收敛或多种子效果。SAM2 实测与 SAM1 历史基线分开记录于 [`docs/probe_3090.md`](docs/probe_3090.md)。历史测试见 [`docs/testing.md`](docs/testing.md)，日志使用见 [`docs/experiments.md`](docs/experiments.md)。

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

**SAM1 与 SAM2 的 checkpoint 不兼容**：`sam_vit_b_01ec64.pth` 等 SAM1 权重不能加载到 SAM2 结构，已有的 SAM1 训练快照也不能用于继续 SAM2 的 A/B/C/D。切换后必须用 SAM2.1 权重从 A 重新开始，并重新测量运行时与显存。代码不会自动下载权重。只加载可信的 checkpoint，训练快照使用 PyTorch 序列化，不能把陌生来源文件当成安全输入。

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

影像等比例缩放到最长边 1024、右下填充；SAM2 输入保持 RGB `[0,1]`，标准化在 SAM2 封装内部完成。道路骨架和采样位置在原标注分辨率构建，再映射到 256² 热图，不先缩小二值道路再求骨架。

## 3. 代码组织

```text
configs/3090.yaml           单卡实验配置
scripts/prepare_splits.py  数据清单与复现划分
diffpeak/
  data/                    数据读取、增强、随机监督布局
  prompts.py               固定峰值提取与目标回提检查
  models/
    sam.py                 SAM2.1 封装：冻结图像编码器、高分辨率特征与可适配解码器
    conditioning.py        细节 CNN、多尺度条件与辅助头
    unet.py                四尺度条件去噪网络
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

所有命令在项目根目录运行。下一阶段使用上一阶段的权重初始化；阶段内恢复则使用 `--resume`，不要把二者混用。

| 阶段 | 可更新部分 | 监督 |
|---|---|---|
| A | 仅 SAM2 mask decoder | 真实提示分割 |
| B | 细节/条件编码器 + 去噪 U-Net | 扩散 MSE + 热图重建 + 辅助 |
| C | 仅 SAM2 mask decoder | 扩散生成提示下的分割 |
| D | 仅去噪 U-Net | GRPO + 固定参考策略 KL（可选监督锚定） |

图像编码器与提示编码器始终冻结。A 和 C 只训练 SAM2 解码器；B 才训练条件编码器与 U-Net；D 只训练 U-Net，条件模块和 SAM2 均冻结。SAM2 解码器依赖图像编码器输出的高分辨率特征，迁移时必须显式保留并传入，不能只保留 64×64 图像嵌入。

正式训练使用线性 warmup + cosine 退火；各阶段数值集中于 [`configs/3090.yaml`](configs/3090.yaml)，选择依据与尚未验证的部分见 [`3090 探针记录`](docs/probe_3090.md)。B/C/D 验证基于生成提示；A 的真值提示验证单独作为诊断。

```bash
# 推荐先检查目标热图回提（不需要 SAM2 权重）
python check_targets.py --config configs/3090.yaml --samples 16

# 完整顺序训练；各阶段保留 stage_A.pth ... stage_D.pth
python train.py --config configs/3090.yaml --stage all

# 或分阶段执行；--init 只加载模型，不恢复优化器
python train.py --config configs/3090.yaml --stage A
python train.py --config configs/3090.yaml --stage B --init runs/diffpeak/stage_A.pth
python train.py --config configs/3090.yaml --stage C --init runs/diffpeak/stage_B.pth
python train.py --config configs/3090.yaml --stage D --init runs/diffpeak/stage_C.pth

# 阶段内恢复，默认用 checkpoint 保存的配置
python train.py --resume runs/diffpeak/last.pt

# 内部有标注验证集评估；默认使用 checkpoint 的峰值与扩散配置
python evaluate.py --checkpoint runs/diffpeak/stage_D.pth --split-file splits/val.json --output runs/evaluation

# 官方无标注 valid 只输出预测；不读取掩码
python predict.py --checkpoint runs/diffpeak/stage_D.pth --input DeepGlobe_Road/valid --output predictions/valid --save-points
```

以上正式训练/完整评估命令需准备 SAM2.1 预训练权重后执行，本次未运行完整训练或评估；SAM1 已有结果不能沿用。已在 `runs/grpo_validation/splits` 生成独立验证用划分，并完成 16 张真实样本的目标回提检查；CPU/GPU 测试与漏峰情况单独记录于 [`docs/testing.md`](docs/testing.md)。`--stage all` 按 A→B→C→D 使用阶段结束权重继续；`stage_X.pth` 为阶段结束完整快照，`last.pt` 为最近保存的完整 epoch，`best.pt` 按当前阶段验证 IoU 选择，会随新阶段重新记录。若希望每阶段使用最佳权重而非末轮权重，分阶段运行并把该阶段 `best.pt` 另存后传给下一阶段 `--init`。建议不同实验使用不同 `output_dir`（或用 `--experiment-name` 隔离），不要共享输出目录。

`predict.py`/`evaluate.py` 省略 `--config` 时使用 checkpoint 内的配置；显式传入 YAML 表示你主动变更协议。checkpoint 保存已解析的数据、划分、SAM2 和输出绝对路径；换机器时应显式提供含新路径的配置。恢复以完整 epoch 为单位，不承诺逐 batch 中断恢复。`train.py --resume` 会恢复 checkpoint 的优化器学习率与调度曲线，即使显式传入新 YAML；阶段内恢复应保持 batch size、梯度累积、划分长度和计划 epoch 数不变，变更训练预算请另用 `--init` 启动新实验。`evaluate.py` 的 `--limit` 只用于子集调试，不应与完整验证集结果混用。

当前 3090 配置为 batch size 1、梯度累积 4、AMP，并在实测后关闭 U-Net 激活检查点；D 保留 `rl.amp_init_scale=64`。A/B/C 的监督有效批次为 4。D 每图 4 条轨迹串行采样、轨迹暂存 CPU，其组语义不同于监督批次。

从旧 DDPO 快照迁移时，使用 `python train.py --config configs/3090.yaml --stage D --init <旧快照>`，显式传入新的 GRPO 配置并只加载模型权重；省略 `--config` 会继承旧快照配置，而不会自动启用本次学习率与预热设置。**不要**把旧 DDPO 快照直接 `--resume` 到 GRPO D。优化器、调度器与 RNG 状态本身可技术性加载，但旧 DDPO 的更新目标与 GRPO 不同，不应沿用其优化器和调度状态；其中冻结的 `ref_unet` 权重仍可复用，旧配置也可能缺少 `rl.algorithm`、`rl.point_cost_warmup_steps` 等键。需要保留旧实验时，请使用独立 `output_dir`。

### 实验日志与 TensorBoard

实验日志已通过文件/事件回读、异常关闭、RNG 隔离与微型 A–D/resume 集成测试；TensorBoard 2.21.0 本地 HTTP 页面及标量接口检查通过。日志已在 CPU 测试与真实 SAM2 有界探针中运行，但尚未验证完整多轮训练的日志轮转与长期曲线；SAM1 时期结果不能外推为 SAM2 结果。

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
- TensorBoard tag 按阶段独立，例如 `A/train_batch/loss`、`D/train_group/reward`、`A/train_epoch/loss`、`A/val_epoch/iou`、`A/system/epoch_seconds`。epoch 级轴为 1-based；batch/group 轴使用 `epoch_index * len(loader) + index`，其中 `epoch_index` 为该轮从 0 开始的内部序号，`index` 为该 epoch 内从 1 开始的 batch 序号，D 阶段改用 `group_index`。`--resume` 另起 session，`metadata.json` 记录 resume 快照路径、阶段与 epoch，避免把未保存 epoch 的点追加到旧曲线。

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

## 5. 扩散、峰值与 RL 协议

- SAM2.1 Hiera-B+：输入 1024²，冻结图像嵌入 256×64²，并保留各 256 通道的 256² / 128² 原始 FPN 特征；mask decoder 独立将其投影为 32 / 64 通道。
- 条件金字塔：256/128/64/32，通道 64/128/256/256；统一融合 SAM2 两级原始高分辨率 FPN，投影参数不与解码器共享，不再保留旧低分辨率条件分支或切换开关。旧结构 checkpoint 不支持直接初始化或恢复，新实验从官方 SAM2 权重执行 A→B→C→D。
- 两个独立提示通道：256²，既不互斥 Softmax，也不逐图 min-max。
- 训练扩散 1,000 步，生成 20 步；所有随机转移方差非零，包括末步，标准差下限 0.02。
- 固定峰值：平滑 sigma=0.5，h=0.05，绝对阈值 0.3，NMS 半径 4 热图像素。
- 每通道峰数可变，无固定 K；不设置隐含 top-k。异常密集热图可能带来很高解码成本，需要监控。
- 正点/负点一起输入 SAM2；没有正峰时输出空掩码，不强行补全全局最大点。点成本按提取出的峰总数统计：即使负点-only 候选被空掩码规则短路，冗余负峰仍受成本惩罚。
- 保留既有四尺度残差 U-Net，不强制加入参考方案中的瓶颈注意力，以兼容已有预训练权重；20 步采样、峰值参数和可选低噪声热图重建沿用现有实现，均需进一步消融校准。
- 默认奖励 `0.6*IoU + 0.4*clDice - point_cost*(Kpos+Kneg)/32`，其中 `reward.point_cost=0.05`。D 阶段 `rl.point_cost_warmup_steps=100` 时，策略点成本按**已处理的 D 组数**在 100 组内线性升到该值（第 1 组为 1/100）；`0` 表示禁用预热。组内奖励完全常量、没有实际 backward 的组也会推进预热，避免预热永远停在 0；验证始终使用未预热的基准成本。
- 原始 `checkpoints/sam2.1_hiera_base_plus.pt` 在训练和预测构造时都需要保留，即便训练快照已经包含 SAM2 权重；SAM1 权重 `sam_vit_b_01ec64.pth` 不可替代。完整快照较大，实验磁盘空间需预留。
- 可开启双向抽样路径代理（例如权重 0.5/0.3/0.2），**不是正式 APLS**；二值标注无法区分立交跨越。
- D 阶段用 GRPO：同一影像的 G 条轨迹组成一个组，优势为 `(R_i - mean(R)) / max(std(R), advantage_std_delta)`，均值和标准差只来自该组。`rl.algorithm=grpo` 显式标识协议。动作维联合 log-prob 对所有热图维度**求和**，没有用均值替代联合密度；默认裁剪半径 `rl.ppo_clip=1e-4` 是高维策略的谨慎起点，需监控实际 KL、奖励方差与更新比例。键名保留 `ppo_clip` 只是为了兼容旧配置。
- `rl.advantage_std_delta=1e-4` 是组内标准差的分母下限，不是“低方差组跳过阈值”：非零但方差很小的组仍按该下限缩放后参与学习，只有奖励完全相同的组优势为零。KL 同样对动作维度求和，再按样本/轨迹/时间步取平均。
- 每条轨迹的旧策略 log-prob 在该组 rollout 时保存并冻结，重放时只重算当前策略；它与长期参考策略不是同一个对象。首次从 B/C 进入 D 时，参考 U-Net 从当前预训练 U-Net 深拷贝，之后永久冻结，只用于 KL。从已有 D 快照初始化或恢复时保留其中的 `ref_unet`，避免把已经后训练的策略误当作原始参考；旧快照缺少参考时才从加载后的 U-Net 建立新参考，应记录这一实验差异。对峰值提取/SAM2 奖励不反传；策略梯度来自保存动作的重新计分。
- `rl.anchor_weight` 默认 `0.0`，即不启用监督锚定。参考 GRPO 设计没有把目标热图 MSE 列为必需项；设为非零时是**可选**监督锚定，只作为稳定项，不是当前默认协议。
- 推理默认沿用随机生成协议；确定性采样不能默认与训练等价。

## 6. 实验注意事项

消融与主流模型比较按 [实验协议](docs/experimental_protocol.md) 冻结数据、预算、模型选择及评估口径。运行参数集中于配置，资源实测及开发筛选集中于 [3090 探针记录](docs/probe_3090.md)，避免在多份文档维护重复参数表。SAM2 替换 SAM1 属于骨干变更：旧 SAM1 快照、A/B/C/D 结果和探针数据全部失效，必须用 SAM2.1 权重从 A 重新执行 A→B→C→D，并重新测量吞吐与显存。探针工具已归档到 `experiments/3090-throughput-probes-20261021` 分支，主分支仅保留训练实现。正式训练建议用 `--cpu-threads 2 --interop-threads 1` 限制 CPU 线程；SAM1 与 SAM2 的 worker 数、吞吐和显存均按骨干分开记录，参见实测文档。

先做 A→B→C 的监督链条再进行 D；D 可能因组内奖励相同、优势为零而停滞，不保证必然改善。先检查目标热图的 `target_recovery_report`，确保随机目标峰经最终提取器仍可恢复。困难负点只在具备阶段性假阳性数据时启用，不把普通背景采样冒充困难负点挖掘。

D 阶段至少记录终端奖励、组内奖励标准差、无效（零优势）组比例、正负点数、联合 log-ratio、KL 与裁剪比例。`advantage_std_delta` 只限制标准差分母；若许多组奖励完全相同，应检查探索是否不足，而不是把它当作已发生的学习。点成本预热只在前 100 个已处理 D 组内改变成本权重，比较实验时应固定该协议或显式设为 0。

建议分别记录 IoU、F1、clDice、正负点数分布、无正峰率、点落点正确率、多种子波动和包含峰值 CPU 处理的端到端耗时。指标聚合需区分逐图宏平均与全数据混淆矩阵微平均。无道路图的空预测得分规则见奖励实现。

若 OOM，先检查意外解冻和保留计算图，再降低 micro-batch、开启检查点并保留 AMP。worker 数主要影响主存和数据等待；改变 G、采样步数或分辨率会改变实验定义，需单独登记。实测显存范围与样本数见探针记录。

本实现是原设计的可扩展研究基线，不是复现已发表性能的成品；交叉注意力、自动困难负点闭环、正式 APLS 评估、细边界细化和全部消融实验需单独开展。

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

GPU smoke 使用 128² 输入、32² 热图、G=2 和 4 步采样，检查 AMP、GRPO 组内优势（奖励 `[0.4, 0.6]` 得到 `[-1, 1]`）、KL、逐步重算 log-prob、梯度与参数更新，而非完整 SAM2 分割效果。历史实测：完整 CPU 测试 221 项 OK；GPU smoke `passed=true`，重算 ratio 最大误差 0，B/D 峰值 allocated 339.51 / 369.08 MiB。该合成 smoke 不加载真实 SAM2 权重，也不替代完整训练。SAM2 接口测试及真实 3090 有界训练另见实测记录；不要把合成 smoke 的数字冒充完整模型占用。历史范围与未执行项见 [`docs/testing.md`](docs/testing.md)。
