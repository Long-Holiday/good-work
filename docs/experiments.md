# 实验与结果口径

当前模型使用 `signed_ridge_v1`，单通道有符号场、SAM 全阶段冻结，ABC/D 都只更新条件编码器与 DiT。训练和检查点语义以 [训练说明](joint_training.md) 为准，实验设计与尚未验证项见 [山脊场方案](signed_ridge_diffusion_plan.md)。下述日志接口继续使用。

先用 `check_targets.py` 检查目标几何、细路保留和有效区域，再用 `scripts/diagnose_ridge_sam.py` 在内部验证集比较标注场/二值提示、固定尺度及提示干预。正式训练前确认冻结 SAM 对提示有效响应。主实验依次比较确定性同场预测、监督扩散、监督后 GRPO，以及关闭监督锚定；不得把多候选标注最优结果报告为可部署成绩。

ABC 保留完整采样分割监督，不能只凭扩散损失选模。D 每图独立成组，奖励采用确定性分割和固定奖励采样种子。记录生成场分布、空预测率、IoU/F1/clDice、组内奖励方差、活跃组比例、Gaussian KL、联合 log-ratio、裁剪比例、实际更新数与监督锚定。没有点数、峰值、点成本预热或终端 Bernoulli 策略日志。

SAM 全冻结参考无需复制；D 只保存固定参考条件编码器与 DiT。普通 init 与 resume 都严格验证新协议；显式 `--init --migrate-legacy --config` 仅迁移兼容扩散权重，另起实验，不恢复训练状态。单通道改变动作维与概率尺度，旧超参数及吞吐均需重新校准。

## 实验运行目录、日志与 TensorBoard

本节描述 `train.py` 的实验日志协议。TensorBoard 2.21.0 的 SummaryWriter/event accumulator 写读已在 `pytorch` 环境通过，`pip check` 无 broken requirements；尚未用真实 SAM2 权重运行完整 ABC/D 训练来验证真实训练负载下的日志轮转、flush 行为或曲线；SAM1 时期的日志结果不能视为 SAM2 证据。`evaluate.py`/`predict.py` 的既有输出不受本协议影响。

### 输出目录与 CLI

| 参数 | 作用 |
|---|---|
| `--output-dir PATH` | 覆盖 YAML 的 `output_dir`，作为输出根目录 |
| `--experiment-name NAME` | 有效输出目录变为 `<output_dir>/NAME`，用于隔离独立实验；省略时沿用原 `output_dir` |
| `--no-tensorboard` | 本次不写 TensorBoard 事件；其他日志照常 |

`NAME` 应为单层目录名，不要包含路径分隔符、绝对路径或 `..`。`--experiment-name` 用于启动新实验，不与 `--resume` 同用；resume 默认沿用 checkpoint 记录的训练目录，也可用 `--output-dir` 显式指向该目录，并在其中新建 session。有效输出目录确定后：

- checkpoint 与既有汇总文件仍直接位于有效输出目录根部，保持现有语义：`metrics.json`、`last.pt`、`best.pt`、`stage_X.pth`。
- 每次 `train.py` 启动（包括每次 `--resume`）都新建一个独立 session：

```text
<有效输出目录>/
  metrics.json                 # 既有训练历史汇总，保持现有行为
  last.pt / best.pt / stage_X.pth
  experiments/
    <UTC时间戳>_<随机ID>/          # 例如 20261021T121500.123456Z_ab12cd34
      config.json
      metadata.json
      train.log
      metrics.jsonl
      tensorboard/
        events.out.tfevents...
```

时间戳为 UTC（含微秒）加 8 位随机十六进制 ID，避免同一秒启动或多个并发进程覆盖。`tensorboard/` 仅在 TensorBoard 启用时创建；`--no-tensorboard` 时其余四个文件仍照常写入。若未安装 TensorBoard 且未使用 `--no-tensorboard`，启动时应报错并提示安装 `tensorboard` 或关闭该功能。

示例：

```bash
# 有效输出 runs/diffpeak/grpo-baseline
python train.py --config configs/3090.yaml --stage all --experiment-name grpo-baseline

# 有效输出 runs/ablation/seed43
python train.py --config configs/3090.yaml --stage all \
  --output-dir runs/ablation --experiment-name seed43

# 关闭 TensorBoard
python train.py --config configs/3090.yaml --stage all --no-tensorboard
```

### YAML 配置

`logging.*` 的当前数值统一见 `configs/3090.yaml`，文档不复制参数表。语义保持如下：`tensorboard` 控制是否写事件，`--no-tensorboard` 优先覆盖为关闭；`log_every` 表示每处理多少个 batch（D 阶段为 group）记录一次，每轮首批、末批也记录，训练 epoch 汇总始终记录，验证指标按 `train.val_every` 及末轮记录；`flush_secs` 只控制 TensorBoard 事件刷新周期，`metrics.jsonl` 与 `train.log` 按行/每次写入刷新，不依赖该周期。`log_every`、`flush_secs` 转为整数后下限为 1；CLI 加载配置时补齐默认值，配置及 CLI 覆盖写入 session 的 `config.json`。

### Session 文件内容

- `config.json`：本次运行实际解析后的配置，包含 YAML、checkpoint 内配置、CLI 覆盖和日志配置；路径应尽量为绝对路径。
- `metadata.json`：包含 `session_id`、`status`、启动/结束时间、持续时间、命令行、Python/PyTorch/CUDA/conda 环境、TensorBoard 是否启用、git revision/status 与异常信息；resume 时另记 `resume`、`resume_stage`、`resume_epoch`，并记录本次执行阶段与 seed。
- `train.log`：训练进程 stdout 与 stderr 的合并 tee，包含普通进度、警告与异常 traceback；写入时刷新。
- `metrics.jsonl`：UTF-8 JSON Lines，追加写入，一行一个对象。每条记录含 `event` 和 ISO 8601 UTC `time`；指标记录为 `event="metrics"`，并带 `stage`（大写 ABC/D）、`scope`（如 `train_batch`/`train_group`/`train_epoch`/`val_epoch`/`system`）、`step`、1-based `epoch`，具体标量放在 `metrics` 对象中，例如 batch 的损失分量、D 组的奖励/优势统计/KL/裁剪比例、epoch 的汇总损失与耗时、验证的 IoU/F1/clDice。session 开始/结束使用 `session_start`/`session_end` 记录。
- `tensorboard/`：TensorBoard 事件文件；关闭 TensorBoard 时不创建。

根目录的 `metrics.json` 继续按既有逻辑保存训练历史，不替代每个 session 的 `metrics.jsonl`。

### TensorBoard tag 与坐标轴

tag 以阶段名为第一级，避免同一输出目录顺序执行 A→B→C→D 时把不同训练目标混在一条曲线上：

```text
<stage>/train_batch/<metric>    # 每个 batch 的损失等，例如 A/train_batch/loss
<stage>/train_group/<metric>    # D 阶段每个 group 的奖励等，例如 D/train_group/reward
<stage>/train_epoch/<metric>    # 每轮训练汇总，例如 A/train_epoch/loss
<stage>/val_epoch/<metric>      # 每轮验证，例如 A/val_epoch/iou
<stage>/system/<metric>         # 耗时/学习率/AMP/显存，例如 A/system/epoch_seconds
```

- 阶段前缀使用实际的 `A`/`B`/`C`/`D`，因此 A/C 的 SAM2 解码器指标与 B/D 的 DiT 指标不会混称。
- epoch 级标量（`train_epoch`、`val_epoch`、按 epoch 统计的 `system`）使用 1-based epoch 作为 step，即第一个完整 epoch 为 1。
- batch/group 标量使用 `global_step = epoch_index * len(loader) + index`：`epoch_index` 为该轮从 0 开始的内部序号，`index` 为该 epoch 内从 1 开始的 batch 序号；D 阶段以 `group_index` 替代 `batch_index`（`len(loader)` 为该阶段训练 loader 长度）。因此第一个 epoch 的第一个 batch/group 为 step 1，同一 session 内该轴单调递增且跨 epoch 不重叠。`metrics.jsonl` 中的 `epoch` 字段仍是 1-based。
- `metrics.jsonl` 中指标记录的 `step` 与 TensorBoard 标量使用同一个 step。
- 事件文件与 session 一一对应；不要把不同 session 追加到同一事件文件。

### Resume 的 session 隔离

每次 `--resume` 都创建**新的 session 目录**，不向旧 session 追加：

- 新 session 的 `metadata.json` 记录 `resume`（快照路径）、`resume_stage` 和 `resume_epoch`；`config.json` 记录本次实际解析后的配置。
- 旧 session 的 `metrics.jsonl`、`train.log` 与 TensorBoard 事件保持不变。
- epoch 级记录从 `resume_epoch + 1` 开始；batch/group 轴按上式继续。由于 `last.pt` 只对应最后一个完整 epoch，中断 epoch 的更新不会写入 checkpoint；另起 session 可避免把未保存 epoch 的点与已保存 epoch 重叠，或把两次运行的曲线错误拼接。
- 若 resume 的阶段已完成并继续后续阶段，stage 前缀仍会把各阶段曲线分开。

### 查看 TensorBoard

```bash
conda activate pytorch
tensorboard --logdir runs/diffpeak --port 6006
```

浏览器打开 `http://localhost:6006`。若使用了 `--output-dir` 或 `--experiment-name`，把 `--logdir` 指向对应的有效输出目录；只想看某个实验的所有 session 时，可指向 `<有效输出目录>/experiments`。TensorBoard 会递归发现各 session 的 `tensorboard/` 事件目录。

## 指标约定

- 同时报 `macro IoU/F1` 与由 TP/FP/FN 累加后计算的 `micro IoU/F1`，不要二者混用。
- clDice 对二值掩码骨架计算，训练 soft-clDice 是可微近似，不是完全相同的数值。
- 空预测且空标注的质量得分为 1，仅一方为空则为 0。报告空图比例，避免宏平均误导。
- 路径指标是双向骨架图抽样代理，基于八邻域像素图，不能区分跨越与交叉，也不能冠名正式 APLS。
- 统计正负点均值/中位数/P95、有道路但无正峰率、点落点正确率，并明确坐标量化容差。
- 耗时统计包括图像编码、细节条件、配置的扩散采样步数、GPU→CPU 热图复制、h-maxima/NMS、SAM2 解码及输出恢复；单独报模型内核耗时不能冒充端到端耗时。

## 配置与内存

输入 1024²、热图 256² 是当前硬编码的基础几何，不是可随意调换的缩放开关。3090 的 batch、AMP、DiT 激活检查点、冻结 SAM2 编码器、串行轨迹和 CPU 暂存等取值统一见 `configs/3090.yaml`，已测范围见 [探针记录](probe_3090.md)。改变实验因素需单独登记，不能把降配悄悄混入主表。

去噪器从 ConditionalUNet 换成 ConditionalDiT 是架构变更：旧 U-Net 的 checkpoint 参数结构与 DiT 不兼容，`--resume`/`--init` 会被严格加载显式拒绝，不会静默使用随机权重。U-Net 时期已完成的 A/B/C/D 结果与探针数字保留为历史记录，不能作为 DiT 的实测依据；U-Net/DiT 对照需单独登记架构因素并匹配实验协议。新实验需从官方 SAM2.1 权重重新执行 A→B→C→D。

SAM2 替换 SAM1 是骨干变更：SAM1 的 checkpoint、A/B/C/D 快照和已有探针结果都不再适用，必须用 SAM2.1 Hiera-B+ 权重（`sam2.1_hiera_base_plus.pt`，官方配置 `configs/sam2.1/sam2.1_hiera_b+.yaml`）从 A 重新训练并重新测量。3090 探针脚本固定 `--cpu-threads 2 --interop-threads 1` 限制 CPU 线程；SAM1 历史数字与 SAM2 新探针分开保存，不交叉复用。

单位语义必须保持：目标采点间距使用**原始标注像素**；热图高斯和 NMS 使用**热图像素**；路径匹配半径使用**分割评估像素**；提示坐标交给 SAM2 时使用其 1024 输入空间的 `(x,y)`。更换影像分辨率时应重新检查这些参数的物理含义。

先确定并冻结峰值提取参数，再运行 C/D。D checkpoint 推理必须复用训练的峰值和随机采样协议；改变它们属于新实验，不是等价加速。
