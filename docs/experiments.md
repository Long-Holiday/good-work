# 实验与结果口径

本文是代码使用约定，不表示已开展实验或获得结果。用于消融、主流模型比较和 RTX 3090 24GB 训练的统一规则见 [正式实验协议](experimental_protocol.md)，其中区分已实现能力、待实测配置和正式运行前需补齐的条件。

## 最小实验路线

1. 固定 `splits/protocol_v1/*.json`（或登记过的等价三分清单）与依赖版本；不对官方无掩码 valid/test 计算指标，内部 test 只在全部选模完成后统一评估。划分与测试集使用历史见 [实验协议](experimental_protocol.md) 第 2 节。
2. 先运行 `python check_targets.py --config configs/3090.yaml --samples 16`，检查目标点回提 precision / recall、数量、最大定位误差及无正峰率。少量样本仅用于排错，正式定阈值需覆盖整个内部验证集。若仍在用旧两分清单，这 16 张可能落入新 test，必须按协议登记用途，不能据此声称测试集从未被查看。
3. A 阶段只训练 SAM2 解码器，确认真实提示下 SAM2 能学道路；SAM2 迁移后必须从 A 重新开始 A→B→C→D。A 的真值提示指标是受提示监督的诊断指标，不能与 B/C/D 自动生成提示成绩混称。
4. B 阶段训练细节/条件编码器与去噪 U-Net。先检查扩散 MSE 与低中噪声重建，再看生成热图幅度、正负峰数与无正峰率；只看扩散 MSE 不足以选择最终模型。
5. C 阶段只训练 SAM2 解码器，确认生成提示分布下解码器适配有效，再冻结解码器进入 D。
6. D 阶段只训练扩散 U-Net，使用 GRPO。同时记录终端奖励、组内奖励标准差、零优势组比例、联合 log-ratio、KL、裁剪比例与点数。若同图四条轨迹奖励完全一致，优势为零是正确行为，不应替换成跨图基线；非零但方差低于 `advantage_std_delta` 的组只按该下限缩放，仍会参与学习，不能误记为“已跳过”。
7. 完整实验以每张影像一次采样为主协议，多种子重复。不要用标注从多轨迹中挑最优结果当成无标注推理性能。

## 优先消融

| 对照 | 固定因素 | 改变因素 |
|---|---|---|
| B/C 对比 D | 条件、SAM2、峰值与推理采样 | 是否做 GRPO 后训练 |
| 不惩罚点数 | 全部模型与数据 | `reward.point_cost=0`，需重新后训练 |
| 点成本预热 | 模型、数据、点数成本 | `rl.point_cost_warmup_steps` 为 `0` 与配置值，需分别重新后训练 |
| 监督锚定 | 其余 GRPO 协议 | `rl.anchor_weight=0` 对比非零值；非零只是可选监督锚定，不是默认协议 |
| 加路径代理 | 其余协议 | 在实验 YAML 中启用非零路径权重；只作为独立消融，不改主协议 |
| 正点 vs 双通道 | 正通道与模型 | 负点是否参与，需定义训练一致的消融 |
| h-maxima vs 普通局部极大 | 数据和模型 | 重新校准且固定峰值提取协议 |
| 提示依赖性 | 同一模型 | 移除或打乱提示，仅用于评估诊断 |

确定性 CNN 热图基线、直接分割基线、正式 APLS、512² 热图和细边界细化是后续独立实验；当前没有为这些消融声称结果。

## D 阶段 GRPO 协议与迁移口径

当前 D 阶段配置显式写 `rl.algorithm=grpo`，表示 DDPO 已被同图组内相对优势的 GRPO 取代。当前实现规范是 [`docs/model_grpo.md`](model_grpo.md)，已被替代的旧设计稿不再保留在工作树。D 的学习率、warmup、裁剪半径等**数值起点统一见 `configs/3090.yaml`**，本文只保留语义，不复制参数表。

- `ppo_clip` 键名为兼容旧配置保留，语义是 GRPO 裁剪半径 epsilon，不是 PPO 的原始超参。
- `advantage_std_delta` 是组内优势分母的标准差下限，不是“跳过阈值”；非零但低方差的组仍按该下限缩放后参与学习，只有奖励完全相同的组优势为零。
- `point_cost_warmup_steps` 按**已处理的 D 组数**推进，不依赖该组是否真正产生 optimizer step；这样常量奖励组或 AMP 溢出不会让预热永久停在 0。验证固定使用未预热的基准 `reward.point_cost`，避免同一 checkpoint 的验证奖励随预热进度漂移；`0` 表示禁用。
- `anchor_weight` 默认 `0`；非零时是可选监督锚定（对目标热图额外加扩散 MSE），不是默认 GRPO 目标。
- D 阶段的 `lr`、`warmup_ratio`、`min_lr_ratio` 在 `stages.D` 中覆盖全局值；具体数值以配置为准，不在文档中重复。
- **两种“旧策略”不要混淆。** 组 rollout 时保存的 `old_log_prob` 在组内冻结并 detach，只作为概率比的分母；重放更新时重算当前策略。首次从 B/C 进入 D 时深拷贝当前预训练 U-Net 为长期参考，之后永久冻结，只用于 KL；从已有 D 快照初始化或恢复时保留其中的 `ref_unet`，缺失时才从已加载权重重建，需记录该协议差异。前者不是长期参考模型。
- **从 DDPO 快照迁移。** 旧 DDPO 模型可以用 `--init` 只加载权重来启动新的 D 阶段；不支持把旧 DDPO 快照直接 `--resume` 当作 GRPO 训练继续，因为优化器/调度器/RNG 可加载并不代表旧策略轨迹、组采样或参考策略语义兼容。旧配置也可能缺少 `rl.algorithm` 与 `rl.point_cost_warmup_steps`，应显式提供新 YAML 并使用独立输出目录。
- 以上均为待验证起点，本文不声称已运行 GRPO 实验或取得改进。

## 实验运行目录、日志与 TensorBoard

本节描述 `train.py` 的实验日志协议。TensorBoard 2.21.0 的 SummaryWriter/event accumulator 写读已在 `pytorch` 环境通过，`pip check` 无 broken requirements；尚未用真实 SAM2 权重运行完整 A–D 训练来验证真实训练负载下的日志轮转、flush 行为或曲线；SAM1 时期的日志结果不能视为 SAM2 证据。`evaluate.py`/`predict.py` 的既有输出不受本协议影响。

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
- `metrics.jsonl`：UTF-8 JSON Lines，追加写入，一行一个对象。每条记录含 `event` 和 ISO 8601 UTC `time`；指标记录为 `event="metrics"`，并带 `stage`（大写 A/B/C/D）、`scope`（如 `train_batch`/`train_group`/`train_epoch`/`val_epoch`/`system`）、`step`、1-based `epoch`，具体标量放在 `metrics` 对象中，例如 batch 的损失分量、D 组的奖励/优势统计/KL/裁剪比例、epoch 的汇总损失与耗时、验证的 IoU/F1/clDice。session 开始/结束使用 `session_start`/`session_end` 记录。
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

- 阶段前缀使用实际的 `A`/`B`/`C`/`D`，因此 A/C 的 SAM2 解码器指标与 B/D 的 U-Net 指标不会混称。
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

输入 1024²、热图 256² 是当前硬编码的基础几何，不是可随意调换的缩放开关。3090 的 batch、AMP、U-Net 检查点、冻结 SAM2 编码器、串行轨迹和 CPU 暂存等取值统一见 `configs/3090.yaml`，已测范围见 [探针记录](probe_3090.md)。改变实验因素需单独登记，不能把降配悄悄混入主表。

SAM2 替换 SAM1 是骨干变更：SAM1 的 checkpoint、A/B/C/D 快照和已有探针结果都不再适用，必须用 SAM2.1 Hiera-B+ 权重（`sam2.1_hiera_base_plus.pt`，官方配置 `configs/sam2.1/sam2.1_hiera_b+.yaml`）从 A 重新训练并重新测量。3090 探针脚本固定 `--cpu-threads 2 --interop-threads 1` 限制 CPU 线程；SAM1 历史数字与 SAM2 新探针分开保存，不交叉复用。

单位语义必须保持：目标采点间距使用**原始标注像素**；热图高斯和 NMS 使用**热图像素**；路径匹配半径使用**分割评估像素**；提示坐标交给 SAM2 时使用其 1024 输入空间的 `(x,y)`。更换影像分辨率时应重新检查这些参数的物理含义。

先确定并冻结峰值提取参数，再运行 C/D。D checkpoint 推理必须复用训练的峰值和随机采样协议；改变它们属于新实验，不是等价加速。
