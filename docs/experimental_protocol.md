# DiffPeak-SAM 实验协议：公平比较与 RTX 3090 训练

状态：**开发探针已开展，正式比较协议待冻结**。本文只规定比较规则；可执行数值以 [`configs/3090.yaml`](../configs/3090.yaml) 为唯一来源，目标卡实测集中于 [3090 探针记录](probe_3090.md)。短程开发结果不等于正式训练结果或全局最优超参数。

## 0. 单一来源与顺序

配置唯一来源是 `configs/3090.yaml`：每个正式实验从它派生完整 YAML，CLI 不会自动合并基础配置。比较规则（数据权限、预算、选模、指标、资源口径）以本文为准；目标卡实测写入 `docs/probe_3090.md`，由 `scripts/probe_3090.py` 和 `scripts/benchmark_loader.py` 产出，系统配置和效果参数分别标明测量范围。登记模板见 [`experiment_registration_template.yaml`](experiment_registration_template.yaml)，它不是 `train.py --config` 配置。

执行顺序：冻结三分数据 → 用开发种子 `2026` 在 3090 跑缩短预算的 A/B/C/D 与基线可运行性 → 据实测冻结阶段长度、超参和预算 → 用 `42/43/44` 跑主表与核心消融 → 全部选模完成后统一评估 test。

## 1. 公平性规则

| 轨道 | 固定 | 允许变化 | 可支持结论 |
|---|---|---|---|
| 主流自动分割 | 划分、训练标注、输入视野、评估器、测试预算及调参机会 | 模型、适合模型的优化器和学习率；公开预训练须披露 | 同一任务协议下的系统表现；预训练不同不单独归因于架构 |
| 机制消融 | 同种子父 checkpoint、预训练、数据顺序、训练预算、非目标超参、推理协议 | 一个预先声明的因素 | 该因素在本设置下的贡献 |
| 等计算 | 同一 3090、计时边界、GPU 小时上限 H | 各方法在 H 内的训练分配 | 给定硬件预算下的精度与效率 |
| 提示诊断 | 同一模型和评估集 | 真值提示、删除或打乱提示 | 提示依赖性与上限；不进入自动分割主表 |

主表与资源表分开报告；“相同 epoch”“相同 optimizer step”“相同 GPU 小时”都不能单独称为所有方面公平。同一 SAM2.1 Hiera-B+ 权重和冻结方式的确定性自动提示模型是最直接的预训练控制基线；比较不同预训练模型须列出预训练来源、额外数据、冻结范围和可训练参数，主张排除预训练影响需另设匹配对照。

## 2. 数据与测试集隔离

- 数据源：`DeepGlobe_Road/train/` 中有 mask 的原始影像（README 记录 6,226 对，建清单时复核配对、损坏和重复）；官方无标注 `valid/`、`test/` 只作推理/提交。
- 划分种子固定 `42`，所有模型和训练种子共用同一 manifest。先划分原图再裁块、增强、缓存和采样；同源、重复或重叠影像不得跨集合。按影像随机划分不保证地理独立；有地区/场景分组时优先组隔离，比例可近似。
- 三分工具：`scripts/prepare_splits.py` 现支持 `--test-fraction`（默认 `0` 保持旧两分）。上游环境已用 `--val-fraction 0.1 --test-fraction 0.1 --seed 42` 生成 `splits/protocol_v1/{train,val,test}.json`，记录为 **4,980 / 623 / 623**；本仓库不附带生成文件，正式运行前登记实际数量和 SHA-256。工具拒绝覆盖已有清单，不要复用旧目录。
- **旧清单用途与污染说明**：既有 5,603/623 两分样本只用于目标回提诊断（`runs/grpo_validation/check_targets_16.jsonl`），未用于模型训练或选模。按本地数据和同一算法复核，16 张诊断样本中 **100081、100841、101395** 落入新 test。因此不能笼统声称新 test“从未被查看”：登记表 `data.test_previously_used` 必须写明诊断用途、样本 ID 和核对依据。这没有模型前向、训练或按 test 选模，但仍是使用历史；若不能接受，应从未参与任何检查的数据重新建立 test。
- 权限：train 用于梯度、训练提示、GRPO 奖励和增强；val 用于预实验、有限调参、阈值校准和 checkpoint 选择；test 在协议冻结且全部选模完成后按预注册列表统一评估，不得据此挑种子、阈值、模型或增加采样次数。查看 test 后修改方法即新实验轮次，不能继续把同一 test 当作完全未见数据。
- 训练真值提示、热图标签、骨架监督和奖励只能来自 train；验证真值提示只用于独立诊断轨道，不进入自动推理主表。

## 3. 输入、提示与推理

- 几何：RGB、最长边缩放到 1024、右下 padding，评估恢复到原始标注尺寸；标签最近邻，评估排除 padding。引擎把 `image_size=1024`、`heatmap_size=256` 硬编码，改尺寸需改模型/几何代码，不能只加 YAML 键。本数据为 1024²，当前无实际 padding；引入其他尺寸前先补有效区损失和验证恢复。
- 单位：采点间距、`target_min_distance`、热图高斯、峰值平滑、`h`、NMS 的距离单位各不相同，按 `configs/3090.yaml` 字段语义使用；路径匹配半径用分割评估像素；提示坐标映射到 SAM2 的 1024 输入空间。不得混用原始像素和热图像素。SAM2 解码器同时需要图像嵌入和两级高分辨率特征，迁移时不得只传 64×64 嵌入。
- 归一化：各预训练骨干使用自身规定的输入归一化；SAM2 保持封装内 RGB `[0,1]` 接口。统一可见图像内容比强制相同归一化更重要。
- 自动推理主协议：每图一次预测，不使用真值点/框，不按 test 标注选轨迹。训练时间步、采样步数、候选数、`min_std`、`peaks`、分割阈值、TTA/集成开关均以 `configs/3090.yaml` 为准。峰值阈值与分割阈值是不同参数：峰值协议影响 A/C/D，须在 A 前冻结；分割阈值参与 D 奖励，须在 D 前冻结。无正峰时按当前模型规则输出空掩码，不补全局最大点。
- 随机性：每个样本 ID 从固定评估种子经稳定哈希派生独立 generator，不能用 Python `hash()`。最终独立评估显式使用 `--seed 2024`。不同模型、checkpoint、batch size 或读取顺序不得改变某样本的评估随机输入；不得从多次随机预测中挑最好一次进入主表。多步数、TTA、多候选或多次平均属于独立精度—耗时实验，须报告总成本，不进单次主表。

## 4. 训练预算、阶段语义与选模

### 4.1 A/B/C/D 当前语义

| 阶段 | 可更新部分 | 提示来源 | 初始化 |
|---|---|---|---|
| A | SAM2 mask decoder | train 真值热图提取的提示 | SAM2.1 预训练权重 |
| B | 条件编码器 + 去噪 U-Net | 监督热图、辅助目标 | A 的 `best.pt` |
| C | SAM2 mask decoder | 冻结扩散模型生成的提示 | B 的 `best.pt` |
| D | 去噪 U-Net | 同图 GRPO 采样轨迹 | C 的 `best.pt` |

图像/提示编码器始终冻结；D 的条件模块和 SAM2 冻结，参考 U-Net 固定。A 的 GT-prompt 验证只用于阶段内诊断，**A 不进入自动分割排名**；B/C/D 使用自动提示验证，两类指标不能混称。

每阶段使用独立输出目录，用该阶段预先规定的 `best.pt` 初始化下一阶段；不要混用末轮和 best。当前 `--stage all` 按阶段末轮串联，采用 best 协议必须分阶段运行。D 配方消融从同种子、同一 C `best.pt` 出发；共享 A/B/C 只在训练定义确实未变时复用。

A/B/C 学习率来自短程开发筛选，阶段长度与 warmup/cosine 调度尚未冻结；D 学习率只做了 32 组稳定性比较。这些值都不是最优或收敛结论。预实验后用开发种子 `2026` 一次性冻结实际长度，例如 D 可先看前 1–3 轮再决定。本文不规定固定 95 轮预算，也不要求花满候选长度；所有方法按冻结后的实际曝光和预算执行并报告实际值。不得在看到 test 后改阶段长度，也不得中途 `--resume` 改 epoch 预算后仍按原计划曲线报告。

D 只按当前 GRPO 实现描述，不把 DDPO 或其他策略形式混入本协议：`rl.algorithm=grpo`；`group_size` 在同图组内标准化优势；`ppo_clip` 是 GRPO 裁剪半径（键名为兼容旧配置保留）；`advantage_std_delta` 只是优势分母的标准差下限，不是跳过阈值；`point_cost_warmup_steps` 按**已处理 D 组数**推进，常量奖励组或 AMP 跳步也推进，验证始终使用未预热基准成本；`anchor_weight=0` 为默认，非零只是可选监督锚定。联合动作 log-prob 按维度求和，不用 mean 偷换目标。“旧策略”和“参考策略”不同：组 rollout 的 `old_log_prob` 只在组内冻结，作概率比分母；参考 U-Net 在首次进入 D 时从预训练权重深拷贝并永久冻结，只用于 KL。从已有 D 快照 `--init` 会保留 `ref_unet`，不会自动重置；`--init` 只加载权重，不继承优化器。

### 4.2 预算与 checkpoint 选择

- 两条互补轨道分别报告：(1) **数据曝光受控**：所有方法使用同一冻结的最大训练集遍历数（由 M0 预实验确定，不预设固定值），直接分割模型在同一上限内用 val 选 checkpoint；(2) **等计算**：用 3090 预实验确定统一 GPU 小时上限 H，冻结 H、验证/保存频率和计时边界。H 至少容纳完整预注册 A/B/C/D；若只够完成 C，须标为截断变体，不能称完整模型。
- 调参：每种方法最多 3 个候选 × 开发种子 `2026`，记录每个候选改了什么、看过哪些验证结果。主表统一用训练种子 `42/43/44`，划分不随训练种子变化；扩到 5 个种子时所有主表方法同等扩展，不能只补跑本方法或只保留最好种子。
- 选模：全部方法按相同曝光间隔做完整 val，候选间隔为每 5 次遍历一次并含末轮，具体值预实验后冻结；正式 YAML 必须显式写 `train.val_every`（未设置时引擎默认每轮验证）。按 val **macro IoU** 选择，平局取较早 checkpoint。A 的 GT-prompt macro IoU 只作阶段内诊断；B/C/D 用自动提示 val。固定预算主表不因表现差提前停止；数值失败可终止，但须登记原因、已用资源和重跑情况。
- 资源登记至少区分：原图展示次数、监督 optimizer update、D 已处理组数、非零优势组数、实际 optimizer step、轨迹数、轨迹转移数、GPU 小时。零优势不等于无更新：KL 非零时常量奖励组仍可能执行 optimizer step；KL 和 anchor 都为 0 时才可能整组跳过。AMP 跳步不算实际更新；A/C 无有效提示也可能跳过反传。计划更新数不等于实际值，必须记录实际值和调度进度。
- 计划量按冻结后的长度和实际 train 数量计算。例如 N_train=4,980、A/B/C batch=1、累积=4 时，每轮计划约 `ceil(4980/4)=1,245` 次更新；D 每轮约 N_train 个图像组，轨迹数和采样转移数按 `group_size × sampling_steps` 相乘。这些是算术下界，不含 AMP 跳步、重放、参考前向或 CPU 后处理，不能直接换算训练小时。旧 5,603/623 划分的数字不得混入新主表。

## 5. 消融与比较

D 配方消融从同一种子、同一 C checkpoint 出发，使用同一参考策略、组大小、采样、调度、验证样本和推理随机数方案，被消融字段除外。共享前缀只在没有改变前缀训练定义时复用。

| 编号 | 实验 | 改变内容 | 起点 |
|---|---|---|---|
| M0 | 完整 A+B+C+D | 基准 | A |
| A1 | A+B+C（无 GRPO） | 去掉 D | 复用 C |
| A2 | 等额外曝光监督延训 | 用与 D 相同曝光轮数的监督 U-Net 代替 D，冻结范围同 D | C；需新增入口 |
| A3 | 无点数成本 | `reward.point_cost=0` | C→D |
| A4 | 无 KL | `rl.kl_weight=0` | C→D；记录实际步数和调度差异 |
| A5 | 无 C 适配 | B 后直接 D，控制 D 预算 | B |
| A6 | 正点单通道 | 监督、生成、提取和训练提示一致修改 | 最早受影响阶段 |
| A7 | 确定性 CNN 热图 + 同款 SAM2.1 | 替换扩散生成器 | 对应生成器及适配阶段；需实现 |
| A8 | 普通局部极大替代 h-maxima | `peaks.h=0`，重新校准 | A 及所有依赖阶段 |

A2 不能用 `--stage B` 代替，因为 B 会更新条件模块，冻结范围和目标均不同。
点成本预热、G/采样步数扫描作为可选稳定性/成本实验，分别单因素变化，固定组数
和固定轨迹数两种口径分开登记。其余探索可先单种子，但必须标为探索性，不据此
作稳定性结论。共享 A/B/C 减少的是本轮总开销；单个方法仍须报告完整 A/B/C/D，
不能只给本方法报 D 的增量。

## 6. RTX 3090 预实验与当前限制

### 6.1 目标机与探测

当前目标机为 seetacloud2：RTX 3090 24 GiB、cgroup 14 vCPU、90 GiB RAM、`/dev/shm` 45 GiB。基础环境默认 torch 线程数可能高于 cgroup 配额（记录为 56 vs 14），`scripts/probe_3090.py` 在导入 torch 前显式设置线程数，默认 `--cpu-threads 2 --interop-threads 1`。任何吞吐数字必须同时记录实际线程、worker、功耗/时钟策略、温度和同卡其他进程。

预实验必须使用真实 SAM2.1 Hiera-B+ 权重（`sam2.1_hiera_base_plus.pt`，官方配置 `configs/sam2.1/sam2.1_hiera_b+.yaml`）、完整 1024 输入、256 热图和代表性稠密提示样本；缩小输入的 smoke 不能替代显存验收。A/B/C 分别预热并测足够 micro-batch，D 预热并测足够组，覆盖有奖励差异、零优势和较多峰的样本。记录每阶段 images/s 或 groups/s、单更新耗时、allocated/reserved 峰值、CPU RAM、验证/写盘耗时、AMP 跳步、非有限值和峰数分布。独占 GPU，用 CUDA 同步确定计时边界，分别记录纯训练区间和含数据/验证/保存的总墙钟。`scripts/benchmark_loader.py` 只测 CPU 预处理吞吐，不代表 GPU 训练吞吐。实测结果写入 `docs/probe_3090.md`；在真实结果出现前，不得把任何配置描述为已验证、稳定或最优。

**SAM2 迁移口径：** SAM1 与 SAM2 的 checkpoint 不兼容，必须用 SAM2.1 权重从 A 重新执行 A→B→C→D；SAM1 的快照或测量不能混入 SAM2 主表。SAM2 已开展有界探针，具体阶段、预算和结论范围以 [探针记录](probe_3090.md) 为准，不从历史结果推断新骨干表现。

### 6.2 当前实现限制（协议相关）

- 训练内验证的扩散评估种子仍取训练 seed（`evaluation_seed(self.seed, id)`）；最终独立评估可用 `--seed 2024`。正式训练前应把验证基础种子独立为固定值，并核查验证前后 RNG 状态和重复输出。
- CUDA autocast 固定 FP16，尚无 BF16 开关；未知 YAML 键可能被静默忽略，`amp_dtype: bf16` 不是可用配置。记录 TF32 和确定性实际状态；deterministic 路径的 warn-only 行为不保证逐位可复现。
- `image_size=1024`、`heatmap_size=256` 在引擎中硬编码；改尺寸必须改代码。训练验证固定 batch=1，实现逐样本 generator 后才可改批量验证。`train.val_every` 可读但 `configs/3090.yaml` 未显式设置，正式 YAML 应写入冻结值。
- `--stage all` 按阶段末轮串联，best 串联需分阶段运行；父 checkpoint SHA-256 需外部登记。`--resume` 不对 seed、batch、workers、划分、奖励、峰值和计划长度做完整硬校验，需外部对照配置与哈希。
- 其他待补：主流模型统一数据/评估适配、A2 监督延训入口、训练验证的原尺寸/插值/空图规则等价性、reserved 峰值与 CPU RSS 分项统计。

## 7. 评价、统计与资源报告

- 选模指标固定为 val macro IoU。主表同时报告 **macro IoU、micro IoU、F1、clDice**，F1 明确 macro/micro，不能互换。macro 为逐图得分等权平均；micro 先累加所有 TP/FP/FN 再计算。多种子统计在每个种子的全 test 指标上计算。
- 评估在恢复后的原始影像尺寸进行：连续概率先按统一插值规则恢复再阈值化，标签不做双线性插值。空预测且空标签得分为 1，仅一方为空为 0；报告无道路影像比例。clDice 使用二值骨架；训练 soft-clDice 不作为最终 clDice 数值。路径指标若使用双向骨架抽样代理，不得称正式 APLS；引入 APLS 时统一图构建、简化和匹配单位。
- 全量 test，禁止 `--limit` 或挑图。保存逐图指标、预测、样本 ID 和失败记录；遇错不得跳过失败图报子集均值。输出附命令、代码/依赖、checkpoint 和 split 哈希。官方无标注提交须核对预期 ID 和输出数量一致。
- 报告 3 个训练种子的均值±样本标准差并公开各次结果；机制消融按训练种子配对计算 ΔIoU/ΔclDice，共享父模型的相关性是设计的一部分。3 个种子不足以支持很强的显著性断言；如给置信区间，须说明重采样单位（逐图配对 bootstrap 不替代种子不确定性；有分组按组重采样）。
- 效率表必须覆盖：总参数/可训练参数及训练专属副本；全部阶段 GPU 小时、总墙钟、分阶段分项、AMP/精度、峰值 allocated/reserved、CPU RAM；调参候选次数、累计开发成本和失败/重跑成本；推理端到端 P50/P95、images/s、GPU 峰值及样本数/预热次数；正/负峰均值、中位数/P95、无正峰率、采样步数/候选数及随机数方案；离线缓存构建时间/大小及是否计入总成本。
- 推理计时边界统一为“已读取 RGB 输入 → 原尺寸概率/掩码可用”，包括预处理、GPU 传输、编码、条件模块、扩散、GPU→CPU、CPU h-maxima/NMS、SAM2 解码和尺寸恢复；CPU 后处理不得从 DiffPeak 成本中扣除。模型加载、图片读取和写盘可另列冷启动/完整文件流水线，所有方法保持相同口径。

## 8. 冻结与登记

所有必填项确认后，将本文和登记表标记为 `protocol_v1`（frozen）。冻结后任何改变记录为新版本，保留原因和受影响运行。复制 [`experiment_registration_template.yaml`](experiment_registration_template.yaml) 填写；它不是训练配置。

必填：数据版本、三份 manifest 路径与 SHA-256、分组/重复检查、旧 test 使用历史；代码 commit/工作区差异、依赖锁定、Python/PyTorch/torchvision/CUDA/驱动；划分种子 `42`、训练种子 `42/43/44`、开发种子 `2026`、评估种子 `2024` 及逐样本派生算法；模型来源、backbone、预训练来源与 SHA-256、冻结范围；各阶段冻结 epochs、实际更新规则、FP16、batch/累积/workers、确定性/TF32 实际状态、损失、调度和 D 参数；选模频率、随机输入、平局规则和阶段初始化选择；调参预算、曝光上限、等计算 H、计时边界；推理几何、采样、峰值参数、阈值、插值/空图规则；实测资源与余量；产物路径和失败记录。

分阶段调用示例（仅表示调用方式；正式 YAML 从 `configs/3090.yaml` 派生并补齐本文条件）：

```bash
python train.py --config configs/experiments/protocol_v1/full_seed42.yaml \
  --stage A --output-dir runs/protocol_v1/full/seed42/A
python train.py --config configs/experiments/protocol_v1/full_seed42.yaml \
  --stage B --init runs/protocol_v1/full/seed42/A/best.pt \
  --output-dir runs/protocol_v1/full/seed42/B
python train.py --config configs/experiments/protocol_v1/full_seed42.yaml \
  --stage C --init runs/protocol_v1/full/seed42/B/best.pt \
  --output-dir runs/protocol_v1/full/seed42/C
python train.py --config configs/experiments/protocol_v1/full_seed42.yaml \
  --stage D --init runs/protocol_v1/full/seed42/C/best.pt \
  --output-dir runs/protocol_v1/full/seed42/D
python evaluate.py --checkpoint runs/protocol_v1/full/seed42/D/best.pt \
  --split-file splits/protocol_v1/test.json --seed 2024 \
  --save-predictions --save-points \
  --output results/protocol_v1/full_seed42
```

推荐目录（生成文件不入库）：

```text
splits/protocol_v1/{train,val,test,protocol}.json
configs/experiments/protocol_v1/<method>_seed<seed>.yaml
runs/protocol_v1/<method>/seed<seed>/<stage>/
results/protocol_v1/{per_image,summary,efficiency}/
```

D 配方分支使用各自完整 YAML 和输出目录，共同初始化始终指向同一种子相同 C 的 `best.pt`。恢复训练使用 `--resume`，保持 seed、batch、workers、划分、奖励、峰值、采样和计划长度完全一致；当前代码不做完整硬校验，须外部对照配置与哈希。
