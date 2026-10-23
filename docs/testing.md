# 测试记录

## 单通道山脊场重构（2026-10-23）

当前协议 `signed_ridge_v1`：单通道有符号场，SAM 全阶段冻结，ABC 完整监督与 D 扩散轨迹 GRPO。下方双通道、峰值和 A/B/C 记录保留为历史证据，不代表当前接口或性能。

- 本地 `pytorch` 环境（Python 3.11.15、PyTorch 2.14.0+cu130）执行 `python -m pytest tests -q`：**395 passed、192 subtests passed，13.28 秒**。覆盖源场/缩放/padding、细道路与特殊几何、四区域归一、全无效区域、单通道采样、裁剪前高斯动作、冻结 SAM 输入梯度、ABC/D 参数和状态不变、同图奖励分组、参考固定、概率重放、常量奖励、显式迁移及严格新协议恢复。
- 官方 SAM2.1 模块随机权重集成测试实际运行，验证单通道提示编码、固定尺度、半像素 padding 和分割梯度；没有使用 SAM2.1 预训练权重，不能说明真实提示有效性。
- CUDA `python scripts/gpu_smoke.py`：**passed=true**，RTX 3050 Laptop 4GB；单通道、AMP、可微采样与扩散 GRPO 更新通过，初始重放概率比最大误差 **0.0**，参数及梯度有限。微型 DiT + 模拟 SAM 特征的 ABC/D 峰值 allocated 为 **73.84 / 58.91 MiB**；该工具不运行真实 SAM 完整分割，不代表完整模型显存。
- `check_targets.py --config configs/3090.yaml --samples 16` 在真实训练清单前16张运行成功：场范围 [-1,1]，未分区有效像素总数0。首样本 100034 的保守投影相对中心最近邻保留 **1.49 倍道路单元、4.58 倍骨架单元**；投影道路/骨架召回均为1，后者是几何一致性检查，不能作为模型性能。没有读取 test 清单。
- `scripts/diagnose_ridge_sam.py --mock --size 64` 运行成功，覆盖山脊/二值提示、多尺度、零值/打乱/局部扰动。mock 仅验证诊断工具，不能替代冻结预训练 SAM 的真实接口实验。
- 本地忽略目录 `runs/signed_ridge_validation/` 保存 `targets_16.jsonl`、`gpu_smoke.json`、`oracle_mock.json` 和回归日志。`python -m compileall -q diffpeak tests scripts train.py predict.py evaluate.py check_targets.py` 与 `git diff --check` 通过。
- **尚未执行：**真实预训练 SAM2.1 的尺度选择与提示有效性验证、真实小样本收敛、3090 全尺寸吞吐重测、完整训练和确定性基线/GRPO/无锚定对照。当前本地只有 SAM1 权重；代码验收不构成分割质量或 GRPO 收益结论。

## DiT 架构替换验证（2026-10-22）

- 默认去噪器替换为 `ConditionalDiT`，保留四尺度条件、双通道噪声预测、DDIM 与 GRPO 接口；既有采样背景先验回归仍通过。
- 使用本地 `pytorch` 环境执行 `python -m pytest tests -q`：**300 passed、157 subtests passed**。微型 DiT 覆盖完整 A–D 控制流及精确断点恢复；新增测试覆盖动态尺寸、四尺度条件/时间步影响、反向梯度、eval 模式激活检查点、模型构建和旧 U-Net 快照拒绝。
- `python scripts/gpu_smoke.py`：**passed=true**，RTX 3050 Laptop 4GB、PyTorch 2.14.0+cu130；使用小型 DiT、模拟 SAM2 特征，验证 AMP、B 阶段更新和 D 阶段采样/概率重算/策略梯度。
- 默认 DiT（宽度 256、6 层、8 头、patch=8）在完整 256×256 热图与四尺度条件上通过 CUDA 前向及冻结输入的 eval 模式反向检查。此检查仅覆盖去噪器，不代表完整 SAM2 训练显存或吞吐。
- 存在旧条件融合随机测试抖动：`tests/test_conditioning.py:114` 的随机初始化 + `torch.allclose` 断言在主流程的一次条件/SAM 定向回归中失败（39 passed / 1 failed），随后单独重跑通过；最终全量回归为上述 300 passed。未修改该旧测试或条件编码器实现，不能将通过结果理解为测试套件已无随机失败。
- 补充检查点注意力头数校验：`num_heads` 改变不会改变参数形状，因此除严格权重加载外，还检查快照配置中的头数与当前 DiT 一致。
- 尚未进行 DiT 的 3090 完整训练、收敛或精度评估。下方 U-Net 时期的硬件与训练记录是历史结果，不能作为当前 DiT 的验证依据。

## DiT 旧接口清理验证（2026-10-22）

- 删除旧 `diffpeak/models/unet.py` 及其导出；时间步正弦编码移至 `dit.py`，条件编码器测试独立为 `tests/test_conditioning.py`。
- 模型工厂、engine、推理、扩散调用和训练快照统一使用 `dit` / `ref_dit`，不保留旧键或属性别名。旧键快照因缺少必需 `dit` 字段直接失败，参数仍严格加载。
- 本地 `pytorch` 环境全量回归：`python -m pytest tests -q`，**297 passed、157 subtests passed，8.95 秒**。测试数量变化来自删除 UNet 专属测试、保留时间步编码测试。
- `python -m compileall -q diffpeak scripts train.py predict.py evaluate.py` 与 `git diff --check` 通过。
- 本轮尚未启动 Vast.ai 实例，未取得真实数据 B 阶段的收敛结果；CPU 测试通过不构成收敛证据。

## SAM2.1 与探针工具验证（2026-10-21）

- 在本地 `pytorch` 环境安装官方 SAM2 固定提交 `2b90b9f5ceec907a1c18123530e92e794ad901a4`，`pip check` 通过。最初 build isolation 尝试额外安装 CUDA/PyTorch 导致临时目录空间不足；改用 `SAM2_BUILD_CUDA=0 ... pip install --no-build-isolation`，保留原 PyTorch 后成功。
- 完整回归：`conda run -n pytorch python -m pytest tests -q`，**293 passed、169 subtests passed，10.16 秒**。包含真实官方随机权重 SAM2 模块的解码/梯度集成、SAM1 快照拒绝、显式高分辨率特征传递、有界探针和线程 CLI。
- 数据三分工具新增独立 test，保持旧默认两分兼容；固定种子、互斥清单与拒绝覆盖测试通过。
- 峰区域查找与网格 NMS 优化已对旧实现进行随机/边界/密集峰等价检查；没有改阈值、峰数上限或提示语义。
- 3090 真实 SAM2 A/B 已完成有界训练和实际更新；资源结果、参数筛选及尚未完成项只在 [3090 探针记录](probe_3090.md) 维护，不能从 CPU 测试推断完整训练收敛。

以下章节均为 **SAM2 迁移前的历史记录**；其中“未下载权重”“未测 3090”等描述只适用于对应历史时点。

## 实验日志与 TensorBoard 增量验证（2026-10-21）

在已有 `pytorch` conda 环境安装 `tensorboard==2.21.0`，`pip check` 无依赖冲突。

- `conda run -n pytorch python -m pytest -q`：**221 passed、91 subtests passed，9.40 秒**。
- `conda run -n pytorch python -m unittest discover -s tests -q`：**221 tests，OK，6.858 秒**。
- 新增日志文件/JSONL/TensorBoard 事件回读、非有限值处理、stdout/stderr、异常/中断状态、初始化失败清理、RNG 隔离、CLI 参数及微型 A–D/resume 会话隔离覆盖。
- 实际启动本地 TensorBoard HTTP 服务：主页返回 200，scalars tags 接口可读取 `B/train_epoch/loss` 和 `D/train_group/reward`；使用明确标记为 synthetic smoke 的诊断值，**不是模型实验结果**。检查后已关闭服务。
- `conda run -n pytorch python scripts/gpu_smoke.py`：`passed=true`，exit code 0；仍为模拟 SAM 特征的 CUDA/AMP/GRPO 小型检查，不代表真实 SAM 完整训练。
- `train.py --help`、`git diff --check` 通过。未下载预训练 SAM 权重，未启动真实 A–D 完整训练，未验证 3090 正式配置。

以下保留 GRPO 迁移时的历史验证记录，其 203 项数量对应日志功能加入前。

## 当前环境（2026-10-21）

按授权使用已有 `pytorch` conda 环境，并从显式 PyPI 索引安装缺失依赖。未下载 SAM 预训练权重。

| 项目 | 实际环境 |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.14.0+cu130 |
| CUDA | 13.0（可用） |
| NumPy | 2.4.6 |
| Pillow | 12.3.0 |
| scipy | 1.17.1（本次安装） |
| scikit-image | 0.26.0（本次安装） |
| PyYAML | 6.0.3（本次安装） |
| segment-anything | 1.0（本次安装） |
| tqdm | 4.70.1（本次安装） |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU，4095.5 MiB |

安装命令：

```bash
conda run -n pytorch python -m pip install --no-cache-dir \
  --index-url https://pypi.org/simple \
  "PyYAML>=6.0" "scipy>=1.13" "scikit-image>=0.24" \
  "segment-anything==1.0" "tqdm>=4.66"
```

该环境不是 `requirements.txt` 推荐的 Python 3.10 / torch 2.5.1 cu121 组合，因此结果不能视作固定依赖组合的认证。`segment-anything` 只用于随机权重架构测试，没有加载 `sam_vit_b_01ec64.pth`。

## 完整 CPU 测试

```bash
conda run -n pytorch python -m pytest -q
conda run -n pytorch python -m unittest discover -s tests
```

最终结果：**pytest 203 passed、84 subtests passed（7.39 秒）；unittest 203 tests，OK**，均 exit code 0。包含微型 A–D 控制流测试，耗时随机器负载波动。最终日志为 `runs/grpo_validation/pytest_verified.log` 和 `unittest_verified.log`。另运行：

```bash
conda run -n pytorch python -m compileall -q diffpeak tests scripts \
  train.py predict.py evaluate.py check_targets.py
```

源码语法编译通过。

覆盖范围：

- GRPO：组内相对优势（多图列独立标准化、常量组零优势、低方差组按 `delta` 缩放）、裁剪代理损失、固定方差高斯 KL；
- 调度：线性 warmup + cosine 退火、`LambdaLR` 时序、单步/短调度与越界 clamp；
- 引擎：A–D 冻结范围、训练/评估模式、batch>1 的 `G*T*B` 归一化、无效组跳过 backward/optimizer、点成本预热、点集 diversity、日志字段；
- 端到端控制流：微型 A–D 全阶段训练、阶段 checkpoint、D 阶段中断后 `--resume` 与不中断训练的模型参数一致性（patch `build_models`，不加载真实 SAM 权重）；
- checkpoint：GRPO 算法/调度元数据写入，旧 DDPO D 阶段快照禁止直接 `--resume`；
- 扩散：非零末步方差、联合动作 log-prob 求和、采样记录与重放；
- 奖励：IoU/clDice/路径代理、空图规则、提示成本；
- 峰值：h-maxima + NMS、平顶确定性、正负重合消歧、回提诊断；
- 数据/几何/划分/推理边界；
- SAM：真实官方 `segment_anything` 模块的随机权重集成测试（架构、单掩码接口、提示配对、空提示常量输出、梯度路径），以及无权重张量逻辑测试。

迁移过程中出现并已修复的问题如实记录：

1. 首次运行有 5 个测试模块因缺少 scipy / scikit-image / PyYAML 导入失败；安装依赖后消除。
2. `Predictor.predict_file` 直接使用 `Path.stem`，`scene_sat.jpg` 得到 `scene_sat`，与 `canonical_image_id` 的 `scene` 不一致；已改为统一种子 id。
3. 混合 batch 中“无正点样本不构造解码器梯度路径”的测试最初断言单行 `requires_grad=False`，这与 `torch.cat` 的自动微分语义不符；已改为验证空行反传后解码器梯度为零。
4. 点成本预热的日志/检查点字段与轻量测试对象缺少初始化属性；已增加安全默认值与回退。

5. checkpoint 虽恢复了调度元数据，`LambdaLR` 闭包原先仍捕获新配置参数；改为读取恢复后的元数据，并验证显式覆盖 warmup/floor 后仍按 checkpoint 原曲线继续。

6. 独立复核确认多轮 GRPO 的 `skipped_groups` 原先误用了跳过更新次数；现分别记录 `skipped_update_steps` 与整组未更新的 `skipped_groups`，并增加跨轮统计回归测试。

这些修复后全量测试为上述 203 项通过，没有保留已知测试失败。合成孤立高斯目标额外验证了 8 个固定噪声种子下，末步 `min_std=0.02` 未破坏目标峰；这不等同于已验证真实生成热图的峰稳定性。`pip check`、训练/推理/评估 CLI 的 `--help` 和 `git diff --check` 也通过。

## RTX 3050 小规模 GPU 测试

```bash
conda run -n pytorch python scripts/gpu_smoke.py
```

**历史记录（U-Net 时期）。** 下表数值来自旧 `ConditionalUNet` 脚本：输入 128²、提示潜变量/目标热图均为 32²；U-Net 通道 64/128/256/256、梯度检查点、CUDA float16 autocast 和 GradScaler。SAM 编码特征使用随机张量模拟，**没有加载真实 SAM 或计算真实分割奖励**。当前 `scripts/gpu_smoke.py` 已改用小型 `ConditionalDiT`（patch 4、hidden 128、depth 2、4 heads，`condition_channels=(64,128,256,256)`）并默认开启梯度检查点；2026-10-22 已重新运行并通过（见文首验证记录）。下表中的参数计数、显存与耗时仍是 U-Net 历史值，不能作为 DiT 验收结果。

| 检查 | 结果 |
|---|---|
| B 条件 + 扩散 + 辅助损失前向 | 全部有限，total≈1.803092 |
| B 反向及参数更新 | 无缺失/非有限梯度，参数确实更新 |
| GRPO 工具（多图组、常量组、裁剪、KL） | 多图优势 ∓1；常量组优势/梯度为 0；KL 梯度非零 |
| D CPU generator 固定种子重采样 | 最大差 0 |
| D 保存轨迹与重算 log-prob | 初始 ratio 最大误差 0 |
| D GRPO 优势 | 组内奖励 `[0.4, 0.6]` 得到 `[-1, 1]` |
| D GRPO + KL + anchor 反向 | 无缺失/非有限梯度，更新后参数有限 |
| D 参数更新元素数 | 12,604,613 |
| B / D 峰值 allocated | 339.51 / 369.08 MiB |
| 峰值 reserved | 386 MiB |
| 占 4 GiB 设备比例 | 9.01% |

G=2、扩散训练步数 32、采样 4 步，仅为快速验证算子、梯度、采样/重算一致性与 GRPO 工具，不是 1024² 正式性能测试，**不能把这些显存数字外推为 3090 的完整训练占用**。smoke 额外施加了合成扩散锚定以打通该梯度路径；正式配置默认 `rl.anchor_weight=0.0`。

D 阶段使用保守的 AMP 初始 scale=64；GPU smoke 的梯度增长间隔较长，不构成长程训练数值稳定性的保证。它验证的是模块组合与 GRPO 更新路径，不是完整 `engine.py` A–D 控制流。

## 真实数据检查

```bash
conda run -n pytorch python scripts/prepare_splits.py \
  --root DeepGlobe_Road --output runs/grpo_validation/splits --seed 42 --val-fraction 0.1
conda run -n pytorch python check_targets.py \
  --config runs/grpo_validation/check_targets_config.yaml --samples 16
```

内部划分为 train=5603、val=623，另有无标注影像 2344 张。16 张真实影像使用 1024 输入 / 256 热图完成数据加载和目标回提：**14/16 张完全回提；1799/1801 点匹配，无额外点**。`100841`、`10255` 各漏 1 点，F1 分别为 0.997963、0.997167。检查脚本退出码为 0 不代表全部目标峰可恢复；这些漏峰是默认参数仍需校准的实验诊断，不隐去。原始报告见 `runs/grpo_validation/check_targets_16.jsonl`，环境和结果汇总见 `runs/grpo_validation/environment.json`。

## 未执行范围

- 未下载 SAM ViT-B 预训练权重，未执行预训练 SAM 图像编码/完整 `Predictor` 或真实数据上的完整 A–D 训练/验证/CLI 全链条。随机权重 SAM 提示编码/解码已验证。
- 未在完整 DeepGlobe 训练集上运行数据加载、真实分割奖励与耗时统计。
- 未在 RTX 3090 24GB 上验证 1024² 训练峰值显存、吞吐、收敛或精度。
- 未执行多随机种子实验、GRPO 与 DDPO 的配对消融，也不声称 GRPO 一定改善指标。
