# 测试记录

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

最终 `passed=true`，exit code 0，脚本内约 3.5 秒。输入 128²、提示潜变量/目标热图均为 32²；使用默认 U-Net 通道 64/128/256/256、梯度检查点、CUDA float16 autocast 和 GradScaler。SAM 编码特征使用随机张量模拟，**没有加载真实 SAM 或计算真实分割奖励**。

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
