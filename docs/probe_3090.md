# RTX 3090 实测与 SAM2 迁移

## 状态

2026-10-21，已将 SAM ViT-B 替换为 **SAM 2.1 Hiera-B+**。SAM1 探针已停止并单列为历史基线；SAM2 已通过官方模块集成测试和真实 3090 A/B 全尺寸更新检查，C/D 与效果筛选继续验收。短程测量只支持其测试范围，不是全局最优超参数或收敛证明。

可执行参数集中于 `configs/3090.yaml`，比较规则见 [实验协议](experimental_protocol.md)。短程筛选不替代完整验证、多种子比较或收敛实验；未实测项保持未确认，不用猜测补齐。

## Vast.ai 四阶段吞吐补测（2026-10-21）

实例57042404：RTX3090 24GB/350W，80GB盘，约$0.243/小时；PyTorch2.8/cu128镜像，使用pytorch conda环境。只传256对train和32对val，SAM2权重现场下载并校验。原扫描88次、有效batch16扫描24次、batch32容量检查6次、D扫描8次，共126次候选运行；其中B batch32两次OOM，其余124次通过有效性检查。证据已回收，实例销毁命令返回0，销毁后实例列表为空。

本地证据：`runs/vastai_57042404/evidence/probe-results/`；原始传回压缩包、远端日志和销毁凭证在其上级目录。以下均workers4、关闭检查点、两次吞吐中位数，单位张/秒（D为影像轨迹组/秒）。

| 阶段 | batch2 | batch4 | batch8 | batch16 | batch32 |
|---|---:|---:|---:|---:|---:|
| A | 13.09 | 13.03 | 9.72 | 6.94 | 5.36 |
| B | 12.19 | 12.78 | 9.95 | 6.92 | OOM×2 |
| C | 1.805 | 1.912 | 1.975 | 1.919 | 1.724 |

batch2/4/8/16固定有效batch16，32单独有效batch32；不可当作相同优化配置比较。B batch16 allocated22.16GiB，32无法装下；A batch32 allocated15.94GiB、C batch32约20.54GiB，能装下但没有吞吐优势。D固定G4/20步、预热8组测量16组：batch1/2/4/8分别0.1562/0.1807/0.2015/0.2133组/秒，batch8 allocated9.31GiB，相比batch1吞吐提高约36.6%。

原有效batch4扫描：A最佳约13.64张/秒（batch2、workers4，开/关检查点近似）；B最佳12.62（batch2、workers4、关检查点）。C和D的大批次收益比A/B明确；初步复测候选为A batch2、B batch2或4、C batch8、D batch8，均workers4/关检查点。不要直接覆盖正式配置：批次会改变随机流和优化行为，C/D父权重只是短程bootstrap，不代表收敛提示分布；D尚未覆盖100组点成本预热；每候选短测只有64张展示或16轨迹组，DataLoader启动/预取和小数据集热缓存会影响结果。需要在正式父权重、长测量和完整验证下确认。

## 环境与证据位置

- 连接：`ssh seetacloud2`；授权工作目录 `/root/autodl-tmp/diffpeak-probe-20261021`。
- RTX 3090：24,576 MiB；驱动 580.76.05；Python 3.12.3，PyTorch 2.8.0+cu128，torchvision 0.23.0+cu128。
- cgroup：14 vCPU、90 GiB 内存；`/dev/shm` 45 GiB。宿主 CPU affinity 暴露 112 核，PyTorch 原默认 56 线程，不能照此使用。
- 隔离环境：工作目录内 `.venv`，复用原 PyTorch，NumPy 1.26.4、SciPy 1.13.1、scikit-image 0.24.0、Pillow 10.4.0。
- SAM2 源码：官方 `facebookresearch/sam2`，固定提交 `2b90b9f5ceec907a1c18123530e92e794ad901a4`，关闭可选 CUDA 扩展。
- SAM2 权重：`checkpoints/sam2_download/sam2.1_hiera_base_plus.pt`，323,606,802 字节，SHA-256 `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`。
- 迁移前原始证据：服务器 `loader_benchmark*.jsonl`、`suite_capacity.jsonl`、`suite_learning.jsonl`、`results/*/{results,validation,environment,resolved_config,splits}.json` 与 `raw.jsonl`；本地副本 `runs/probe_3090_20261021/`。生成证据不入 Git。
- SAM2 使用独立的 `results_sam2/`、`suite_sam2_*.jsonl`，禁止混入 SAM1 权重或覆盖历史产物。

数据：有标注 6,226 对，官方 valid/test 无掩码。新三分 manifest 为 4,980/623/623，划分种子 42；本次训练只读 train，筛选只读 val。旧目标诊断与新 test 的重叠使用历史见实验协议，不能声称所有 test 标注从未查看。

## CPU 与内存：迁移前预处理实测

固定 train 子集，32 张预热、256 张测量，数值库与 torch intra/inter-op 各 1 线程，只测数据加载与预处理：

| workers | 张/秒 |
|---:|---:|
| 4 | 15.79 |
| 8 | 28.25 |
| 12 | 42.70 |

数据包括 JPEG/PNG 读取、原分辨率骨架、随机采点与热图。8 张预热/64 张测量的初轮排序相同，workers=0 约 4.05 张/秒。这不是 GPU 训练吞吐：当加载已快于 GPU 消费，更多 worker 不一定提升训练速度。

主进程 probe 设置 `--cpu-threads 2 --interop-threads 1`，环境 `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`；正常 `train.py` 也支持这两个线程选项。充分利用内存以减少数据等待、允许 worker 预取，不把内存占满当作性能目标。不假设翻转 SAM embedding 等价于重新编码，未引入改变增强语义的离线缓存。

## SAM2.1 全尺寸实测

真实官方 B+ 权重、输入 1024²、热图 256²、FP16、种子 2026。A 初始探针 3 次更新预热 + 13 次测量更新，全部实际执行，无 AMP 跳步、可训练参数有限；测量 52 张约 **6.43 张/秒**，allocated **1,073.04 MiB**、reserved **1,092 MiB**。16 张 GT-prompt 验证 IoU 0.3187，只作为接口/学习诊断。

B 系统探针共享这个 A 父权重，预热 3 次更新、测量 16 次更新，每候选测量 64 次影像展示；有效 batch=4。各候选更新均成功、参数有限：

| workers | 激活检查点 | micro-batch×累积 | 张/秒 | allocated MiB | reserved MiB |
|---:|:---:|:---:|---:|---:|---:|
| 4 | 开 | 1×4 | 8.93 | 1,548.27 | 1,666 |
| 12 | 开 | 1×4 | 8.35 | 1,548.27 | 1,666 |
| 8 | 关 | 1×4 | 9.74 | 2,099.81 | 2,148 |
| 12 | 关 | 1×4 | 9.39 | 2,099.81 | 2,148 |
| 8 | 关 | 2×2 | 11.39 | 3,473.45 | 3,622 |
| 8 | 关 | 4×1 | 9.93 | 6,172.52 | 6,928 |

本轮 B 最快候选为 workers=8、关闭检查点、2×2，比同 workers 的 1×4 快约 17%。LR 比较仍固定 1×4，避免同轮同时改系统与优化变量；2×2 上的效果需要单独复核。12 workers 在 CPU 独立测量更快，但本轮 GPU 训练没有更快，因此不按占用核数选参。

SAM2 A 学习率筛选固定 32 次更新、32 张 GT-prompt val、恒定 LR，同种子与样本：`3e-5 / 1e-4 / 3e-4` 的 macro IoU 分别 **0.3637 / 0.3180 / 0.4267**。本轮选 `3e-4` 作为 B 候选共享的开发父权重；与 SAM1 的筛选排序不同，进一步说明迁移后不能沿用旧“最优值”。此选择只适用于当前小预算，尚未验证完整调度或收敛。

SAM2 B 候选使用相同 A(`3e-4`)父权重，每候选 128 次更新/512 次展示、固定 32 张自动提示 val：

| B LR | macro IoU | 平均提示点 | 测量期平均扩散 MSE |
|---:|---:|---:|---:|
| 3e-5 | 0.36253 | 3,432.91 | 0.8005 |
| 1e-4 | 0.36285 | 2,882.19 | 0.4309 |
| 3e-4 | 0.36265 | 3,416.59 | 0.1903 |

自动 IoU 差异小于 0.0004，不能据此声称有效优势；保留 `1e-4` 作为 C 开发父权重。较低扩散 MSE 不一定对应更高自动分割质量，所有候选仍有大量提示，需后续收敛检查。

C 候选均完成 16 次更新/64 次展示，固定 32 张自动 val：`2e-5 / 5e-5 / 1e-4` 的 IoU 分别 **0.3653 / 0.3784 / 0.3930**；本轮选 `1e-4`，各候选无 AMP 跳步、参数有限。

D 系统检查使用这个 C 父权重，G=4、20 步，1 组预热后测量 4 组：关闭检查点 **29.44 秒**，开启 **33.13 秒**，均为 4 次实际更新、无 AMP 跳步。关闭时 allocated **1,846.62 MiB**、reserved **1,970 MiB**，全部组均有非零奖励方差、点集多样性为 1；没有通过减少 G/步数换取速度。首次重放联合 log-ratio 误差为 0 是更新前同策略重评分的预期，不证明多组后的策略漂移为 0。长期 KL 需继续监控。

D 已完成 `3e-7` 与 `1e-6` 两个候选的 32 组预算：均为 32 次实际更新、无 AMP 跳步、参数有限；固定 val IoU 分别 **0.39290 / 0.39284**，差异远小于筛选噪声，不能据此宣称效果增益。测量期平均 KL 分别约 **47.79 / 197.0**，`1e-6` 的偏离明显更大。`3e-6` 候选运行结果未纳入本记录。因此 D 学习率**尚未冻结**：暂保留配置中的 `1e-6` 作为原基线，若以数值稳定性优先，`3e-7` 是更保守的待确认候选。32 组预算未覆盖 100 组点成本预热，确认脚本 `scripts/confirm_probe.py` 已提供但结果未产生；不得把本段写成 D 收敛或增益结论。

## SAM1 历史基线：不作为 SAM2 结论

真实 SAM ViT-B、1024²/256²、FP16、开发种子 2026。B 候选共享 A 初始权重、256 张训练子集，预热 3 次更新后测量 16 次更新，每次有效批次 4；主进程 2/1 线程。所有候选实际更新 16 次，未 AMP 跳步，参数有限。

| B workers | 激活检查点 | micro-batch×累积 | 张/秒 | allocated MiB | reserved MiB |
|---:|:---:|:---:|---:|---:|---:|
| 4 | 开 | 1×4 | 7.05 | 1,833.75 | 3,472 |
| 12 | 开 | 1×4 | 6.72 | 1,832.25 | 3,472 |
| 8 | 关 | 1×4 | 7.32 | 1,963.22 | 3,482 |
| 12 | 关 | 1×4 | 7.46 | 1,963.22 | 3,482 |
| 8 | 关 | 2×2 | 8.72 | 3,296.48 | 6,456 |
| 8 | 关 | 4×1 | 7.97 | 5,921.81 | 12,358 |

这批短测量不是长尾 OOM 保证，batch/worker 会改变当前训练随机流。不能只按有效 batch 相等声称训练完全等价。初版报告将未使用的损失项 NaN 误纳入 `all_metrics_finite`，该聚合标志后来修复；原始各有效损失、实际更新和参数检查均有限，原始记录不覆盖。

A 初始探针：16 次更新，52 张测量约 4.68 张/秒，allocated 1,715.54 MiB；16 张 GT-prompt 验证 IoU 0.4902，仅作为诊断。

### 已完成的 SAM1 学习率筛选

同种子、同样本、恒定 LR、无调度混杂；A 每候选 32 次更新/128 次影像展示，固定 32 张验证子集：

| A LR | GT-prompt macro IoU |
|---:|---:|
| 3e-5 | 0.4951 |
| 1e-4 | 0.4815 |
| 3e-4 | 0.4500 |

选择 `3e-5` 仅用于 SAM1 后续候选的共同开发父权重。B `3e-5` 的自动验证 IoU 为 0.1583、平均提示点 3,593.72/图，暴露密集峰 CPU 后处理瓶颈。用户要求切换 SAM2 后中止余下 SAM1 筛选；**没有得到 SAM1 C/D 最优参数，更不能继承为 SAM2 最优参数**。

## 可迁移的 A/B/C/D 吞吐探针（租机前准备）

使用 `pytorch` conda 环境。子进程沿用 `sys.executable`，不会切回默认 Python；实际 torch/CUDA 版本由单候选环境记录保存。本地当前环境版本可能高于项目目标 2.8/cu128，不将本地数值当作目标 3090 结果。

只迁移探针子集，无需上传完整3.9GB数据。以下命令固定种子2026，从现有 train/val 划分分别复制256/32对图像掩码到新目录，保存逐文件SHA256和源清单哈希，不读取test：

```bash
conda run -n pytorch python scripts/prepare_probe_data.py \
  --output-dir runs/vastai_probe_data_20261021
```

本地已生成 `runs/vastai_probe_data_20261021/`，图像掩码合计140,632,424字节（约134MiB），共576个数据文件；上传该目录即可。远端运行或本地预演时指定子集路径：

```bash
conda run -n pytorch python scripts/probe_throughput.py \
  --data-root runs/vastai_probe_data_20261021/data \
  --train-split runs/vastai_probe_data_20261021/splits/train.json \
  --val-split runs/vastai_probe_data_20261021/splits/val.json \
  --output-dir runs/vastai_3090_subset_throughput --bootstrap --dry-run
```

子集用于容量/吞吐筛选，不代表完整数据集收敛或质量结论；大batch长程探针可能重复展示这256张训练图，数据缓存也会比完整数据更热。

若使用原始数据路径，先做无训练、无输出目录写入的预演：

```bash
conda run -n pytorch python scripts/probe_throughput.py \
  --output-dir runs/vastai_3090_throughput --bootstrap --dry-run
```

`--config --data-root --sam-checkpoint --train-split --val-split --output-dir` 均可指定；命令自动使用绝对路径，默认输入来自当前项目配置。预演检查完整 manifest 的图像/掩码存在、路径边界和 train/val 不重叠；缺少 SAM2 或显式父权重会列出警告，增加 `--check-inputs` 则将其视为错误。真实执行必须具备全部输入。输出目录已存在时拒绝执行。

`--bootstrap` 用各阶段第一个健康基线候选保存的权重作为下一阶段所有候选的共同父权重，用于新机器的容量探测；父候选失败或有跳步时，不继续用它的权重。**这不是完整训练或效果筛选**，C/D 的提示密度和耗时仍需在正式已训练父权重上复测。有现成权重时优先显式提供 `--init-b A.pt --init-c B.pt --init-d C.pt`，各阶段可通过 `--stages A B C D` 单独执行。

```bash
# 输入完整后去掉 --dry-run 即执行；默认最长运行 3 小时。
conda run --no-capture-output -n pytorch python scripts/probe_throughput.py \
  --sam-checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
  --output-dir runs/vastai_3090_throughput --bootstrap \
  --workers 4 8 --batches 1 2 4 --effective-batch 4 \
  --checkpointing off on --repeats 2 --max-seconds 10800
```

- A/B/C 分别扫描 micro-batch 1/2/4、累积 4/2/1，固定有效 batch=4；workers 4/8、检查点关/开，每候选重复两次。batch/worker 改动会改变随机流，相同有效 batch 仍需独立确认质量。
- D 保持 YAML 中 G=4、20步、奖励与更新次数不变，不引入梯度累积。默认比较 batch=1/2，可用 `--d-batches 1` 缩短初筛；预热/测量组数必须整除 batch，避免组预算取整。默认预热2组、测量4组只支持短程容量判断；完整点成本预热用 `--d-warmup-groups 100 --d-measure-groups 28`，并先评估总时限是否足够。
- 扩大有效 batch 是单独实验，例如 `--effective-batch 8 --batches 1 2 4 8`，不要与固定有效 batch=4 的结果混排。若总时限内未完成所有候选，`completion.json` 标记不完整，退出码非零。
- `--candidate-timeout` 默认1800秒；超时终止当前子进程并保留日志，然后在剩余总时限内继续。套件通过可见设备集合的协作锁避免同套件重叠运行；运行前仍需确认设备上无其他训练任务。
- 每阶段按全部重复均健康的候选吞吐中位数排序，保留最小/最大值；无实际优化更新、AMP跳步、非有限指标/参数、train/val子集哈希不一致均不参与推荐。吞吐排名不按利用率百分比排序，也不使用验证 IoU 选父权重。
- 单候选预热与测量分别记录 `gpu_telemetry_warmup.jsonl` / `gpu_telemetry_measurement.jsonl`，每0.25秒采样选定GPU的利用率、显存占用、功耗和温度；汇总位于各阶段的 `gpu_telemetry.metrics`。不混入初始化或验证。无法可靠映射GPU时明确标记不可用，不猜测多卡序号；采样异常不打断训练。驱动显存占用与torch allocated/reserved峰值含义不同。
- 输出 `plan.json`、逐候选日志和原始 probe 目录、`suite.jsonl`、`summary.json`、`completion.json`，及独立的 `recommended_A.yaml` 至 `recommended_D.yaml`。推荐配置只覆盖吞吐设置，原始 `configs/3090.yaml` 不变；未测量或失败阶段不导出配置。

本地测试：

```bash
conda run -n pytorch python -m unittest discover -s tests -p 'test_probe*.py'
```

## 探针入口与验收

- `scripts/benchmark_loader.py`：独立预处理吞吐。
- `scripts/probe_3090.py`：单候选，真实 engine、全尺寸、独占输出；记录实际更新/跳步、时间、显存、线程、哈希、固定子集和失败 traceback；`--save-weights` 仅供开发 `--init`，不是可恢复训练快照。
- `scripts/run_probe_suite.py`：顺序运行系统配置或 A/B 等预算 LR 候选。
- `scripts/probe_followup.py`：开发 B→C→D；未训练或弱父模型只能支持容量/稳定性判断。
- `scripts/confirm_probe.py`：从选定 C 父权重开始，先处理 100 组点成本预热，再测量 28 组稳态；用于补齐 32 组候选未覆盖的预热。运行结果到达前不填写数值。
- 本地 `runs/probe_3090_20261021/` 只保存汇总 JSONL；逐候选 `results*/`、`raw.jsonl`、日志与权重留在服务器授权目录，属于生成证据、不入 Git。

A/B 全尺寸与 C/D 系统检查已完成；正式冻结前仍需足够组数的 D 预热确认、完整 val 对比、长尾提示/内存检查与阶段长度选择。任何停止、失败、额外验证和实现修复均保留日志；不把少量更新的赢家写成全局最优。
