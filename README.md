# DiffPeak-SAM

依据 [`docs/model.md`](docs/model.md) 的道路分割研究实现：**条件扩散生成双通道提示热图 → h-maxima + 距离 NMS → 道路适配 SAM → 拓扑奖励后训练**。

> 验证状态：按后续授权在已有 `pytorch` conda 环境运行 CPU 单元测试，七个模块共 **73 项通过**；RTX 3050 小规模 GPU 前向/反向、AMP 和扩散轨迹重算检查通过。未安装或升级依赖，未启动完整训练。测试范围与环境差异见 [`docs/testing.md`](docs/testing.md)。3090 24GB 配置仍是未在目标卡上实测的实验起点，不保证完整训推显存、速度或效果。

## 1. 环境与权重（自行安装）

推荐 Linux / WSL2、Python **3.10**、单张 NVIDIA RTX 3090 24GB。PyTorch CUDA wheel 包含运行时；主机需安装支持 CUDA 12.1 的 NVIDIA 驱动，常规使用不需要另装 CUDA toolkit。WSL2 应使用 Windows 侧驱动。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

固定依赖见 `requirements.txt`。不需要 diffusers、transformers、xformers、OpenCV 或自定义 CUDA 编译扩展。原版 SAM API 使用 `segment-anything==1.0`。

手动从 [SAM 官方仓库](https://github.com/facebookresearch/segment-anything) 获取 ViT-B 权重 `sam_vit_b_01ec64.pth`，放到 `checkpoints/`，或修改 YAML 的 `model.sam_checkpoint`。官方权重地址：

```text
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

代码不会自动下载权重。只加载可信的 checkpoint，训练快照使用 PyTorch 序列化，不能把陌生来源文件当成安全输入。

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
python scripts/prepare_splits.py --root DeepGlobe_Road --output splits --seed 42 --val-fraction 0.1
```

生成 `train.json`、`val.json`（默认约 5,603 / 623），及 `unlabeled.json`、`protocol.json`。脚本拒绝覆盖已有划分。按原影像划分后才做增强，避免同图裁块泄漏；随机划分仍不保证地理独立，有地理分组时应替换为区域隔离划分。内部验证结果不是官方隐藏测试集成绩。

影像等比例缩放到最长边 1024、右下填充；SAM 输入保持 RGB `[0,1]`，标准化在 SAM 封装内部完成。道路骨架和采样位置在原标注分辨率构建，再映射到 256² 热图，不先缩小二值道路再求骨架。

## 3. 代码组织

```text
configs/3090.yaml           单卡实验配置
scripts/prepare_splits.py  数据清单与复现划分
diffpeak/
  data/                    数据读取、增强、随机监督布局
  prompts.py               固定峰值提取与目标回提检查
  models/
    sam.py                 共享冻结图像编码器及可适配解码器
    conditioning.py        细节 CNN、多尺度条件与辅助头
    unet.py                四尺度条件去噪网络
    diffusion.py           噪声监督、随机采样、轨迹对数概率
  losses.py                BCE / Dice / soft-clDice 与辅助损失
  rewards.py               IoU、clDice、可选双向路径代理奖励
  engine.py                阶段冻结、训练、验证与 checkpoint
  inference.py             无标注推理与几何恢复
train.py / predict.py / evaluate.py
```

## 4. 训练阶段

所有命令在项目根目录运行。下一阶段使用上一阶段的权重初始化；阶段内恢复则使用 `--resume`，不要把二者混用。

| 阶段 | 可更新部分 | 默认 epochs / LR | 监督 |
|---|---|---|---|
| A | SAM mask decoder、细节与条件分支、辅助头 | 15 / 1e-4 | 真实提示分割 + 区域/中心线辅助 |
| B | 去噪 U-Net、细节与条件分支、辅助头 | 60 / 1e-4 | 扩散 MSE + 热图重建 + 辅助 |
| C | SAM mask decoder | 10 / 5e-5 | 真实生成提示下的分割 |
| D | 去噪 U-Net | 10 / 3e-6 | DDPO + 固定参考策略 KL + 扩散锚定 |

图像编码器与提示编码器始终冻结。D 的条件模块和 SAM 解码器也冻结。B/C/D 验证基于生成提示，不使用验证标注构造提示再宣称自动分割。

```bash
# 推荐先检查目标热图回提（不需要 SAM 权重）
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

以上数据准备/训练/完整评估命令供你安装依赖后执行，本次未运行这些命令。CPU/GPU 小规模测试单独记录于 [`docs/testing.md`](docs/testing.md)。`--stage all` 按 A→B→C→D 使用阶段结束权重继续；`stage_X.pth` 为阶段结束完整快照，`last.pt` 为最近保存的完整 epoch，`best.pt` 按当前阶段验证 IoU 选择，会随新阶段重新记录。若希望每阶段使用最佳权重而非末轮权重，分阶段运行并把该阶段 `best.pt` 另存后传给下一阶段 `--init`。建议不同实验使用不同 `output_dir`，不要共享输出目录。

`predict.py`/`evaluate.py` 省略 `--config` 时使用 checkpoint 内的配置；显式传入 YAML 表示你主动变更协议。checkpoint 保存已解析的数据、划分、SAM 和输出绝对路径；换机器时应显式提供含新路径的配置。恢复以完整 epoch 为单位，不承诺逐 batch 中断恢复。`evaluate.py` 的 `--limit` 只用于子集调试，不应与完整验证集结果混用。

默认 batch size 1、梯度累积 4、AMP、U-Net 激活检查点；D 使用 `rl.amp_init_scale=64`，避免联合高维策略梯度配合默认大 scale 时的首批溢出。A/B/C 的有效 batch size 为 4。D 每图 4 条轨迹串行采样，轨迹 CPU 暂存；它的轨迹组与监督 batch 的语义不同。

## 5. 扩散、峰值与 RL 协议

- SAM ViT-B：输入 1024²，冻结编码特征 256×64²。
- 条件金字塔：256/128/64/32，通道 64/128/256/256。
- 两个独立提示通道：256²，既不互斥 Softmax，也不逐图 min-max。
- 训练扩散 1,000 步，生成 20 步；所有随机转移方差非零，包括末步，标准差下限 0.02。
- 固定峰值：平滑 sigma=0.5，h=0.05，绝对阈值 0.3，NMS 半径 4 热图像素。
- 每通道峰数可变，无固定 K；不设置隐含 top-k。异常密集热图可能带来很高解码成本，需要监控。
- 正点/负点一起输入 SAM；没有正峰时输出空掩码，不强行补全全局最大点。
- 默认奖励 `0.6*IoU + 0.4*clDice - 0.05*(Kpos+Kneg)/32`。
- 原始 `checkpoints/sam_vit_b_01ec64.pth` 在训练和预测构造时都需要保留，即便训练快照已经包含 SAM 权重。完整快照较大，实验磁盘空间需预留。
- 可开启双向抽样路径代理（例如权重 0.5/0.3/0.2），**不是正式 APLS**；二值标注无法区分立交跨越。
- PPO 联合动作概率对所有热图维度**求和**，没有用均值替代联合密度；默认裁剪范围 1e-4 是高维策略的谨慎起点，需监控实际 KL、奖励方差与更新比例。KL 同样对动作维度求和，再按样本/轨迹/时间步取平均；权重不能直接照搬使用逐像素均值的项目。
- 同图 4 轨迹使用 leave-one-out 奖励基线。对峰值提取/SAM 奖励不反传；策略梯度来自保存动作的重新计分。
- 推理默认沿用随机生成协议；确定性采样不能默认与训练等价。

## 6. 实验注意事项

先做 A→B→C 的监督链条再进行 D；D 可能因奖励分段不变而停滞，不保证必然改善。先检查目标热图的 `target_recovery_report`，确保随机目标峰经最终提取器仍可恢复。困难负点只在具备阶段性假阳性数据时启用，不把普通背景采样冒充困难负点挖掘。

建议分别记录 IoU、F1、clDice、正负点数分布、无正峰率、点落点正确率、多种子波动和包含峰值 CPU 处理的端到端耗时。指标聚合需区分逐图宏平均与全数据混淆矩阵微平均。无道路图的空预测得分规则见奖励实现。

若 OOM：先检查是否意外取消编码器冻结；保持 batch=1、检查点及 AMP；降低 worker 数减少主存压力但不会直接降低 GPU 激活显存；必要时减少梯度累积组中保留的对象和 RL 轨迹数（G≥2）。不要把降低 SAM 图像尺寸当成无影响开关，原版几何和坐标协议需要一并修改。默认配置尚未做显存实测。

本实现是原设计的可扩展研究基线，不是复现已发表性能的成品；交叉注意力、自动困难负点闭环、正式 APLS 评估、细边界细化和全部消融实验需单独开展。

## 7. 测试入口

```bash
# 当前 pytorch 环境已通过的轻依赖 CPU 子集
conda run -n pytorch python -m unittest tests.test_losses tests.test_diffusion tests.test_conditioning_and_unet tests.test_sam tests.test_geometry tests.test_prepare_splits tests.test_splits_utils -v

# 当前 RTX 3050 已通过的小规模 GPU 检查（模拟 SAM 特征）
conda run -n pytorch python scripts/gpu_smoke.py

# 安装全部依赖后自行执行；当前没有跑完这套完整测试
python -m unittest discover -s tests -v
```

GPU smoke 不需要真实 SAM 权重；它使用 128² 输入、32² 热图、G=2 和 4 步采样，验证 AMP、梯度与高斯策略重算，而非完整 SAM 分割效果。实测范围、初始失败和修复、环境版本差异见 [`docs/testing.md`](docs/testing.md)。
