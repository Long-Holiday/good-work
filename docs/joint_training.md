# 单通道山脊场训练

当前协议为 `training_protocol: signed_ridge_v1`，目标版本为 `target_representation: signed_ridge_v1`。训练入口只有 `ABC`、`D` 和 `all`。设计依据见 [改造方案](signed_ridge_diffusion_plan.md)。这次实现更新代码和回归契约，真实预训练 SAM 的提示有效性、收敛及 GRPO 收益仍需实验验证。

2026-10-23 恢复两条 SAM 分割监督。仅场生成版本的前几轮验证 IoU 约 0.025–0.029，前景/背景响应接近；这不证明仅场损失绝不收敛，但当前实验按用户指令终止并保留产物。新运行 `diffpeak_seg_restored_20261023T094816` 从此前同协议、两条分割监督的最佳 ABC 权重初始化（旧第3轮 val IoU 0.204288），使用新优化器与调度，不能视为优化器状态连续恢复。LR3e-5 为保守起点，未在恢复后的目标下重新调优；batch4/累积1/workers8/CPU8、关闭检查点、20步可微采样、真实3090三次更新检查通过，reserved16390MiB，无AMP跳步；28项梯度/流程/CLI回归通过。新实验独立目录 `/root/autodl-tmp/runs/diffpeak_seg_restored_20261023T094816/` 包含启动配置、源代码差异、初始化SHA-256与日志；新验证结果到达前不声称已恢复0.2043的效果。

## 模型与数值约定

2026-10-23 骨干由 DiT 切换为多尺度条件 U-Net。上述 Run04 是历史 DiT 实验，不代表 U-Net 的训练结果。新模型在 256/128/64/32 四尺度编码和解码层分别注入条件，逐块使用时间 scale/shift 调制，通过拼接跳连保留细节；默认通道 64/128/256/256，每级两个残差块，仅32²尺度使用8头自注意力。默认开启非重入梯度检查点。随后按用户要求切换为条件Flow Matching，并保留随机轨迹GRPO。旧DDPM/DDIM不再是默认生成过程，历史吞吐数据不能直接复用。网络接收[0,1]浮点时间，内部乘time_scale=1000，不量化。


冻结 SAM2.1 Hiera-B+ 提供图像特征，可训练条件编码器融合细节特征，条件 U-Net 生成 `1×256×256` 的有符号道路山脊场。有效道路中心线为 +1，道路内部按到边界与中心线的相对距离变化，有效背景为 -1。padding 使用独立 `field_valid_mask`，输入 SAM 时固定为 0，不计入损失或分割奖励。

构造目标时先同步增强图像和标注，建立等比缩放与右下 padding 几何；保留细道路的目标投影、区域划分及诊断由 `diffpeak/data/ridge.py` 实现。`check_targets.py` 检查场语义和细路保留，不提取离散峰值。

Flow Matching直接使用有符号目标，不执行 `2H-1` 或逆变换，不添加旧背景均值补偿。带噪状态不裁剪；最终输出按固定 `clamp(-1,1)` 处理，然后通过 `U=prompt.scale*H` 输入 SAM mask prompt。训练、验证和推理使用同一映射。`prompt.scale=1` 是待验证的起始值。

SAM 的图像编码器、提示编码器、掩码解码器及高分辨率投影在 ABC 和 D 全部 `requires_grad=False` 并保持 eval。SAM 图像特征可以在 no_grad 下提取；分割监督所需的提示编码及掩码解码保留输入梯度，损失可经 SAM 回传到条件编码器和 U-Net。

## ABC 监督预训练

`L_ABC = L_flow + heat_weight*L_field + aux_weight*L_aux + seg_weight*L_seg_reconstructed + sampled_seg_weight*L_seg_generated`

线性路径为 `x_t=(1-t)*epsilon+t*H*`，目标速度为 `H*-epsilon`。单步终端估计为 `H_hat=x_t+(1-t)*v_theta(x_t,t,C)`；送入SAM前裁剪，场重建损失本身在未裁剪估计上计算。

- `L_flow`：有效场区域的连续随机时间速度 MSE。
- `L_field`：中心线邻域、其余道路、边界外窄带、远端背景四个互斥区域分别归一化，再对存在的区域求平均；缺失区域跳过。默认仅数据端 `t>=field_loss_min_time=0.5` 参与。
- `L_aux`：条件分支道路与骨架的 BCE/Dice，排除 padding。
- `L_seg_reconstructed`：带噪目标的一步重建场经冻结 SAM 的 BCE/Dice/clDice。
- `L_seg_generated`：从纯噪声完整可微采样场经冻结 SAM 的相同分割损失，覆盖实际生成分布。

优化器只更新条件编码器和 U-Net。扩散损失下降不能证明生成有效；应同步检查完整采样后的场分布、空预测率和分割指标。单通道配置延续旧吞吐设置作为起点，旧双通道 3090 吞吐不能直接作为新协议的显存或速度保证。

## D 随机 Flow Matching 轨迹 GRPO

同一影像独立生成 G 条高斯扩散轨迹，冻结 SAM 输出经 `sigmoid >= inference.threshold` 确定性分割后计算奖励。每张影像单独形成一组，优势为 `(R-mean(R))/max(std(R), advantage_std_delta)`。

`L_D = L_GRPO_flow_policy + kl_weight*L_KL_flow_policy + anchor_weight*L_ABC`

奖励为 IoU、clDice 和可选路径代理的加权和；默认先关闭路径项。路径采样在同图候选之间固定种子。没有离散点、点成本、终端 Bernoulli 动作、终端策略损失或参考 SAM 解码器。

采样从t=0噪声端升序走向t=1数据端。对dt>0，Euler均值为 `x+dt*v(x,t)`；默认Heun先作Euler预测，再用起点和终点速度的平均计算均值。动作服从 `N(mean, policy_std²*dt*I)`，所有步（包括末步）都加噪；中间状态不裁剪。此策略是为GRPO定义的随机积分核，不是确定性FM的精确边缘分布，需靠ABC生成监督和实验验证质量。20步Heun需要40次速度网络前向，而非20次。

高斯动作维 log-prob 与 KL 均求和，再按 `G*T*B` 归一。每条轨迹保留裁剪前原始动作和浮点时间；终端裁剪不改变动作概率。固定参考使用相同求解器、步数和方差，KL为同方差高斯解析KL。采样和重放采用一致的eval/AMP模式；不把旧DDIM概率用于FM。

D 同时更新条件编码器与 U-Net，固定参考条件编码器和 U-Net 提供 KL。默认保留 `anchor_weight=0.01` 的完整 ABC 监督锚定；设为 0 进行消融。奖励完全常量且 KL、锚定均关闭时不 backward、不推进优化器和学习率调度；非零低方差组仍参与学习。监测奖励方差、活跃组比例、联合 log-ratio、KL、裁剪比例和实际更新数。单通道改变联合概率尺度，超参数必须重新校准。

## 检查点与迁移

版本7快照显式记录 `conditional_unet_flow_v1` 速度网络、`conditional_flow_matching_v1`生成过程、协议、目标表示、单通道、SAM 全冻结、提示映射、完整配置、积分求解器/步数/探索方差、GRPO 配置及状态。D 快照仅保存 `ref_condition_encoder` 和 `ref_unet` 两个参考。`--resume` 严格恢复同架构的模型、参考、优化器、调度器、GradScaler 与 RNG。旧 DiT 或DDPM式U-Net快照均不得普通初始化或恢复为FM；连续时间、输出含义和生成过程不同。配置改用flow_matching，旧diffusion配置显式拒绝。

恢复分割监督后，未使用分割损失的场生成快照不能直接 `--resume`：引擎会检查分割权重，目标不同必须用显式配置与 `--init` 新建实验。此前相同单通道、SAM 全冻结、两条分割监督的 DiT 快照曾用于 Run04 初始化；这不适用于新 U-Net。新骨干必须重新初始化，旧快照仅可显式迁移条件编码器。

旧双通道或曾微调 SAM 的快照禁止直接 `--resume` 或普通 `--init`。若需要迁移，显式指定新配置并使用：

```bash
python train.py --config configs/3090.yaml --stage ABC \
  --init old_checkpoint.pt --migrate-legacy --output-dir runs/ridge-migrated
```

只迁移条件编码器的兼容权重，U-Net 全部重新初始化；不迁移 SAM、旧 DiT、优化器、参考或训练进度。终端会报告加载、跳过和重新初始化项；该入口不是断点恢复。

```bash
python check_targets.py --config configs/3090.yaml --samples 16
python scripts/diagnose_ridge_sam.py --help
python train.py --config configs/3090.yaml --stage all
python train.py --resume runs/diffpeak/last.pt
python train.py --config configs/3090.yaml --stage D --init runs/diffpeak/stage_ABC.pth
```

正式训练前，用验证集标注场与二值 mask 提示比较若干固定尺度，检查正常、零值、打乱及局部扰动提示的响应，再确定提示尺度。优先比较确定性同场预测基线、监督扩散、监督加 GRPO、GRPO 无锚定；主要报告固定协议的单次采样及跨种子波动，多候选最优只作潜力诊断。回归测试通过不能替代这些实验。
