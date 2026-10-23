# 单通道山脊场训练

当前协议为 `training_protocol: signed_ridge_v1`，目标版本为 `target_representation: signed_ridge_v1`。训练入口只有 `ABC`、`D` 和 `all`。设计依据见 [改造方案](signed_ridge_diffusion_plan.md)。这次实现更新代码和回归契约，真实预训练 SAM 的提示有效性、收敛及 GRPO 收益仍需实验验证。

## 模型与数值约定

冻结 SAM2.1 Hiera-B+ 提供图像特征，可训练条件编码器融合细节特征，条件 DiT 生成 `1×256×256` 的有符号道路山脊场。有效道路中心线为 +1，道路内部按到边界与中心线的相对距离变化，有效背景为 -1。padding 使用独立 `field_valid_mask`，输入 SAM 时固定为 0，不计入损失或分割奖励。

构造目标时先同步增强图像和标注，建立等比缩放与右下 padding 几何；保留细道路的目标投影、区域划分及诊断由 `diffpeak/data/ridge.py` 实现。`check_targets.py` 检查场语义和细路保留，不提取离散峰值。

扩散直接使用有符号目标，不再执行 `2H-1` 或逆变换，不添加旧背景均值补偿。带噪状态不裁剪；最终输出按固定 `clamp(-1,1)` 处理，然后通过 `U=prompt.scale*H` 输入 SAM mask prompt。训练、验证和推理使用同一映射。`prompt.scale=1` 是待验证的起始值。

SAM 的图像编码器、提示编码器、掩码解码器及高分辨率投影在 ABC 和 D 全部 `requires_grad=False` 并保持 eval。SAM 图像特征可以在 no_grad 下提取；分割监督所需的提示编码及掩码解码保留输入梯度，损失可经 SAM 回传到条件编码器和 DiT。

## ABC 监督预训练

`L_ABC = L_diff + heat_weight*L_field + aux_weight*L_aux + seg_weight*L_seg_reconstructed + sampled_seg_weight*L_seg_generated`

- `L_diff`：有效场区域的随机时间步噪声 MSE。
- `L_field`：中心线邻域、其余道路、边界外窄带、远端背景四个互斥区域分别归一化，再对存在的区域求平均；缺失区域跳过。低噪声重建项保留时间步门控。
- `L_aux`：条件分支道路与骨架的 BCE/Dice，排除 padding。
- `L_seg_reconstructed`：带噪目标的一步重建场经冻结 SAM 的 BCE/Dice/clDice。
- `L_seg_generated`：从纯噪声完整可微采样场经冻结 SAM 的相同分割损失，覆盖实际生成分布。

优化器只更新条件编码器和 DiT。扩散损失下降不能证明生成有效；应同步检查完整采样后的场分布、空预测率和分割指标。单通道配置延续旧吞吐设置作为起点，旧双通道 3090 吞吐不能直接作为新协议的显存或速度保证。

## D 扩散轨迹 GRPO

同一影像独立生成 G 条高斯扩散轨迹，冻结 SAM 输出经 `sigmoid >= inference.threshold` 确定性分割后计算奖励。每张影像单独形成一组，优势为 `(R-mean(R))/max(std(R), advantage_std_delta)`。

`L_D = L_GRPO_diff + kl_weight*L_KL_diff + anchor_weight*L_ABC`

奖励为 IoU、clDice 和可选路径代理的加权和；默认先关闭路径项。路径采样在同图候选之间固定种子。没有离散点、点成本、终端 Bernoulli 动作、终端策略损失或参考 SAM 解码器。

高斯动作维 log-prob 与 KL 均求和，再按 `G*T*B` 归一。每条轨迹保留裁剪前的原始高斯动作，终端场裁剪不改变动作概率定义。所有转移（包括末步）由正的 `min_std` 保持非退化；不对确定性动作计算非退化高斯密度。采样和重放采用一致的 eval/AMP 模式。

D 同时更新条件编码器与 DiT，固定参考条件编码器和 DiT 提供 KL。默认保留 `anchor_weight=0.01` 的完整 ABC 监督锚定；设为 0 进行消融。奖励完全常量且 KL、锚定均关闭时不 backward、不推进优化器和学习率调度；非零低方差组仍参与学习。监测奖励方差、活跃组比例、联合 log-ratio、KL、裁剪比例和实际更新数。单通道改变联合概率尺度，超参数必须重新校准。

## 检查点与迁移

版本 5 快照记录协议、目标表示、单通道、SAM 全冻结、提示映射、完整配置、扩散调度、GRPO 配置及状态。D 快照仅保存 `ref_condition_encoder` 和 `ref_dit` 两个参考。`--resume` 严格恢复新协议的模型、参考、优化器、调度器、GradScaler 与 RNG。

旧双通道或曾微调 SAM 的快照禁止直接 `--resume` 或普通 `--init`。若需要迁移，显式指定新配置并使用：

```bash
python train.py --config configs/3090.yaml --stage ABC \
  --init old_checkpoint.pt --migrate-legacy --output-dir runs/ridge-migrated
```

只迁移条件编码器及 DiT 的兼容内部权重，不迁移 SAM、DiT 输入/输出头、优化器、参考或训练进度。终端会报告加载、跳过和重新初始化项；该入口不是断点恢复。

```bash
python check_targets.py --config configs/3090.yaml --samples 16
python scripts/diagnose_ridge_sam.py --help
python train.py --config configs/3090.yaml --stage all
python train.py --resume runs/diffpeak/last.pt
python train.py --config configs/3090.yaml --stage D --init runs/diffpeak/stage_ABC.pth
```

正式训练前，用验证集标注场与二值 mask 提示比较若干固定尺度，检查正常、零值、打乱及局部扰动提示的响应，再确定提示尺度。优先比较确定性同场预测基线、监督扩散、监督加 GRPO、GRPO 无锚定；主要报告固定协议的单次采样及跨种子波动，多候选最优只作潜力诊断。回归测试通过不能替代这些实验。
