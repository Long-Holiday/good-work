# ABC 联合预训练与 D 后训练

当前训练协议为 `training_protocol: joint_abc_v1`，入口只接受 `ABC`、`D` 或 `all`。这是本任务的联合监督预训练，不是从零预训练 SAM：仍加载官方 SAM2.1 权重。ABC 将 SAM 全部冻结，只更新扩散分支（条件编码器与 DiT）；D 才微调 SAM 掩码解码器及其高分辨率投影，图像骨干和提示编码器保持冻结。

## 模型与预训练

图像经过冻结 SAM 图像编码器以及可训练条件编码器，条件 DiT 生成正/负双通道提示热图。正负通道组合为有符号密集 mask prompt，经 SAM 提示编码器进入掩码解码器。提示编码器虽然参数冻结，前向仍保留热图输入的梯度。没有正峰的样本也参与分割训练，避免旧空提示常量输出阻断学习。

每批使用一个总损失：

`L_ABC = L_diff + heat_weight * L_heat + aux_weight * L_aux + seg_weight * L_seg_reconstructed + sampled_seg_weight * L_seg_generated`

- `L_diff`：随机时间步噪声预测 MSE。
- `L_heat`：峰值/背景平衡的热图重建损失。
- `L_aux`：条件分支的道路/骨架辅助监督。
- `L_seg_reconstructed`：由带噪目标的一步重建热图进入 SAM，计算 BCE/Dice/clDice。
- `L_seg_generated`：从纯噪声按完整采样调度生成热图，再进入 SAM 计算同样的分割损失。

两条分割通路都反传到 DiT 与条件编码器；后者覆盖推理时的提示分布，不依赖单独 A 初始化或 C 适配。分割监督只计算每图有效区域。ABC 优化器只包含条件编码器与 DiT，SAM 全部参数 requires_grad=False 并保持 eval。SAM 提示编码器和解码器前向仍保留计算图，不能包裹 no_grad 或 detach 输入热图；冻结权重不妨碍计算分割损失对热图的梯度。梯度累积后统一更新，没有逐模块训练或交替冻结。

完整采样的反传比旧 B 训练昂贵。2026-10-22 在 seetacloud3 的全尺寸 ABC 短测后，默认采用 batch=4、累积=1、workers=4、关闭 DiT 激活检查点；120 张测量约 4.095 张/秒、峰值 reserved 16.03 GiB，比原 batch1×累积4、开启检查点的短测约快 2.94 倍。有效 batch 仍为 4，但随机采样及数值行为不保证逐步等价；这不是收敛或长期显存保证。完整候选、计时范围和产物位置见 docs/probe_3090.md。D 联合版本未单独测速，保留 batch=1。

## 整体强化学习

D 的动作包括扩散过程的高斯转移，以及 SAM logits 定义的终端 Bernoulli 二值分割。奖励在采样的终端分割上计算，每张图的 G 个候选独立形成一组，组内标准化为优势。

扩散转移重放重新计算带梯度的条件特征，更新条件编码器和 DiT；终端动作的概率重放更新 SAM 解码器。两部分共享同一组奖励和优化器。采样与重放全部采用 eval 模式，参数仍可求梯度，避免 dropout 等造成旧/新策略初始概率不一致。

高斯与 Bernoulli 的动作维 log-prob 都求和；分别在扩散的 `G*T*B` 和终端的 `G*B` 上平均策略项。KL 分别对应固定参考条件编码器/DiT 与固定参考 SAM 解码器。可选 `rl.anchor_weight` 对完整 ABC 总损失加锚定，默认关闭。只有奖励完全相同且 KL/锚定均关闭时才跳过该组更新。

密集提示不消耗离散点，实际点数成本为零。峰值统计仅作热图诊断。验证/推理采用确定性阈值化 SAM 概率，D 的采样奖励与确定性验证指标应分开解读。

## 权重与恢复

保存 `stage_ABC.pth`、`stage_D.pth` 以及 `last.pt`、`best.pt`。D 快照同时保存三个固定参考模块，恢复时严格要求参考完整。旧 A/B/C/D 优化器状态不能直接 `--resume` 新协议；此前训练 SAM 的 ABC 快照也不能直接恢复到全冻结版本，因为优化器参数集合不同。新快照以 `abc_sam_frozen: true` 标识；旧快照可用显式新配置加 `--init` 只迁移兼容模型权重。

```bash
conda run -n pytorch python train.py --config configs/3090.yaml --stage all
conda run -n pytorch python train.py --resume runs/diffpeak/last.pt
```

新协议必须重新做验证集与完整训练评估。扩散 loss 下降本身不能证明热图有效或分割收敛；应同时检查生成热图、空预测率、IoU/clDice 和 D 的奖励方差。
