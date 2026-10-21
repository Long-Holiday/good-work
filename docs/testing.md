# 测试记录

## 当前环境（2026-10-21）

用户后续授权使用已有 `pytorch` conda 环境和本机 RTX 3050 进行小规模测试。没有安装或升级任何包，没有下载 SAM 权重。

| 项目 | 实际环境 |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.14.0+cu130 |
| NumPy | 2.4.6 |
| Pillow | 12.3.0 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU，约 4 GiB |
| CUDA | 可用 |
| 缺失依赖 | scipy、scikit-image、PyYAML、segment-anything |

这不是 `requirements.txt` 推荐的 Python 3.10 / torch 2.5.1 cu121 环境，因此测试结果不能视作该固定依赖组合的认证。

## CPU 单元测试

```bash
conda run -n pytorch python -m unittest \
  tests.test_losses tests.test_diffusion tests.test_conditioning_and_unet \
  tests.test_sam tests.test_geometry tests.test_prepare_splits \
  tests.test_splits_utils -v
```

最终结果：**73 tests，OK，exit code 0**。另外运行 `conda run -n pytorch python -m compileall -q diffpeak tests scripts train.py predict.py evaluate.py check_targets.py`，源码语法编译通过。

覆盖：损失空图/细线/float16 大图归约、扩散调度与非零末步方差、联合动作 log-prob 求和、条件/U-Net 小尺寸前向、SAM 空提示及填充张量逻辑（无真实权重）、几何映射、划分生成及路径约束。

过程中有两次未通过：第一轮 55 项通过但两个测试模块因数据包初始化提前加载缺失的 skimage 而导入失败；将数据包重依赖按需导入后，第二轮遇到测试文件临时动态导入的 dataclass 注册错误。恢复正常模块导入后，上述最终命令 73 项全部通过。这些初始失败不隐去，也不是最终仍存在的失败。

## RTX 3050 小规模 GPU 测试

```bash
conda run -n pytorch python scripts/gpu_smoke.py
```

最终 **exit code 0**，脚本内约 4.72 秒。输入 128²、提示潜变量/目标热图均为 32²；使用默认 U-Net 通道 64/128/256/256、梯度检查点、CUDA float16 autocast 和 GradScaler。SAM 编码特征使用随机张量模拟，**没有加载真实 SAM 或计算真实分割奖励**。

| 检查 | 结果 |
|---|---|
| B 条件 + 扩散 + 辅助损失前向 | 全部有限，total≈1.80309 |
| B 反向及参数更新 | 无缺失/非有限梯度，参数确实更新 |
| D CPU generator 固定种子重采样 | 最大差 0 |
| D 保存轨迹与重算 log-prob | 初始 ratio 最大误差 0 |
| D state/action / log-prob 精度 | float32 / float64，CPU 暂存 |
| D PPO + KL + anchor 反向 | 有效非零梯度，更新后参数有限 |
| B / D 峰值 allocated | 339.51 / 369.11 MiB |
| 峰值 reserved | 386 MiB |

G=2、扩散训练步数 32、采样 4 步，仅为快速验证算子、梯度、采样/重算一致性。不是 1024² 正式性能测试，**不能把这些显存数字外推为 3090 的完整训练占用**。

测试中默认 AMP 初始 scale=65536 曾使 DDPO 的 fp16 中间梯度溢出；改为 64 后通过。业务引擎已加入 `rl.amp_init_scale=64`，并在 AMP 跳过更新时不消耗 scheduler/global_step。GPU smoke 的梯度增长间隔较长，不构成长程训练数值稳定性的保证。它验证的是模块组合而非完整 `engine.py` 控制流；完整引擎仍受缺失依赖/权重限制。

## 未执行范围

- 缺少 scipy / scikit-image，未执行数据采样、真实 DeepGlobe 监督样本、h-maxima、完整奖励与路径代理相关测试。
- 缺少 PyYAML / segment-anything 及真实权重，未执行真实 SAM、完整 Predictor、完整 A–D 训练/验证、checkpoint 端到端恢复或 CLI 全链条。
- 未在 RTX 3090 24GB 上验证 1024² 训练峰值显存、吞吐、收敛或精度。
- `tests/` 包含上述领域的测试代码不代表这些测试已经通过；全部依赖安装完毕后可运行 `python -m unittest discover -s tests -v`。
