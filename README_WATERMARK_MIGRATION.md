# 攻击效果评估 README（InvWAN 水印方案）

本文档专门用于“评估攻击效果”，包含：
- 评估流程（从输入到指标）
- 水印算法配置
- 水印算法权重
- 预处理流程
- 可直接执行的评估命令

## 1. 评估目标与指标

评估攻击模型输出 `I_R`（重建后图像）在视觉质量和水印解码上的表现。

- 视觉指标
  - `PSNR(IR, I)` / `SSIM(IR, I)`：与干净图对比
  - `PSNR(IR, Iw)` / `SSIM(IR, Iw)`：与水印图对比
- 水印指标
  - `BER`（Bit Error Rate，越高表示攻击越成功）
  - `BitAcc`（Bit Accuracy，越低表示攻击越成功，`BitAcc = 1 - BER`）

默认评估脚本中常见两个尺度：
- `256` 尺度：攻击网络原生输出尺度
- `400` 尺度：将 `I_R(256)` 双三次上采样到 `400` 后再评估（与 Stega 解码分辨率对齐）

## 2. 评估所需文件（最小集合）

### 2.1 核心代码
- `train.py`（`load_ber_decoder`、`build_multiwm_decoders`）
- `datasets.py`
- `model.py`、`backbone.py`、`losses.py`（攻击模型前向与指标）
- `utils/preprocess.py`
- `utils/watermarkers.py`

### 2.2 水印算法包与权重
- `stegastamp_pkg/`
- `ssl_watermarking-main/`
- `HiDDeN-master/`

### 2.3 关键权重文件
- StegaStamp
  - `stegastamp_pkg/weights/encoder_best_loss_0.005250_step_66185.pth`
  - `stegastamp_pkg/weights/decoder_best_loss_0.005250_step_66185.pth`
- SSL
  - `ssl_watermarking-main/models/dino_r50_plus.pth`
  - `ssl_watermarking-main/normalayer/out2048_coco_orig.pth`
  - `data/invwan_multiwm_400_256/labels/ssl_carrier_seed2025.pt`
- HiDDeN
  - `HiDDeN-master/experiments/jpeg-compression/checkpoints/epoch-300.pyt`

### 2.4 评估脚本
- `scripts/eval_attack_all_best_checkpoints.py`
- `scripts/eval_attack_prepared_test.py`
- `scripts/eval_ablation_variants.py`
- `scripts/test_attack_on_images.py`
- `scripts/test_multiwm_decoders.py`
- `scripts/verify_ssl_decoder.py`

## 3. 水印算法配置与权重映射

| 算法 | payload_bits | 配置键 | 必需权重/文件 |
|---|---:|---|---|
| StegaStamp | 100 | `evaluation.ber` 或 `evaluation.multiwm.algos.stega` | encoder + decoder |
| DWT-DCT-SVD | 32 | `evaluation.multiwm.algos.dwt` | 无外部权重 |
| RivaGAN | 32 | `evaluation.multiwm.algos.rivagan` | `onnxruntime`（`imwatermark` 内置 ONNX 模型） |
| SSL-Latent | 30 | `evaluation.multiwm.algos.ssl` | `dino_r50_plus.pth` + `out2048_coco_orig.pth` + `ssl_carrier_seed2025.pt` |
| HiDDeN | 30 | `evaluation.multiwm.algos.hidden` | `epoch-300.pyt` |

### 3.1 单水印（Stega）配置模板

```yaml
evaluation:
  ber:
    encoder_path: stegastamp_pkg/weights/encoder_best_loss_0.005250_step_66185.pth
    decoder_path: stegastamp_pkg/weights/decoder_best_loss_0.005250_step_66185.pth
    decode_size: 400
```

### 3.2 多水印配置模板（Stega + DWT + RivaGAN + SSL + HiDDeN）

```yaml
dataset:
  multiwm:
    enabled: true
    algos: [stega, dwt, rivagan, ssl, hidden]

evaluation:
  multiwm:
    enabled: true
    algos:
      stega:
        type: stega
        encoder_path: stegastamp_pkg/weights/encoder_best_loss_0.005250_step_66185.pth
        decoder_path: stegastamp_pkg/weights/decoder_best_loss_0.005250_step_66185.pth
        decode_size: 400
      dwt:
        type: dwt
        payload_bits: 32
      rivagan:
        type: rivagan
        payload_bits: 32
      ssl:
        type: ssl
        payload_bits: 30
      hidden:
        type: hidden
        payload_bits: 30
```

## 4. 预处理流程（评估链路）

### 4.1 Prepared Test 数据集模式（推荐）

对应脚本：`scripts/eval_attack_all_best_checkpoints.py`、`scripts/eval_attack_prepared_test.py`

数据格式（单水印 Stega 评估）：
- `data/invwan_400_256/test/256/I/*.png`
- `data/invwan_400_256/test/256/Iw/*.png`
- `data/invwan_400_256/test/400/I/*.png`
- `data/invwan_400_256/test/400/Iw/*.png`
- `data/invwan_400_256/test/labels/bits.npy`
- `data/invwan_400_256/test/labels/files.txt`

评估时的实际预处理：
1. 读取 `Iw(256)` 作为攻击网络输入。
2. 用 `I(256)` 双三次下采样得到 `I_LR_ref(64)`（`scale=4`）。
3. 前向攻击：`I_R = pipeline(Iw, I, I_LR_ref, gaussian_scale=...)`。
4. `256` 尺度指标：`I_R` 对比 `I(256)` / `Iw(256)`。
5. `400` 尺度指标：`I_R` 双三次上采样到 `400`，再对比 `I(400)` / `Iw(400)`。
6. BER 用 decoder 计算（decoder 内部会做自身需要的 resize / normalize）。

### 4.2 原始图片快速评估模式

对应脚本：`scripts/test_attack_on_images.py`

预处理流程：
1. 原图先做方形化到 `400x400`（`fit` 或 `pad`）。
2. 使用 Stega encoder 在 `400` 尺度嵌入 100-bit。
3. 下采样到 `256` 后送攻击网络。
4. 输出 `I_R(256)` 上采样回 `400` 做指标与 BER。

### 4.3 decoder 侧预处理（按算法）

- Stega：`load_ber_decoder` 包装器会将输入 resize 到 `decode_size`（默认 400）。
- DWT：解码前强制 resize 到 `256`。
- SSL：先按比例缩放 + 中心裁剪到 `128`，再做 ImageNet normalize，最后 `feature @ carrier` 解码。
- HiDDeN：resize 到 `128`，映射到 `[-1, 1]` 后解码。

## 5. 评估执行流程（可直接跑）

默认都在仓库根目录执行。

### 5.1 单次评估：所有 run 的 best checkpoints

```bash
python scripts/eval_attack_all_best_checkpoints.py
```

输入：
- 自动扫描 `runs/*/checkpoints/best_checkpoints/*.pt`
- 使用 `data/invwan_400_256/test`

输出：
- `outputs/eval_bestcheckpoints/all_runs_bestcheckpoints_summary.json`

### 5.2 单 checkpoint 评估（Prepared Test）

脚本：`scripts/eval_attack_prepared_test.py`

该脚本顶部常量需要先改：
- `CONFIG_PATH`
- `CHECKPOINT_PATH`
- `DATA_ROOT`
- `ENCODER_PATH` / `DECODER_PATH`

然后执行：

```bash
python scripts/eval_attack_prepared_test.py
```

输出：
- `损失权重超参数实验/结果/per_image_<ckpt>.json`
- `损失权重超参数实验/结果/summary_<ckpt>.json`

### 5.3 多变体批量评估（含 BitAcc/PSNR/SSIM + 复杂度）

```bash
python scripts/eval_ablation_variants.py --device cuda --output outputs/ablation_eval_summary.json
```

输出：
- `outputs/ablation_eval_summary.json`
- `outputs/ablation_eval_summary.csv`

### 5.4 多水印 decoder 自检（推荐先跑）

```bash
python scripts/test_multiwm_decoders.py \
  --dataset-root data/invwan_multiwm_400_256 \
  --config configs/train_e2e_cleaner_multiwm_all.yaml \
  --num-samples 10
```

SSL专项一致性检查：

```bash
python scripts/verify_ssl_decoder.py \
  --dataset-root data/invwan_multiwm_400_256 \
  --split val \
  --payload-bits 30
```

## 6. 指标解释与结论规则

- 攻击强度优先看 `BER`（或 `BitAcc`）
  - `BER` 越高越好（对攻击方）
  - `BitAcc` 越低越好（对攻击方）
- 可见性约束看 `PSNR/SSIM`
  - 视觉质量越高通常意味着攻击更隐蔽
- 报告建议同时给出
  - `BER + PSNR(IR,I) + SSIM(IR,I)`（核心三元组）
  - 若含 400 尺度评估，再补 `ber400/psnr400/ssim400`

## 7. 迁移到新仓库时的必改项（硬编码）

### 7.1 SSL carrier 绝对路径

`train.py` 的 `build_multiwm_decoders` 中，SSL carrier 目前是硬编码绝对路径。迁移后建议改为配置项 `carrier_path`。

### 7.2 脚本中的 PROJECT_ROOT 与固定路径

以下脚本有明显硬编码路径，迁移后先改常量再跑：
- `scripts/eval_attack_prepared_test.py`
- `scripts/test_attack_on_images.py`
- `scripts/eval_ablation_variants.py`（默认 `VARIANT_OVERRIDES` 里是绝对路径）

## 8. 依赖建议

最少依赖：
- `torch`, `torchvision`
- `numpy`, `Pillow`, `opencv-python`
- `timm`, `scipy`, `pandas`, `tqdm`, `pyyaml`
- `onnxruntime`（RivaGAN 解码/嵌入需要）
- 可选：`bchlib`

参考：
- `ssl_watermarking-main/requirements.txt`
- `JiT/environment.yaml`
