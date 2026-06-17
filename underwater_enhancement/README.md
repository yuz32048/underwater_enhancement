# 基于物理退化建模与多分支 CycleGAN 的水下图像增强

本项目实现一个面向水下图像增强的科研流程：先对 UIEB/raw-890 真实退化图像进行多标签退化分类，再用 UIEB/reference-890 通过 Jaffe-McGlamery 物理模型生成蓝偏、绿偏、低照度和模糊退化图像，随后监督预训练四个专家分支，最后训练 Multi-Branch CycleGAN。

训练阶段只使用 UIEB。测试阶段使用 UIEB 与 EUVP，并输出 PSNR、SSIM、UIQM、UCIQE 和可视化结果。

## 方法概述

核心流程：

```text
UIEB/raw-890 -> 退化分类 -> 真实退化分支数据
UIEB/reference-890 -> Jaffe-McGlamery -> 物理退化分支数据
真实退化 + 物理退化 -> 四专家分支监督预训练
预训练分支 -> Multi-Branch Generator -> CycleGAN 联合训练
UIEB + EUVP -> 测试与指标统计
```

`G_AB` 使用四分支增强器：

- `BlueCastBranch`：白平衡 + VGG/轻量特征 + CNN
- `GreenCastBranch`：逆物理颜色补偿 + CNN
- `LowLightBranch`：Gamma 校正 + CNN
- `BlurBranch`：细节恢复 CNN

四个分支通过 `AttentionFusion` 融合，并保存 attention 权重用于分析。`G_BA` 使用普通 ResNet Generator，判别器为 PatchGAN。

## 数据集准备

默认数据路径：

```text
data/raw_underwater/
├─UIEB/
│  ├─challenging-60
│  ├─raw-890
│  └─reference-890
└─EUVP/
   ├─EUVP_Paired
   ├─Unpaired
   ├─test_samples
   └─eval_data
```

UIEB 的 `raw-890` 与 `reference-890` 会按同名文件优先配对；如果同名失败，则按排序顺序兜底配对。EUVP 不参与训练，只用于测试。

## 环境安装

```bash
pip install -r requirements.txt
```

## 退化分类

输入 `UIEB/raw-890`，输出四类真实退化数据与分类 CSV：

```bash
python scripts/classify_uieb.py ^
  --input-dir data/raw_underwater/UIEB/raw-890 ^
  --output-dir data/processed/UIEB_classified ^
  --csv-path results/classification_result.csv
```

输出：

```text
data/processed/UIEB_classified/{blue_cast,green_cast,low_light,blur}/
results/classification_result.csv
```

CSV 字段包含 `image_name`、四类标签、`mean_a`、`mean_b`、`mean_v`、`laplacian_var`、`edge_density` 等。

## 物理退化生成

输入 `UIEB/reference-890`，每张参考图生成四类物理退化图：

```bash
python scripts/generate_physical_degradation.py ^
  --input-dir data/raw_underwater/UIEB/reference-890 ^
  --output-dir data/processed/physical_degradation ^
  --mapping-csv results/physical_degradation_mapping.csv
```

输出：

```text
data/processed/physical_degradation/{blue_cast,green_cast,low_light,blur}/
results/physical_degradation_mapping.csv
```

mapping 保留每张退化图对应的 reference，供分支预训练监督使用。

## 分支预训练

一次训练四个分支：

```bash
python train_branch.py --branch all --epochs 20 --batch-size 2 --device auto
```

单独训练某个分支：

```bash
python train_branch.py --branch blue
python train_branch.py --branch green
python train_branch.py --branch lowlight
python train_branch.py --branch blur
```

输出：

```text
checkpoints/pretrained_branches/
├─blue_branch.pth
├─green_branch.pth
├─lowlight_branch.pth
└─blur_branch.pth
logs/branch_train_log.csv
outputs/train_samples/branches/
```

## CycleGAN 整体训练

标准训练：

```bash
python train_cyclegan.py ^
  --epochs 100 ^
  --batch-size 2 ^
  --device auto ^
  --pretrained-branch-dir checkpoints/pretrained_branches
```

训练域严格为：

- Domain A：`UIEB/raw-890 + data/processed/physical_degradation/*`
- Domain B：`UIEB/reference-890`

输出：

```text
checkpoints/generator/
checkpoints/discriminator/
checkpoints/best_model/generator_best.pth
logs/cyclegan_train_log.csv
outputs/train_samples/cyclegan/
```

断点续训：

```bash
python train_cyclegan.py --resume checkpoints/generator/latest.pth
```

## 测试

测试 UIEB + EUVP：

```bash
python test.py ^
  --checkpoint checkpoints/best_model/generator_best.pth ^
  --uieb-root data/raw_underwater/UIEB ^
  --euvp-root data/raw_underwater/EUVP
```

测试集包括：

- `UIEB/raw-890`
- `UIEB/challenging-60`
- `EUVP/EUVP_Paired/underwater_dark`
- `EUVP/EUVP_Paired/underwater_imagenet`
- `EUVP/EUVP_Paired/underwater_scenes*`
- `EUVP/Unpaired`
- `EUVP/test_samples`
- `EUVP/eval_data`

输出：

```text
outputs/test_results/
outputs/visual_comparisons/
results/evaluation_metrics.csv
results/average_metrics.csv
results/attention_statistics.csv
```

有 reference 的样本计算 PSNR、SSIM、UIQM、UCIQE；无 reference 的样本自动跳过 PSNR、SSIM，只计算 UIQM、UCIQE。

## 消融实验

普通 CycleGAN：

```bash
python train_cyclegan.py --plain-cyclegan --ablation-name plain_cyclegan
```

无物理退化建模：

```bash
python train_cyclegan.py --no-physical-degradation --ablation-name no_physical
```

无分支预训练：

```bash
python train_cyclegan.py --no-branch-pretrain --ablation-name no_pretrain
```

无 Attention Fusion：

```bash
python train_cyclegan.py --no-attention --ablation-name no_attention
```

去除单个分支：

```bash
python train_cyclegan.py --disable-branch blue --ablation-name no_blue
python train_cyclegan.py --disable-branch green --ablation-name no_green
python train_cyclegan.py --disable-branch lowlight --ablation-name no_lowlight
python train_cyclegan.py --disable-branch blur --ablation-name no_blur
```

消融日志和配置统一保存到 `results/ablation/{ablation-name}/`。

## 评价指标

- PSNR：需要 reference，衡量像素级恢复质量。
- SSIM：需要 reference，衡量结构相似性。
- UIQM：无参考水下图像质量指标。
- UCIQE：无参考水下颜色质量指标。

指标实现位于 `metrics/`，测试脚本会自动区分有参考与无参考样本。

## 输出目录

```text
data/processed/                 # 分类结果与物理退化数据
checkpoints/pretrained_branches/ # 四个分支权重
checkpoints/generator/           # CycleGAN 生成器检查点
checkpoints/discriminator/       # 判别器检查点
checkpoints/best_model/          # 最佳模型
outputs/train_samples/           # 训练可视化
outputs/test_results/            # 增强结果
outputs/visual_comparisons/      # 前后对比图
results/                         # CSV 指标、mapping、消融结果
logs/                            # 训练日志
```
