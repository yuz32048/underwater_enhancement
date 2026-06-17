# 基于物理退化建模与多分支 CycleGAN 的水下图像增强

## 项目概述

本项目提出一种基于 Jaffe-McGlamery 水下成像模型与 Multi-Branch CycleGAN 的水下图像增强方法。

针对真实水下场景中常见的颜色失真、亮度下降和细节模糊等问题，设计面向不同退化类型的专用增强分支，并通过注意力融合机制实现复杂退化场景下的自适应增强。

项目采用：

- UIEB 数据集进行训练
- UIEB 与 EUVP 数据集进行测试

为了增强训练数据的退化多样性并提高模型泛化能力，引入 Jaffe-McGlamery 水下成像模型构建物理退化数据域。

---

## 研究目标

针对以下四类典型退化现象：

- 蓝偏（Blue Cast）
- 绿偏（Green Cast）
- 低照度（Low Illumination）
- 模糊（Blur）

分别设计对应增强分支：

| 分支 | 目标 |
|--------|--------|
| Blue Branch | 蓝偏恢复 |
| Green Branch | 绿偏恢复 |
| Low-Light Branch | 亮度增强 |
| Blur Branch | 去模糊恢复 |

最终通过 Attention Fusion 实现多分支协同增强。

---

# 数据集说明

## UIEB 数据集

### 数据集结构

```text
UIEB
├─challenging-60
├─raw-890
└─reference-890
```

---

### raw-890

包含 890 张真实水下退化图像。

主要退化类型：

- 蓝偏
- 绿偏
- 低照度
- 模糊
- 低对比度
- 混合退化

用途：

- 退化分类
- 分支预训练
- CycleGAN训练
- UIEB测试

---

### reference-890

包含与 raw-890 一一对应的参考图像。

对应关系：

```text
raw-890/1_img.png
      ↕
reference-890/1_img.png
```

用途：

- 清晰图像域
- 分支监督训练目标
- CycleGAN训练目标
- PSNR、SSIM计算

---

### challenging-60

包含 60 张严重退化图像。

无参考图像。

用途：

- 主观评价
- 泛化测试
- UIQM评价
- UCIQE评价

---

## EUVP 数据集

### 数据集结构

```text
EUVP
├─EUVP_Paired
│  ├─underwater_dark
│  ├─underwater_imagenet
│  └─underwater_scenes
├─Unpaired
├─test_samples
└─eval_data
```

---

### underwater_dark

主要包含：

- 低照度退化
- 深海弱光场景

用于测试低照度增强能力。

---

### underwater_imagenet

主要包含：

- 蓝偏
- 绿偏
- 颜色衰减
- 对比度下降

用于测试颜色恢复能力。

---

### underwater_scenes

主要包含：

- 蓝偏
- 绿偏
- 低照度
- 模糊
- 混合退化

用于测试复杂场景泛化能力。

---

### 数据集用途

EUVP 不参与训练。

仅用于：

- 跨数据集测试
- 泛化能力验证
- 可视化效果比较

---

# Jaffe-McGlamery 水下成像模型

## 引入原因

UIEB 数据集虽然提供了真实退化图像及其参考图像，但存在以下问题：

- 数据规模有限
- 不同退化类型分布不均衡
- 退化程度难以控制
- 某些退化场景样本不足

因此引入 Jaffe-McGlamery 水下成像模型生成具有明确退化标签的训练样本。

---

## 水下成像原理

水下图像退化主要来源于：

### 光吸收（Absorption）

不同波长光衰减速度不同：

```text
红光衰减最快
绿光次之
蓝光最慢
```

导致：

```text
蓝偏现象
```

---

### 光散射（Scattering）

水体悬浮颗粒导致散射增强。

导致：

```text
绿偏现象
```

以及：

```text
图像模糊
细节损失
```

---

### 能量衰减

随着传播距离增加：

```text
光能量不断衰减
```

导致：

```text
低照度现象
```

---

## 成像模型

退化图像表示为：

```text
I(x)=J(x)t(x)+B(1−t(x))
```

其中：

```text
I(x) = 退化图像
J(x) = 清晰图像
t(x) = 透射率
B    = 背景光
```

通过调节透射率与背景光参数，可以生成不同类型的退化图像。

---

# 物理退化数据生成

## 输入数据

```text
UIEB/reference-890
```

作为清晰图像输入。

---

## 输出目录

```text
data/processed/physical_degradation/

├─blue_cast
├─green_cast
├─low_light
└─blur
```

---

## 蓝偏退化生成

退化机理：

```text
红光衰减 > 绿光衰减 > 蓝光衰减
```

生成过程：

```text
Reference Image
      ↓
Blue-Cast Simulation
      ↓
Blue-Cast Image
```

---

## 绿偏退化生成

退化机理：

```text
散射增强
      ↓
绿色通道占主导
```

生成过程：

```text
Reference Image
      ↓
Green-Cast Simulation
      ↓
Green-Cast Image
```

---

## 低照度退化生成

退化机理：

```text
整体能量衰减
      ↓
亮度下降
```

生成过程：

```text
Reference Image
      ↓
Low-Light Simulation
      ↓
Low-Light Image
```

---

## 模糊退化生成

采用：

- Gaussian Blur
- Motion Blur

生成过程：

```text
Reference Image
      ↓
Blur Simulation
      ↓
Blur Image
```

---

## 物理退化数据用途

主要用于：

1. 扩充训练数据规模。
2. 增加退化类型多样性。
3. 控制退化强度。
4. 训练对应增强分支。
5. 提升模型泛化能力。

# 图像退化分类

## 分类目的

UIEB 中的 raw-890 包含多种退化类型。

为了使不同增强分支能够学习对应退化模式，需要首先对 raw-890 中的图像进行退化分类。

分类结果用于：

1. 构建真实退化训练数据集。
2. 训练对应增强分支。
3. 构建退化标签。
4. Attention 融合权重学习。
5. 消融实验设计。

---

## 分类类别

所有图像按照以下类别进行划分：

```text
blue_cast
green_cast
low_light
blur
```

说明：

- 一张图像允许同时属于多个类别。
- 分类采用多标签方式而非单标签方式。

例如：

```text
img001

blue_cast = True
green_cast = False
low_light = True
blur = False
```

表示：

```text
蓝偏 + 低照度
```

混合退化场景。

---

## 分类结果保存

分类结果保存至：

```text
data/processed/UIEB_classified/

├─blue_cast
├─green_cast
├─low_light
└─blur
```

同时生成：

```text
classification_result.csv
```

记录：

```text
image_name
blue_cast
green_cast
low_light
blur
mean_a
mean_b
mean_v
laplacian_var
edge_density
```

---

## 蓝偏与绿偏检测

采用 LAB 颜色空间分析。

提取特征：

```text
Mean(a*)
Mean(b*)
Color Variance
```

判定规则：

```text
b*显著偏负
      ↓
Blue Cast

b*显著偏正
      ↓
Green Cast
```

---

## 低照度检测

采用 HSV 颜色空间。

提取：

```text
Mean(V)
```

判定规则：

```text
Mean(V) < Threshold
      ↓
Low-Light
```

---

## 模糊检测

采用清晰度分析。

提取：

```text
Laplacian Variance
Edge Density
```

判定规则：

```text
Laplacian Variance < Threshold
      ↓
Blur
```

---

# 分支预训练

## 设计思想

本项目采用两阶段训练策略。

第一阶段：

```text
分支预训练
```

第二阶段：

```text
Multi-Branch CycleGAN整体训练
```

预训练阶段的目标是使各增强分支首先掌握对应退化类型的恢复能力。

随后将预训练权重加载至整体网络进行联合优化。

---

## 分支训练数据来源

每个分支均使用：

```text
真实退化数据
+
物理退化数据
```

共同训练。

---

# Blue Branch 预训练

## 功能

负责恢复蓝偏图像。

---

## 输入数据

真实数据：

```text
UIEB_classified/blue_cast
```

物理数据：

```text
physical_degradation/blue_cast
```

---

## 监督目标

真实数据：

```text
blue_cast/raw_xxx
      ↕
reference_xxx
```

物理数据：

```text
blue_sim_xxx
      ↕
reference_xxx
```

---

## 网络结构

```text
Input
  ↓
White Balance
  ↓
VGG Feature Extractor
  ↓
Light CNN
  ↓
Output
```

---

## 学习目标

```text
Blue Cast
      ↓
Color Corrected Image
```

---

# Green Branch 预训练

## 功能

负责恢复绿偏图像。

---

## 输入数据

真实数据：

```text
UIEB_classified/green_cast
```

物理数据：

```text
physical_degradation/green_cast
```

---

## 监督目标

对应 reference 图像。

---

## 网络结构

```text
Input
  ↓
Inverse Physical Model
  ↓
CNN Refinement
  ↓
Output
```

---

## 学习目标

```text
Green Cast
      ↓
Color Corrected Image
```

---

# Low-Light Branch 预训练

## 功能

负责恢复低照度图像。

---

## 输入数据

真实数据：

```text
UIEB_classified/low_light
```

物理数据：

```text
physical_degradation/low_light
```

---

## 监督目标

对应 reference 图像。

---

## 网络结构

```text
Input
  ↓
Gamma Correction
  ↓
Enhancement CNN
  ↓
Output
```

---

## 学习目标

```text
Low-Light Image
      ↓
Brightness Enhanced Image
```

---

# Blur Branch 预训练

## 功能

负责恢复模糊图像。

---

## 输入数据

真实数据：

```text
UIEB_classified/blur
```

物理数据：

```text
physical_degradation/blur
```

---

## 监督目标

对应 reference 图像。

---

## 网络结构

```text
Input
  ↓
Deblur Kernel Estimation
  ↓
Restoration CNN
  ↓
Output
```

---

## 学习目标

```text
Blur Image
      ↓
Sharp Image
```

---

# 分支预训练损失函数

预训练阶段采用监督学习。

损失函数包括：

## L1 Loss

```text
L1(Output, Reference)
```

用于约束像素恢复误差。

---

## SSIM Loss

```text
SSIM(Output, Reference)
```

用于保持结构一致性。

---

## Perceptual Loss

采用 VGG 特征。

```text
VGG(Output)
      ↓
VGG(Reference)
```

用于提升视觉质量。

---

## 总损失

```text
Loss_pretrain

=
λ1 * L1
+
λ2 * SSIM
+
λ3 * Perceptual
```

---

# 预训练输出

训练完成后保存：

```text
checkpoints/pretrained_branches/

├─blue_branch.pth
├─green_branch.pth
├─lowlight_branch.pth
└─blur_branch.pth
```

这些权重将在整体 Multi-Branch CycleGAN 训练阶段加载。

---

# 第一阶段训练流程

```text
UIEB raw-890
      │
      ▼
退化分类
      │
      ▼
真实退化数据

UIEB reference-890
      │
      ▼
Jaffe-McGlamery
      │
      ▼
物理退化数据

真实退化数据
+
物理退化数据
      │
      ▼
Blue Branch 预训练

真实退化数据
+
物理退化数据
      │
      ▼
Green Branch 预训练

真实退化数据
+
物理退化数据
      │
      ▼
Low-Light Branch 预训练

真实退化数据
+
物理退化数据
      │
      ▼
Blur Branch 预训练

      │
      ▼

保存预训练权重
```

# Multi-Branch CycleGAN 网络设计

## 设计目标

经过第一阶段预训练后：

```text
Blue Branch
Green Branch
Low-Light Branch
Blur Branch
```

已经具备对应退化类型的专项恢复能力。

第二阶段的目标是：

```text
融合各分支能力
+
学习复杂混合退化场景
+
提升泛化能力
```

因此构建 Multi-Branch CycleGAN。

---

# 整体网络结构

```text
                    Input Image
                          │
                          ▼
                Feature Extraction
                          │
                          ▼
      ┌────────┬────────┬────────┬────────┐
      │        │        │        │
      ▼        ▼        ▼        ▼
 Blue Branch Green Branch Low-Light Branch Blur Branch
      │        │        │        │
      ▼        ▼        ▼        ▼
 BlueFeat GreenFeat LowFeat BlurFeat
      │        │        │        │
      └────────┴────────┴────────┴────────┘
                          │
                          ▼
                 Attention Fusion
                          │
                          ▼
                  Fusion Feature
                          │
                          ▼
                  Reconstruction
                          │
                          ▼
                  Enhanced Image
```

---

# Generator 结构

## G_AB

作用：

```text
退化域
      ↓
清晰域
```

即：

```text
Domain A
      ↓
Domain B
```

对应：

```text
Degraded Image
      ↓
Enhanced Image
```

---

## G_BA

作用：

```text
清晰域
      ↓
退化域
```

即：

```text
Domain B
      ↓
Domain A
```

对应：

```text
Reference Image
      ↓
Synthetic Degraded Image
```

用于构建 Cycle Consistency。

---

# Blue Branch

输入：

```text
Blue-Degraded Feature
```

核心模块：

```text
White Balance
+
VGG Feature Extraction
+
Light CNN
```

输出：

```text
Blue Restoration Feature
```

主要负责：

```text
颜色校正
```

---

# Green Branch

输入：

```text
Green-Degraded Feature
```

核心模块：

```text
Inverse Physical Model
+
CNN Refinement
```

输出：

```text
Green Restoration Feature
```

主要负责：

```text
绿色偏色恢复
```

---

# Low-Light Branch

输入：

```text
Low-Light Feature
```

核心模块：

```text
Gamma Correction
+
Enhancement CNN
```

输出：

```text
Brightness Restoration Feature
```

主要负责：

```text
亮度恢复
```

---

# Blur Branch

输入：

```text
Blur Feature
```

核心模块：

```text
Kernel Estimation
+
Deblur CNN
```

输出：

```text
Sharp Restoration Feature
```

主要负责：

```text
细节恢复
```

---

# Attention Fusion 模块

## 设计目的

真实水下图像通常同时包含多种退化：

例如：

```text
蓝偏
+
低照度
```

或者：

```text
绿偏
+
模糊
```

因此不能简单平均融合。

需要动态分配各分支权重。

---

## 输入

```text
Blue Feature

Green Feature

Low-Light Feature

Blur Feature
```

---

## 多头注意力机制

采用：

```text
Multi-Head Attention
```

构造：

```text
Q = Query
K = Key
V = Value
```

计算：

```text
Attention(Q,K,V)
=
Softmax(QK^T / √d)
× V
```

---

## 输出权重

例如：

```text
Blue Weight      = 0.50
Green Weight     = 0.10
Low-Light Weight = 0.30
Blur Weight      = 0.10
```

表示：

```text
当前图像主要受到蓝偏和低照度影响
```

---

## 融合结果

最终输出：

```text
Fusion Feature

=
w1 × BlueFeature
+
w2 × GreenFeature
+
w3 × LowFeature
+
w4 × BlurFeature
```

---

# Reconstruction Module

融合后特征经过重建网络：

```text
Fusion Feature
      ↓
Residual Blocks
      ↓
Conv Layers
      ↓
Enhanced Image
```

输出最终增强结果。

---

# 判别器设计

采用：

```text
PatchGAN
```

---

## D_A

判别：

```text
是否属于退化域
```

输入：

```text
Raw Image
Physical Degradation Image
Generated Degradation Image
```

输出：

```text
Real / Fake
```

---

## D_B

判别：

```text
是否属于清晰域
```

输入：

```text
Reference Image
Enhanced Image
```

输出：

```text
Real / Fake
```

---

# Domain 构建

## Domain A

退化域：

```text
UIEB/raw-890

+

physical_degradation/blue_cast

+

physical_degradation/green_cast

+

physical_degradation/low_light

+

physical_degradation/blur
```

---

## Domain B

清晰域：

```text
UIEB/reference-890
```

---

# 加载预训练权重

第二阶段训练开始时：

```text
Blue Branch
      ↓
blue_branch.pth

Green Branch
      ↓
green_branch.pth

Low-Light Branch
      ↓
lowlight_branch.pth

Blur Branch
      ↓
blur_branch.pth
```

加载至对应模块。

---

# 第二阶段训练流程

```text
加载预训练权重
        │
        ▼

构建 Multi-Branch Generator

        │
        ▼

构建 D_A

        │
        ▼

构建 D_B

        │
        ▼

Domain A
+
Domain B

        │
        ▼

CycleGAN Training

        │
        ▼

Attention Fusion Learning

        │
        ▼

Enhanced Image
```

# 损失函数设计

## 设计目标

本项目采用监督预训练与 CycleGAN 联合训练相结合的方式。

损失函数需要同时保证：

- 图像真实性
- 内容一致性
- 结构保持
- 感知质量
- 颜色稳定性

---

# Generator Loss

Generator 总损失定义为：

```text
L_G

=
λ_adv * L_adv
+
λ_cyc * L_cycle
+
λ_id * L_identity
+
λ_ssim * L_ssim
+
λ_per * L_perceptual
```

---

## Adversarial Loss

### G_AB

目标：

```text
G_AB(A)
```

能够欺骗：

```text
D_B
```

损失：

```text
L_adv_AB

=
E[(D_B(G_AB(A)) - 1)^2]
```

---

### G_BA

目标：

```text
G_BA(B)
```

能够欺骗：

```text
D_A
```

损失：

```text
L_adv_BA

=
E[(D_A(G_BA(B)) - 1)^2]
```

---

## Cycle Consistency Loss

保证图像内容一致。

### Forward Cycle

```text
A
 ↓
G_AB
 ↓
B
 ↓
G_BA
 ↓
A'
```

损失：

```text
||A - A'||
```

---

### Backward Cycle

```text
B
 ↓
G_BA
 ↓
A
 ↓
G_AB
 ↓
B'
```

损失：

```text
||B - B'||
```

---

总损失：

```text
L_cycle

=
||A - A'||
+
||B - B'||
```

---

## Identity Loss

目的：

保持颜色稳定。

```text
B
 ↓
G_AB
 ↓
B'
```

希望：

```text
B ≈ B'
```

损失：

```text
L_identity

=
||B - G_AB(B)||
```

---

## SSIM Loss

保持图像结构一致。

```text
L_ssim

=
1 - SSIM(Output, Target)
```

---

## Perceptual Loss

采用 VGG19 特征提取。

```text
VGG(Output)
      ↓
VGG(Target)
```

损失：

```text
L_perceptual

=
||Φ(Output)-Φ(Target)||
```

---

# Discriminator Loss

采用 Least Squares GAN。

---

## D_A

```text
L_DA

=
(D_A(real)-1)^2

+
(D_A(fake))^2
```

---

## D_B

```text
L_DB

=
(D_B(real)-1)^2

+
(D_B(fake))^2
```

---

# 训练流程

## 第一阶段

分支预训练

---

### 输入

真实退化数据：

```text
UIEB_classified/*
```

物理退化数据：

```text
physical_degradation/*
```

---

### 输出

```text
blue_branch.pth
green_branch.pth
lowlight_branch.pth
blur_branch.pth
```

---

## 第二阶段

整体 Multi-Branch CycleGAN 训练

---

### Step1

加载分支权重：

```text
blue_branch.pth

green_branch.pth

lowlight_branch.pth

blur_branch.pth
```

---

### Step2

构建：

```text
G_AB

G_BA

D_A

D_B
```

---

### Step3

构建训练域

#### Domain A

```text
UIEB/raw-890

+

physical_degradation/*
```

---

#### Domain B

```text
UIEB/reference-890
```

---

### Step4

训练流程

```text
Input A
      ↓

G_AB
      ↓

Enhanced Image
      ↓

D_B

------------------

Input B
      ↓

G_BA
      ↓

Degraded Image
      ↓

D_A

------------------

Cycle Consistency

------------------

Backpropagation
```

---

### Step5

保存模型

```text
checkpoints/

├─generator/
├─discriminator/
└─best_model/
```

---

# 测试流程

## UIEB测试

测试数据：

```text
UIEB/raw-890
```

目标：

```text
reference-890
```

计算：

```text
PSNR
SSIM
UIQM
UCIQE
```

---

## Challenging-60测试

测试数据：

```text
challenging-60
```

无参考图像。

计算：

```text
UIQM
UCIQE
```

并生成：

```text
Visual Comparison
```

---

## EUVP测试

测试数据：

```text
underwater_dark

underwater_imagenet

underwater_scenes

Unpaired

test_samples

eval_data
```

---

### 有参考图像

计算：

```text
PSNR
SSIM
UIQM
UCIQE
```

---

### 无参考图像

计算：

```text
UIQM
UCIQE
```

---

# 输出结果

增强结果保存：

```text
outputs/test_results/
```

结构：

```text
outputs/

├─UIEB
├─challenging60
├─EUVP
└─visual_comparison
```

---

# 评价指标

## PSNR

评价：

```text
像素级恢复质量
```

数值越大越好。

---

## SSIM

评价：

```text
结构保持能力
```

范围：

```text
0 ~ 1
```

越接近：

```text
1
```

越好。

---

## UIQM

评价：

```text
水下图像视觉质量
```

包含：

- 色彩质量
- 清晰度
- 对比度

---

## UCIQE

评价：

```text
水下图像颜色质量
```

包含：

- 色度
- 饱和度
- 对比度

# 消融实验设计

## 实验目的

验证以下模块对最终增强性能的贡献：

1. 多分支结构有效性
2. Attention Fusion有效性
3. 物理退化建模有效性
4. 分支预训练有效性
5. 各增强分支贡献

---

# 实验一：单分支 CycleGAN

## 配置

去除多分支结构。

采用：

```text
普通 CycleGAN
```

结构：

```text
Input
  ↓
Generator
  ↓
Output
```

---

## 目的

验证：

```text
Multi-Branch
```

是否优于：

```text
Single Generator
```

---

# 实验二：去除物理退化建模

## 配置

训练数据仅使用：

```text
UIEB/raw-890
```

不使用：

```text
physical_degradation/*
```

---

## 目的

验证：

```text
Jaffe-McGlamery
```

生成的退化数据是否能够提升模型性能。

---

# 实验三：去除 Attention Fusion

## 配置

采用：

```text
Feature Concatenation
```

替代：

```text
Attention Fusion
```

---

## 融合方式

```text
Blue Feature

Green Feature

Low-Light Feature

Blur Feature

        ↓

Concat

        ↓

Conv

        ↓

Output
```

---

## 目的

验证：

```text
Attention
```

对于混合退化场景的重要性。

---

# 实验四：去除蓝偏分支

## 配置

移除：

```text
Blue Branch
```

---

## 目的

验证：

```text
Blue Branch
```

对颜色恢复能力的贡献。

---

# 实验五：去除绿偏分支

## 配置

移除：

```text
Green Branch
```

---

## 目的

验证：

```text
Green Branch
```

对颜色恢复能力的贡献。

---

# 实验六：去除低照度分支

## 配置

移除：

```text
Low-Light Branch
```

---

## 目的

验证：

```text
Low-Light Branch
```

对亮度恢复能力的贡献。

---

# 实验七：去除模糊分支

## 配置

移除：

```text
Blur Branch
```

---

## 目的

验证：

```text
Blur Branch
```

对细节恢复能力的贡献。

---

# 实验评价指标

所有实验均统计：

```text
PSNR

SSIM

UIQM

UCIQE
```

并绘制：

```text
Metric Comparison Chart
```

用于分析不同模块贡献。

---

# 对比实验

## 对比模型

至少包含：

```text
CycleGAN

UGAN

FUnIE-GAN

UWCNN

WaterNet

Proposed Method
```

---

## 对比内容

### 定量评价

统计：

```text
PSNR

SSIM

UIQM

UCIQE
```

---

### 定性评价

展示：

```text
Input

Reference

CycleGAN

UGAN

FUnIE-GAN

Proposed
```

增强效果对比图。

---

# 项目目录结构

```text
project/

├─data/
│
├─datasets/
│
├─models/
│   ├─branches/
│   │
│   ├─generator/
│   │
│   ├─discriminator/
│   │
│   └─attention/
│
├─metrics/
│
├─utils/
│
├─scripts/
│
├─checkpoints/
│   ├─pretrained_branches/
│   ├─generator/
│   └─discriminator/
│
├─outputs/
│
├─results/
│
├─logs/
│
├─pipeline.md
│
├─README.md
│
├─train_branch.py
│
├─train_cyclegan.py
│
├─test.py
│
└─requirements.txt
```

---

# 训练脚本说明

## train_branch.py

负责：

```text
退化分类

↓

分支预训练
```

输出：

```text
blue_branch.pth

green_branch.pth

lowlight_branch.pth

blur_branch.pth
```

---

## train_cyclegan.py

负责：

```text
加载预训练分支

↓

构建 Multi-Branch CycleGAN

↓

整体训练
```

输出：

```text
generator_best.pth

discriminator_best.pth
```

---

## test.py

负责：

```text
UIEB测试

EUVP测试

指标统计

结果保存
```

输出：

```text
evaluation_metrics.csv

average_metrics.csv
```

---

# README 内容要求

README 至少包含：

## 项目简介

介绍：

```text
Multi-Branch CycleGAN

+

Jaffe-McGlamery

+

Attention Fusion
```

---

## 数据集准备

说明：

```text
UIEB

EUVP
```

下载与目录组织方式。

---

## 数据预处理

说明：

```text
退化分类

物理退化生成
```

执行方法。

---

## 分支预训练

运行：

```bash
python train_branch.py
```

---

## 整体训练

运行：

```bash
python train_cyclegan.py
```

---

## 模型测试

运行：

```bash
python test.py
```

---

## 实验结果

展示：

```text
PSNR

SSIM

UIQM

UCIQE
```

以及增强效果图。

---

# Codex 开发规范

## 基本原则

优先复用现有代码。

禁止：

```text
推翻重写整个项目
```

应当：

```text
增量开发
```

---

## 推荐依赖

```text
Python

PyTorch

Torchvision

OpenCV

NumPy

Pandas

Matplotlib

scikit-image

tqdm
```

---

## 实现顺序

```text
Step1
退化分类

↓

Step2
物理退化生成

↓

Step3
分支预训练

↓

Step4
Attention Fusion

↓

Step5
Multi-Branch CycleGAN

↓

Step6
测试与评价

↓

Step7
消融实验
```

---

# 项目总体流程

```text
UIEB/raw-890
        │
        ▼
退化分类
        │
        ▼
真实退化数据

UIEB/reference-890
        │
        ▼
Jaffe-McGlamery
        │
        ▼
物理退化数据

真实退化数据
+
物理退化数据
        │
        ▼
四个专家分支预训练

Blue Branch

Green Branch

Low-Light Branch

Blur Branch

        │
        ▼
保存预训练权重

        │
        ▼
构建 Multi-Branch Generator

        │
        ▼
Attention Fusion

        │
        ▼
CycleGAN 联合训练

Domain A:
UIEB raw + Physical Degradation

Domain B:
UIEB reference

        │
        ▼
增强结果

        │
        ▼
UIEB测试

        │
        ▼
EUVP测试

        │
        ▼
评价指标统计

        │
        ▼
消融实验分析
```