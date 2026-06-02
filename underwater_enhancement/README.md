# 水下图像增强项目

本项目是一个基于 Python + PyTorch 的水下图像增强论文工程实现，包含图像统计特征分析、自动分类、Jaffe-McGlamery 物理退化建模、多分支 CycleGAN 训练与测试评估。

## 环境安装

在项目根目录执行：

```bash
cd underwater_enhancement
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果当前环境中的 `torch` 和 `torchvision` 版本不匹配，模型仍可运行：代码会在 VGG16 预训练模型不可用时自动退回到轻量 CNN 特征提取模块。

## 数据目录结构

建议按以下结构放置数据：

```text
underwater_enhancement/
├── data/
│   ├── raw_underwater/      # 原始水下图像，用于统计分析、分类和退化生成
│   ├── clean_images/        # 清晰图像域，用于 CycleGAN 的 B 域训练
│   ├── test_underwater/     # 测试阶段输入的水下图像
│   └── test_clean/          # 可选，测试阶段对应参考图，用于计算 PSNR/SSIM
└── outputs/
```

所有路径都在 `config.yaml` 中配置，不需要在代码中硬编码路径。

## 主要命令

请在项目根目录运行：

```bash
python main.py --config config.yaml analyze
python main.py --config config.yaml classify
python main.py --config config.yaml degrade
python main.py --config config.yaml train
python main.py --config config.yaml test
```

如果 `data/raw_underwater/` 中放的是 `.zip`、`.rar` 等压缩包，可以先执行：

```bash
python main.py --config config.yaml prepare-data
```

也可以直接执行 `analyze` 或 `degrade`，程序会在读取图像前自动解压。

下载 UIEB 和 EUVP 数据集：

```bash
python tools/download_datasets.py --datasets uieb euvp --output-root data/external
```

各模块也可以单独运行：

```bash
python analysis/feature_extraction.py --config config.yaml
python analysis/classify_images.py --config config.yaml
python degradation/generate_degraded_dataset.py --config config.yaml
python train/train_cyclegan.py --config config.yaml
python eval/test.py --config config.yaml
```

## 功能说明

### 0. 数据集下载

执行：

```bash
python tools/download_datasets.py --datasets uieb euvp --output-root data/external
```

脚本会下载：

- UIEB：官方 Google Drive 中的 raw 图像和 reference 图像；
- EUVP：IRVLab 官方页面提供的 Google Drive 文件夹。

默认输出：

```text
data/external/UIEB/
data/external/UIEB/archives/
data/external/UIEB/raw/
data/external/UIEB/reference/
data/external/EUVP/
```

只下载不解压：

```bash
python tools/download_datasets.py --datasets uieb --output-root data/external --no-extract
```

注意：

- UIEB 官方页面声明数据集仅限学术/非商业用途，并禁止二次分发；
- Google Drive 链接可能因权限、确认页或流量限制下载失败，此时需要先在浏览器打开官方页面确认访问权限；
- EUVP 是 Google Drive 文件夹下载，建议安装最新版 `gdown`。

### 压缩包解压

如果已经手动把数据集压缩包放入 `data/raw_underwater/`，例如：

```text
data/raw_underwater/UIEB.rar
data/raw_underwater/EUVP.zip
```

执行：

```bash
python main.py --config config.yaml prepare-data
```

解压结果会保存到：

```text
data/raw_underwater/_extracted_archives/UIEB/
data/raw_underwater/_extracted_archives/EUVP/
```

后续 `analyze`、`classify`、`degrade` 仍然按原逻辑递归读取 `data/raw_underwater/` 下的图像文件，不需要额外改路径。

压缩包相关配置在 `config.yaml` 中：

```yaml
archive_extraction:
  enabled: true
  extract_dir_name: _extracted_archives
  overwrite: false
  passwords: ["", "1234567", "8901234", "5678901"]
```

说明：

- `.zip` 文件使用 Python 标准库解压；
- `.rar` 文件需要系统中安装 7-Zip、unrar，或安装 `rarfile` 并配置可用的 RAR 后端；
- 已解压成功的压缩包会生成 `.extracted` 标记，默认不会重复解压；
- 如果要强制重新解压，把 `overwrite` 改为 `true`，或删除对应的 `.extracted` 标记文件。

### 1. 图像统计特征分析

执行：

```bash
python main.py --config config.yaml analyze
```

输出内容：

```text
outputs/analysis/image_features.csv
outputs/analysis/image_features.xlsx
outputs/analysis/rgb_histograms/
outputs/analysis/brightness_histograms/
outputs/analysis/edges/
```

提取的特征包括：

- RGB 三通道均值、标准差和直方图；
- LAB 空间中 L、a、b 通道统计特征；
- HSV 空间 V 通道亮度均值；
- 低亮度像素比例；
- Canny 边缘数量和边缘密度；
- Laplacian 方差；
- 清晰度、亮度、颜色偏移等指标。

### 2. 图像自动分类

执行：

```bash
python main.py --config config.yaml classify
```

输出内容：

```text
outputs/classified_images/classification_results.csv
outputs/classified_images/classification_results.xlsx
outputs/classified_images/color_distortion_blue/
outputs/classified_images/color_distortion_green/
outputs/classified_images/low_light/
outputs/classified_images/blurry/
outputs/classified_images/normal/
```

分类类别包括：

- `color_distortion_blue`
- `color_distortion_green`
- `low_light`
- `blurry`
- `normal`

支持一张图像被复制到多个类别文件夹。分类阈值在 `config.yaml` 的 `classification` 字段中配置，分类结果中会保存可解释原因，例如 `blue_mean`、`green_mean`、`brightness`、`laplacian_var` 等。

### 3. Jaffe-McGlamery 退化建模

执行：

```bash
python main.py --config config.yaml degrade
```

退化模型形式：

```text
I_c(x) = J_c(x) * T_c(x) + B_c * (1 - T_c(x))
```

支持的退化类型：

- `blue_shift`
- `green_shift`
- `low_light`
- `blur`

输出内容：

```text
outputs/degraded_images/{blue_shift,green_shift,low_light,blur}/
outputs/degraded_images/degradation_params.csv
outputs/degraded_images/*/depth/
outputs/degraded_images/*/comparisons/
```

每张图像会生成伪深度图，支持随机深度、随机背景光、随机退化强度，并可通过 `config.yaml` 配置每张图像生成的退化数量。

### 4. 多分支 CycleGAN 训练

执行：

```bash
python main.py --config config.yaml train
```

模型包含：

- `G_AB`：水下/退化图像到清晰图像；
- `G_BA`：清晰图像到退化图像；
- `D_A`：A 域 PatchGAN 判别器；
- `D_B`：B 域 PatchGAN 判别器。

Generator 使用四分支结构：

- `BlueCastBranch`：白平衡 + VGG/轻量特征提取 + CNN；
- `GreenCastBranch`：颜色通道补偿；
- `LowLightBranch`：Gamma 校正 + CNN 增强；
- `BlurBranch`：卷积去模糊分支。

四个分支通过 attention fusion 融合后输出增强图像。

总损失函数：

```text
total_loss = lambda_adv * L_adv
           + lambda_cycle * L_cycle
           + lambda_identity * L_identity
           + lambda_ssim * L_ssim
```

训练输出：

```text
outputs/checkpoints/latest.pth
outputs/checkpoints/epoch_*.pth
outputs/samples/
outputs/logs/training_log.csv
outputs/logs/train.log
```

断点续训方式：

在 `config.yaml` 中设置：

```yaml
training:
  resume: outputs/checkpoints/latest.pth
```

然后重新执行训练命令。

### 5. 测试与评估

执行：

```bash
python main.py --config config.yaml test
```

测试会读取 `config.yaml` 中 `testing.input_dir` 的图像，并使用 `testing.checkpoint` 指定的模型；如果未指定，则默认读取 `outputs/checkpoints/latest.pth`。

输出内容：

```text
outputs/test_enhanced/
outputs/test_enhanced/comparisons/
outputs/test_enhanced/evaluation_metrics.csv
```

评估指标包括：

- PSNR：需要存在参考图；
- SSIM：需要存在参考图；
- UIQM：无参考水下图像质量指标；
- UCIQE：无参考水下图像质量指标。

## 配置说明

常用配置在 `config.yaml` 中修改：

```yaml
paths:
  input_dir: data/raw_underwater
  clean_dir: data/clean_images
  degraded_dir: outputs/degraded_images

training:
  image_size: 256
  batch_size: 2
  epochs: 100
  lr: 0.0002
  checkpoint_dir: outputs/checkpoints

classification:
  blue_mean_threshold: 115
  brightness_threshold: 75
  laplacian_var_threshold: 80
```

## 注意事项

- OpenCV 读取图像后会立即从 BGR 转为 RGB。
- 所有图像张量统一归一化到 `[0, 1]`。
- 保存图像前会转换回 `uint8`，并从 RGB 转回 BGR 写入文件。
- 路径、训练参数、分类阈值和退化参数都应通过 `config.yaml` 配置。
- 训练数据为非配对 CycleGAN 数据：`degraded_images` 作为 A 域，`clean_images` 作为 B 域。
