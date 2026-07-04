# SOTA 对比实验

本目录用于在同一份 UIEB 数据划分上对比以下水下图像增强模型：

- CycleGAN
- UGAN
- FUnIE-GAN
- UWCNN
- WaterNet


## 默认统一配置

默认配置尽量与当前实验方案保持一致，并且都可以通过命令行参数调整：

- 数据划分：`train/validation/test = 0.7/0.15/0.15`
- 图像尺寸：`256`
- batch size：`2`
- 随机种子：`42`
- 优化器：Adam
- Adam betas：`(0.5, 0.999)`
- 学习率：`2e-4`
- epoch 数：`30`
- 评价指标：`PSNR`、`SSIM`、`UIQM`、`UCIQE`

不同模型保留各自训练范式：

- CycleGAN：非配对 CycleGAN 训练
- UGAN：配对条件 GAN 训练
- FUnIE-GAN：配对条件 GAN 训练
- UWCNN：配对监督训练
- WaterNet：配对监督训练

## 输出目录

默认输出在：

```bash
experiments/sota_benchmarks/workdir
```

每个模型的结果会分别保存到：

```bash
experiments/sota_benchmarks/workdir/{model_name}/checkpoints
experiments/sota_benchmarks/workdir/{model_name}/test_results
```

总汇总表保存到：

```bash
experiments/sota_benchmarks/workdir/summary.csv
```

## 推荐运行方式

先准备统一数据划分：

```bash
python -m experiments.sota_benchmarks.benchmark prepare
```

训练单个模型：

```bash
python -m experiments.sota_benchmarks.benchmark train --model funie-gan
```

测试单个模型：

```bash
python -m experiments.sota_benchmarks.benchmark test --model funie-gan
```

可选模型名：

```bash
cyclegan
ugan
funie-gan
uwcnn
waternet
```

一键训练并测试所有 SOTA 模型，同时生成 `summary.csv`：

```bash
python -m experiments.sota_benchmarks.benchmark run-all
```

## 常用参数

指定训练轮数：

```bash
python -m experiments.sota_benchmarks.benchmark train --model waternet --epochs 80
```

指定学习率和 batch size：

```bash
python -m experiments.sota_benchmarks.benchmark train --model uwcnn --lr 2e-4 --batch-size 2
```

指定 GPU：

```bash
python -m experiments.sota_benchmarks.benchmark run-all --device cuda
```

使用 CPU：

```bash
python -m experiments.sota_benchmarks.benchmark run-all --device cpu
```

重新生成数据划分：

```bash
python -m experiments.sota_benchmarks.benchmark prepare --overwrite
```

## 快速可行性测试

下面命令只训练每个模型 1 个 batch、验证 1 个 batch、测试 1 张图片，用于检查代码是否能跑通：

```bash
python -m experiments.sota_benchmarks.benchmark run-all --epochs 1 --max-train-batches 1 --val-max-batches 1 --test-max-images 1 --image-size 64 --batch-size 1 --device cpu
```

该命令只用于 smoke test，不代表正式实验结果。

## 正式实验建议

正式对比实验建议使用默认 `image-size 256`，并根据显存调整 `batch-size`。例如：

```bash
python -m experiments.sota_benchmarks.benchmark run-all --device cuda --epochs 30 --batch-size 2 --image-size 256
```

运行完成后查看：

```bash
experiments/sota_benchmarks/workdir/summary.csv
```
