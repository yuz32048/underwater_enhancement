# 消融实验

本目录用于运行当前方法的核心消融实验。消融入口统一放在：

```bash
experiments/ablation_uieb/run_ablations.py
```

## 当前消融项

| ID | 实验名 | 说明 |
|---|---|---|
| A0 | full_model | 完整模型，使用 concat fusion |
| A1 | wo_synthetic_degradation | 去掉 Stage 2/3 中的 synthetic degraded domain augmentation |
| A2 | wo_branch_expert_pretrain | 去掉分支专家预训练 |
| A3 | wo_multibranch_experts | 去掉多分支退化专家，改用 single generator |
| A4 | wo_blue_branch | 去掉 blue-cast branch |
| A5 | wo_green_branch | 去掉 green-cast branch |
| A6 | wo_lowlight_branch | 去掉 low-light branch |
| A7 | wo_blur_branch | 去掉 blur branch |

暂不包含 Stage3 微调消融、三阶段训练消融、fusion 消融、loss 消融、synthetic ratio 消融。

## 查看消融列表

```bash
python -m experiments.ablation_uieb.run_ablations list
```

## 运行全部消融

```bash
python -m experiments.ablation_uieb.run_ablations run --device cuda --fusion concat
```

默认输出目录：

```bash
experiments/ablation_uieb/workdir
```

每个消融实验会有独立目录：

```bash
experiments/ablation_uieb/workdir/A0_full_model
experiments/ablation_uieb/workdir/A1_wo_synthetic_degradation
...
```

## 运行部分消融

只运行完整模型和去掉 synthetic degradation：

```bash
python -m experiments.ablation_uieb.run_ablations run --ablations A0,A1 --device cuda --fusion concat
```

只运行分支消融：

```bash
python -m experiments.ablation_uieb.run_ablations run --ablations A4,A5,A6,A7 --device cuda --fusion concat
```

## 快速可行性测试

下面命令只用于检查入口、训练、测试、汇总是否能跑通：

```bash
python -m experiments.ablation_uieb.run_ablations run --ablations A3 --stage1-epochs 1 --stage2-epochs 0 --stage3-epochs 0 --image-size 64 --batch-size 1 --test-max-images 1 --device cpu --overwrite
```

## 汇总结果

训练和测试完成后会自动生成：

```bash
experiments/ablation_uieb/workdir/summary.csv
```

也可以单独重新汇总：

```bash
python -m experiments.ablation_uieb.run_ablations summary
```
