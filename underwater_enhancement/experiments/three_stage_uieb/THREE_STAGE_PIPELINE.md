# Three-Stage UIEB Training Pipeline

This document describes the added experimental pipeline. It does not replace the project README.

## 1. Complete Three-Stage Training Flow

1. Split UIEB paired samples by image identity.
   - Source pairs: `UIEB/raw-890` and `UIEB/reference-890`.
   - Splits: `train`, `validation`, `test`.
   - The same image identity cannot appear in more than one split.

2. Generate train-only synthetic degraded images.
   - Source: `train/reference`.
   - Output: `synthetic_degraded/train/{blue_cast,green_cast,low_light,blur}`.
   - Validation and test references are never used for synthetic degradation.

3. Stage 1: supervised paired enhancement.
   - Input: `train/raw`.
   - Target: `train/reference`.
   - No physical degradation is used.
   - Objective: learn basic raw-to-reference enhancement ability.

4. Stage 2: unpaired CycleGAN training.
   - Clean domain: `train/reference`.
   - Degraded domain: `train/raw + train synthetic degraded`.
   - Synthetic degraded images are domain augmentation only.
   - No synthetic degraded image is used with its source reference as a supervised pair.

5. Stage 3: fine-tuning.
   - Same mixed unpaired domains as Stage 2.
   - Smaller learning rate.
   - Partial unfreezing of CNN branch parameters.

6. Final test.
   - Input: `test/raw`.
   - Target: `test/reference`.
   - The test split is held out from all training and synthetic generation.

## 2. Data Inputs and Outputs

### Split Preparation

Input:

```text
data/raw_underwater/UIEB/raw-890
data/raw_underwater/UIEB/reference-890
```

Output:

```text
experiments/three_stage_uieb/workdir/splits/train/raw
experiments/three_stage_uieb/workdir/splits/train/reference
experiments/three_stage_uieb/workdir/splits/validation/raw
experiments/three_stage_uieb/workdir/splits/validation/reference
experiments/three_stage_uieb/workdir/splits/test/raw
experiments/three_stage_uieb/workdir/splits/test/reference
experiments/three_stage_uieb/workdir/splits/split_manifest.csv
```

### Stage 1

Input:

```text
train/raw -> train/reference
validation/raw -> validation/reference
```

Output:

```text
workdir/checkpoints/stage1/stage1_best.pth
workdir/checkpoints/stage1/latest.pth
workdir/logs/stage1_train.csv
workdir/logs/stage1_validation.csv
workdir/samples/stage1
```

### Synthetic Degraded Generation

Input:

```text
workdir/splits/train/reference
```

Output:

```text
workdir/synthetic_degraded/train/blue_cast
workdir/synthetic_degraded/train/green_cast
workdir/synthetic_degraded/train/low_light
workdir/synthetic_degraded/train/blur
workdir/synthetic_degraded/train_mapping.csv
```

### Stage 2

Input:

```text
Degraded domain:
  workdir/splits/train/raw
  workdir/synthetic_degraded/train/*

Clean domain:
  workdir/splits/train/reference
```

Output:

```text
workdir/checkpoints/stage2/stage2_best.pth
workdir/checkpoints/stage2/latest.pth
workdir/logs/stage2_train.csv
workdir/logs/stage2_validation.csv
workdir/samples/stage2
```

### Stage 3

Input:

```text
Degraded domain:
  workdir/splits/train/raw
  workdir/synthetic_degraded/train/*

Clean domain:
  workdir/splits/train/reference
```

Output:

```text
workdir/checkpoints/stage3/stage3_best.pth
workdir/checkpoints/stage3/latest.pth
workdir/logs/stage3_train.csv
workdir/logs/stage3_validation.csv
workdir/samples/stage3
```

### Test

Input:

```text
workdir/splits/test/raw
workdir/splits/test/reference
workdir/checkpoints/stage3/stage3_best.pth
```

Output:

```text
workdir/test_results/images
workdir/test_results/comparisons
workdir/test_results/metrics.csv
workdir/test_results/average_metrics.csv
workdir/test_results/attention.csv
```

## 3. Degraded Domain Construction

The degraded domain for Stage 2 and Stage 3 is:

```text
Degraded domain = real UIEB train raw images + synthetic degraded train images
```

The synthetic degraded images come only from:

```text
workdir/splits/train/reference
```

They are generated with the existing physical degradation code in `scripts/generate_physical_degradation.py`, using:

```text
blue_cast
green_cast
low_light
blur
```

The synthetic degraded images are used only to augment the unpaired degraded domain. They are not paired with their source reference in any supervised loss.

## 4. Training Objectives and Losses

### Stage 1: Supervised Learning

Objective:

```text
G_AB(raw_train) -> reference_train
```

Loss:

```text
L_stage1 =
  lambda_l1 * L1(output, reference)
+ lambda_ssim * SSIMLoss(output, reference)
+ lambda_perceptual * PerceptualLoss(output, reference)
```

### Stage 2: Unpaired CycleGAN

Objective:

```text
G_AB: degraded domain -> clean domain
G_BA: clean domain -> degraded domain
```

Loss:

```text
L_G =
  lambda_adv * L_adv
+ lambda_cycle * L_cycle
+ lambda_identity * L_identity
+ lambda_ssim * L_cycle_ssim
```

Definitions:

```text
L_adv:
  D_B(G_AB(A)) should be real clean
  D_A(G_BA(B)) should be real degraded

L_cycle:
  ||G_BA(G_AB(A)) - A||_1
+ ||G_AB(G_BA(B)) - B||_1

L_identity:
  ||G_AB(B) - B||_1

L_cycle_ssim:
  SSIMLoss(G_BA(G_AB(A)), A)
+ SSIMLoss(G_AB(G_BA(B)), B)
```

### Stage 3: Fine-Tuning

Objective:

```text
Continue unpaired mixed-domain CycleGAN training with smaller updates.
```

Loss:

```text
Same as Stage 2.
```

Training policy:

```text
small learning rate
partial unfreezing of CNN branches
fixed feature extractors stay frozen
```

## 5. Pipeline Logic Diagram

```text
UIEB raw-890 + UIEB reference-890
        |
        v
paired split by image identity
        |
        +--> train/raw + train/reference
        |         |
        |         +--> Stage 1 supervised G_AB pretraining
        |         |
        |         +--> train/reference
        |                  |
        |                  v
        |           physical degradation
        |                  |
        |                  v
        |           train synthetic degraded
        |                  |
        |                  v
        |     Degraded domain = train/raw + train synthetic degraded
        |     Clean domain    = train/reference
        |                  |
        |                  v
        |           Stage 2 unpaired CycleGAN
        |                  |
        |                  v
        |           Stage 3 fine-tuning
        |
        +--> validation/raw + validation/reference
        |         |
        |         v
        |   validation-only checkpoint selection
        |
        +--> test/raw + test/reference
                  |
                  v
          final held-out evaluation
```

## Commands

Prepare split and synthetic degraded data:

```bash
python experiments/three_stage_uieb/train_three_stage.py prepare
```

Run all training stages:

```bash
python experiments/three_stage_uieb/train_three_stage.py train
```

Run preparation and training:

```bash
python experiments/three_stage_uieb/train_three_stage.py all
```

Test:

```bash
python experiments/three_stage_uieb/test_three_stage.py --checkpoint experiments/three_stage_uieb/workdir/checkpoints/stage3/stage3_best.pth
```

External EUVP test:

```bash
python experiments/three_stage_uieb/test_euvp.py --checkpoint experiments/three_stage_uieb/workdir/checkpoints/stage3/stage3_best.pth
```

Run both UIEB held-out split test and EUVP external test:

```bash
bash experiments/three_stage_uieb/run_tests.sh
```

Optional overrides:

```bash
CHECKPOINT=/path/to/checkpoint.pth DEVICE=cuda:0 bash experiments/three_stage_uieb/run_tests.sh
```
