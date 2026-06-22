#!/bin/bash

set -e

python test.py \
    --checkpoint checkpoints/best_model/no_blue/generator_best.pth \
    --metrics-csv results/ablation/no_blue/evaluation_metrics.csv \
    --average-csv results/ablation/no_blue/average_metrics.csv \
    --attention-csv results/ablation/no_blue/attention_statistics.csv

python test.py \
    --checkpoint checkpoints/best_model/no_blur/generator_best.pth \
    --metrics-csv results/ablation/no_blur/evaluation_metrics.csv \
    --average-csv results/ablation/no_blur/average_metrics.csv \
    --attention-csv results/ablation/no_blur/attention_statistics.csv

python test.py \
    --checkpoint checkpoints/best_model/no_green/generator_best.pth \
    --metrics-csv results/ablation/no_green/evaluation_metrics.csv \
    --average-csv results/ablation/no_green/average_metrics.csv \
    --attention-csv results/ablation/no_green/attention_statistics.csv

python test.py \
    --checkpoint checkpoints/best_model/no_lowlight/generator_best.pth \
    --metrics-csv results/ablation/no_lowlight/evaluation_metrics.csv \
    --average-csv results/ablation/no_lowlight/average_metrics.csv \
    --attention-csv results/ablation/no_lowlight/attention_statistics.csv


echo "========================================"
echo "All branch ablation experiments finished."
echo "========================================"