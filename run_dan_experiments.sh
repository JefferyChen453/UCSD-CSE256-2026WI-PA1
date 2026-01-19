#!/bin/bash

# DAN Model Comparison Experiments Script
# This script runs multiple experiments to compare different configurations
# 
# Experiments cover:
# - Different number of layers (0, 1, 2, 3)
# - Different hidden dimensions (50, 100, 200)
# - Different embedding dimensions (50d, 300d)
# - With/without dropout
# - With/without cosine scheduler
# - Different weight decay values
# - Freeze/unfreeze embeddings
# - Train/freeze UNK token

set -e  # Exit on error

# Base configuration
EPOCHS=100
LR=0.0001
BATCH_SIZE=16
WANDB_PROJECT="CSE256_PA1_DAN_relu"

echo "Starting DAN comparison experiments..."
echo "Total experiments: 15"
echo "=========================================="

# Experiment 1: Baseline - 2 layers, 100 hidden dim, 300d embeddings
echo "[1/15] Experiment 1: Baseline (2 layers, hidden_dim=100, emb_dim=300)"
echo "----------------------------------------"
python main.py \
    --model DAN \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --emb_dim 300 \
    --num_hidden_layers 2 \
    --hidden_dim 100 \
    --dropout False \
    --freeze_embedding True \
    --train_unk_token False \
    --wandb \
    --wandb_project $WANDB_PROJECT

# Experiment 2: Different number of layers - 0 layers
echo "[2/15] Experiment 2: 0 hidden layers"
echo "----------------------------------------"
python main.py \
    --model DAN \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --emb_dim 300 \
    --num_hidden_layers 0 \
    --hidden_dim 100 \
    --dropout False \
    --freeze_embedding True \
    --train_unk_token False \
    --wandb \
    --wandb_project $WANDB_PROJECT

# Experiment 3: Different number of layers - 1 layer
echo "[3/15] Experiment 3: 1 hidden layer"
echo "----------------------------------------"
python main.py \
    --model DAN \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --emb_dim 300 \
    --num_hidden_layers 1 \
    --hidden_dim 100 \
    --dropout False \
    --freeze_embedding True \
    --train_unk_token False \
    --wandb \
    --wandb_project $WANDB_PROJECT

# Experiment 4: Different number of layers - 3 layers
echo "[4/15] Experiment 4: 3 hidden layers"
echo "----------------------------------------"
python main.py \
    --model DAN \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --emb_dim 300 \
    --num_hidden_layers 3 \
    --hidden_dim 100 \
    --dropout False \
    --freeze_embedding True \
    --train_unk_token False \
    --wandb \
    --wandb_project $WANDB_PROJECT

# Experiment 5: Different hidden dimensions - 50
echo "[5/15] Experiment 5: hidden_dim=50"
echo "----------------------------------------"
python main.py \
    --model DAN \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --emb_dim 300 \
    --num_hidden_layers 2 \
    --hidden_dim 300 \
    --dropout False \
    --freeze_embedding True \
    --train_unk_token False \
    --wandb \
    --wandb_project $WANDB_PROJECT

# Experiment 6: Different hidden dimensions - 200
echo "[6/15] Experiment 6: hidden_dim=200"
echo "----------------------------------------"
python main.py \
    --model DAN \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --emb_dim 300 \
    --num_hidden_layers 2 \
    --hidden_dim 500 \
    --dropout False \
    --freeze_embedding True \
    --train_unk_token False \
    --wandb \
    --wandb_project $WANDB_PROJECT

# Experiment 7: Different embedding dimensions - 50d
echo "[7/15] Experiment 7: emb_dim=50"
echo "----------------------------------------"
python main.py \
    --model DAN \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --emb_dim 50 \
    --num_hidden_layers 2 \
    --hidden_dim 100 \
    --dropout False \
    --freeze_embedding True \
    --train_unk_token False \
    --wandb \
    --wandb_project $WANDB_PROJECT

# Experiment 8: With dropout
echo "[8/15] Experiment 8: With dropout (dropout_rate=0.2)"
echo "----------------------------------------"
python main.py \
    --model DAN \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --emb_dim 300 \
    --num_hidden_layers 2 \
    --hidden_dim 100 \
    --dropout True \
    --dropout_rate 0.2 \
    --freeze_embedding True \
    --train_unk_token False \
    --wandb \
    --wandb_project $WANDB_PROJECT

# Experiment 9: With dropout (higher rate)
echo "[9/15] Experiment 9: With dropout (dropout_rate=0.5)"
echo "----------------------------------------"
python main.py \
    --model DAN \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --emb_dim 300 \
    --num_hidden_layers 2 \
    --hidden_dim 100 \
    --dropout True \
    --dropout_rate 0.5 \
    --freeze_embedding True \
    --train_unk_token False \
    --wandb \
    --wandb_project $WANDB_PROJECT

# Experiment 10: With cosine scheduler
echo "[10/15] Experiment 10: With cosine scheduler"
echo "----------------------------------------"
python main.py \
    --model DAN \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --emb_dim 300 \
    --num_hidden_layers 2 \
    --hidden_dim 100 \
    --dropout False \
    --use_cosine_scheduler \
    --freeze_embedding True \
    --train_unk_token False \
    --wandb \
    --wandb_project $WANDB_PROJECT

# Experiment 11: With weight decay (higher)
echo "[11/15] Experiment 11: With higher weight decay (0.001)"
echo "----------------------------------------"
python main.py \
    --model DAN \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --emb_dim 300 \
    --num_hidden_layers 2 \
    --hidden_dim 100 \
    --dropout False \
    --weight_decay 0.001 \
    --freeze_embedding True \
    --train_unk_token False \
    --wandb \
    --wandb_project $WANDB_PROJECT

# Experiment 12: Unfreeze embedding
echo "[12/15] Experiment 12: Unfreeze embedding"
echo "----------------------------------------"
python main.py \
    --model DAN \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --emb_dim 300 \
    --num_hidden_layers 2 \
    --hidden_dim 100 \
    --dropout False \
    --freeze_embedding False \
    --train_unk_token False \
    --wandb \
    --wandb_project $WANDB_PROJECT

# Experiment 13: Train UNK token
echo "[13/15] Experiment 13: Train UNK token"
echo "----------------------------------------"
python main.py \
    --model DAN \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --emb_dim 300 \
    --num_hidden_layers 2 \
    --hidden_dim 100 \
    --dropout False \
    --freeze_embedding True \
    --train_unk_token True \
    --wandb \
    --wandb_project $WANDB_PROJECT

echo ""

# Experiment 14: Combination - dropout + cosine scheduler
echo "[14/15] Experiment 14: Dropout + Cosine Scheduler"
echo "----------------------------------------"
python main.py \
    --model DAN \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --emb_dim 300 \
    --num_hidden_layers 2 \
    --hidden_dim 100 \
    --dropout True \
    --dropout_rate 0.2 \
    --use_cosine_scheduler \
    --freeze_embedding True \
    --train_unk_token False \
    --wandb \
    --wandb_project $WANDB_PROJECT

# Experiment 15: Combination - 3 layers + dropout + higher hidden dim
echo "[15/15] Experiment 15: 3 layers + dropout + hidden_dim=200"
echo "----------------------------------------"
python main.py \
    --model DAN \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --emb_dim 300 \
    --num_hidden_layers 3 \
    --hidden_dim 200 \
    --dropout True \
    --dropout_rate 0.2 \
    --freeze_embedding True \
    --train_unk_token False \
    --wandb \
    --wandb_project $WANDB_PROJECT

echo "=========================================="
echo "All experiments completed!"
