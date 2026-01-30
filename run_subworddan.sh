#!/bin/bash
VOCAB_SIZE=(5000 8000 10000 15000 20000)

for vocab_size in "${VOCAB_SIZE[@]}"; do
    python main.py \
        --model SUBWORDDAN \
        --epochs 200 \
        --lr 1e-4 \
        --batch_size 32 \
        --weight_decay 1e-3 \
        --emb_dim 300 \
        --num_hidden_layers 2 \
        --hidden_dim 300 \
        --hidden_dim 500 \
        --dropout_word True \
        --dropout_hidden True \
        --dropout_rate 0.3 \
        --tokenizer_path tokenizer/bpe_sentiment/${vocab_size} \
        --wandb \
        --wandb_project CSE256_PA1_DAN \
        --run_name SubwordDAN-${vocab_size}-Epochs-200
done
