# ------------------------------- Baseline -------------------------------
python main.py \
    --model DAN \
    --epochs 100 \
    --lr 1e-4 \
    --batch_size 32 \
    --weight_decay 1e-3 \
    --emb_dim 300 \
    --num_hidden_layers 2 \
    --hidden_dim 300 \
    --freeze_embedding False \
    --train_unk_token True \
    --hidden_dim 500 \
    --dropout_word True \
    --dropout_hidden True \
    --dropout_rate 0.3 \
    --load_pretrained_embedding True \
    --freeze_embedding False \
    --train_unk_token True \
    --wandb \
    --wandb_project CSE256_PA1_DAN \
    --run_name baseline

# ------------------------------- Num hidden layers -------------------------------
NUM_HIDDEN_LAYERS=(1 2 3)
for num_hidden_layers in ${NUM_HIDDEN_LAYERS[@]}; do
    python main.py \
        --model DAN \
        --epochs 100 \
        --lr 1e-4 \
        --batch_size 32 \
        --weight_decay 1e-3 \
        --emb_dim 300 \
        --num_hidden_layers ${num_hidden_layers} \
        --hidden_dim 300 \
        --freeze_embedding False \
        --train_unk_token True \
        --hidden_dim 500 \
        --dropout_word True \
        --dropout_hidden True \
        --dropout_rate 0.3 \
        --load_pretrained_embedding True \
        --freeze_embedding False \
        --train_unk_token True \
        --wandb \
        --wandb_project CSE256_PA1_DAN \
        --run_name "${num_hidden_layers} Hidden Layers"
done

# ------------------------------- Num hidden layers -------------------------------
EMB_DIM=(50 300)
for emb_dim in ${EMB_DIM[@]}; do
    python main.py \
        --model DAN \
        --epochs 100 \
        --lr 1e-4 \
        --batch_size 32 \
        --weight_decay 1e-3 \
        --emb_dim ${emb_dim} \
        --num_hidden_layers 2 \
        --hidden_dim ${emb_dim} \
        --freeze_embedding False \
        --train_unk_token True \
        --hidden_dim 500 \
        --dropout_word True \
        --dropout_hidden True \
        --dropout_rate 0.3 \
        --load_pretrained_embedding True \
        --freeze_embedding False \
        --train_unk_token True \
        --wandb \
        --wandb_project CSE256_PA1_DAN \
        --run_name "${emb_dim} Embedding Dimension"
done

# ------------------------------- Optimizer -------------------------------
OPTIMIZER=(sgd adagrad adam)
for optimizer in ${OPTIMIZER[@]}; do
    python main.py \
        --model DAN \
        --epochs 100 \
        --lr 1e-4 \
        --optimizer ${optimizer} \
        --batch_size 32 \
        --weight_decay 1e-3 \
        --emb_dim 300 \
        --num_hidden_layers 2 \
        --hidden_dim 300 \
        --freeze_embedding False \
        --train_unk_token True \
        --hidden_dim 500 \
        --dropout_word True \
        --dropout_hidden True \
        --dropout_rate 0.3 \
        --load_pretrained_embedding True \
        --freeze_embedding False \
        --train_unk_token True \
        --wandb \
        --wandb_project CSE256_PA1_DAN \
        --run_name "${optimizer}"
done

# ------------------------------- Activation Function -------------------------------
ACTIVATION=(relu silu tanh)
for activation in ${ACTIVATION[@]}; do
    python main.py \
        --model DAN \
        --epochs 100 \
        --lr 1e-4 \
        --activation ${activation} \
        --batch_size 32 \
        --weight_decay 1e-3 \
        --emb_dim 300 \
        --num_hidden_layers 2 \
        --hidden_dim 300 \
        --freeze_embedding False \
        --train_unk_token True \
        --hidden_dim 500 \
        --dropout_word True \
        --dropout_hidden True \
        --dropout_rate 0.3 \
        --load_pretrained_embedding True \
        --freeze_embedding False \
        --train_unk_token True \
        --wandb \
        --wandb_project CSE256_PA1_DAN \
        --run_name "${activation}"
done

# ------------------------------- Dropout -------------------------------
python main.py \
    --model DAN \
    --epochs 100 \
    --lr 1e-4 \
    --batch_size 32 \
    --weight_decay 1e-3 \
    --emb_dim 300 \
    --num_hidden_layers 2 \
    --hidden_dim 300 \
    --freeze_embedding False \
    --train_unk_token True \
    --hidden_dim 500 \
    --dropout_word True \
    --dropout_hidden False \
    --dropout_rate 0.3 \
    --load_pretrained_embedding True \
    --freeze_embedding False \
    --train_unk_token True \
    --wandb \
    --wandb_project CSE256_PA1_DAN \
    --run_name "Word Dropout"

python main.py \
    --model DAN \
    --epochs 100 \
    --lr 1e-4 \
    --batch_size 32 \
    --weight_decay 1e-3 \
    --emb_dim 300 \
    --num_hidden_layers 2 \
    --hidden_dim 300 \
    --freeze_embedding False \
    --train_unk_token True \
    --hidden_dim 500 \
    --dropout_word False \
    --dropout_hidden True \
    --dropout_rate 0.3 \
    --load_pretrained_embedding True \
    --freeze_embedding False \
    --train_unk_token True \
    --wandb \
    --wandb_project CSE256_PA1_DAN \
    --run_name "Hidden Dropout"

python main.py \
    --model DAN \
    --epochs 100 \
    --lr 1e-4 \
    --batch_size 32 \
    --weight_decay 1e-3 \
    --emb_dim 300 \
    --num_hidden_layers 2 \
    --hidden_dim 300 \
    --freeze_embedding False \
    --train_unk_token True \
    --hidden_dim 500 \
    --dropout_word True \
    --dropout_hidden True \
    --dropout_rate 0.3 \
    --load_pretrained_embedding True \
    --freeze_embedding False \
    --train_unk_token True \
    --wandb \
    --wandb_project CSE256_PA1_DAN \
    --run_name "Both Dropout"

# ------------------------------- Pretrained Embedding -------------------------------
python main.py \
    --model DAN \
    --epochs 100 \
    --lr 1e-4 \
    --batch_size 32 \
    --weight_decay 1e-3 \
    --emb_dim 300 \
    --num_hidden_layers 2 \
    --hidden_dim 300 \
    --freeze_embedding False \
    --train_unk_token True \
    --hidden_dim 500 \
    --dropout_word True \
    --dropout_hidden True \
    --dropout_rate 0.3 \
    --load_pretrained_embedding True \
    --freeze_embedding False \
    --train_unk_token True \
    --wandb \
    --wandb_project CSE256_PA1_DAN \
    --run_name "Pretrained Embedding"

python main.py \
    --model DAN \
    --epochs 100 \
    --lr 1e-4 \
    --batch_size 32 \
    --weight_decay 1e-3 \
    --emb_dim 300 \
    --num_hidden_layers 2 \
    --hidden_dim 300 \
    --freeze_embedding False \
    --train_unk_token True \
    --hidden_dim 500 \
    --dropout_word True \
    --dropout_hidden True \
    --dropout_rate 0.3 \
    --load_pretrained_embedding False \
    --freeze_embedding False \
    --train_unk_token True \
    --wandb \
    --wandb_project CSE256_PA1_DAN \
    --run_name "Random Embedding"

