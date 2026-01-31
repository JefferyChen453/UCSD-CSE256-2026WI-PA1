# # ------------------------------- Baseline -------------------------------
# python main.py \
#     --model DAN \
#     --epochs 100 \
#     --lr 1e-4 \
#     --optimizer Adam \
#     --activation ReLU \
#     --batch_size 32 \
#     --weight_decay 1e-3 \
#     --emb_dim 300 \
#     --num_hidden_layers 2 \
#     --hidden_dim 300 \
#     --dropout_word True \
#     --dropout_hidden True \
#     --dropout_rate 0.3 \
#     --load_pretrained_embedding True \
#     --freeze_embedding True \
#     --wandb \
#     --wandb_project CSE256_PA1_DAN \
#     --run_name Baseline

# # ------------------------------- Num hidden layers -------------------------------
# NUM_HIDDEN_LAYERS=(1 2 3)
# for num_hidden_layers in ${NUM_HIDDEN_LAYERS[@]}; do
#     python main.py \
#         --model DAN \
#         --epochs 100 \
#         --lr 1e-4 \
#         --optimizer Adam \
#         --activation ReLU \
#         --batch_size 32 \
#         --weight_decay 1e-3 \
#         --emb_dim 300 \
#         --num_hidden_layers ${num_hidden_layers} \
#         --hidden_dim 300 \
#         --dropout_word True \
#         --dropout_hidden True \
#         --dropout_rate 0.3 \
#         --load_pretrained_embedding True \
#         --freeze_embedding True \
#         --wandb \
#         --wandb_project CSE256_PA1_DAN \
#         --run_name "${num_hidden_layers} Hidden Layers"
# done

# # ------------------------------- Embedding Dimension -------------------------------
# EMB_DIM=(50 300)
# for emb_dim in ${EMB_DIM[@]}; do
#     python main.py \
#         --model DAN \
#         --epochs 100 \
#         --lr 1e-4 \
#         --optimizer Adam \
#         --activation ReLU \
#         --batch_size 32 \
#         --weight_decay 1e-3 \
#         --emb_dim ${emb_dim} \
#         --num_hidden_layers 2 \
#         --hidden_dim 300 \
#         --dropout_word True \
#         --dropout_hidden True \
#         --dropout_rate 0.3 \
#         --load_pretrained_embedding True \
#         --freeze_embedding True \
#         --wandb \
#         --wandb_project CSE256_PA1_DAN \
#         --run_name "${emb_dim} Embedding Dimension"
# done

# # ------------------------------- Optimizer -------------------------------
# python main.py \
#     --model DAN \
#     --epochs 100 \
#     --lr 1e-2 \
#     --optimizer SGD \
#     --activation ReLU \
#     --batch_size 32 \
#     --weight_decay 1e-3 \
#     --emb_dim 300 \
#     --num_hidden_layers 2 \
#     --hidden_dim 300 \
#     --dropout_word True\
#     --dropout_hidden True\
#     --dropout_rate 0.3 \
#     --load_pretrained_embedding True \
#     --freeze_embedding True \
#     --wandb \
#     --wandb_project CSE256_PA1_DAN \
#     --run_name "SGD"

# python main.py \
#     --model DAN \
#     --epochs 100 \
#     --lr 1e-2 \
#     --optimizer Adagrad \
#     --activation ReLU \
#     --batch_size 32 \
#     --weight_decay 1e-3 \
#     --emb_dim 300 \
#     --num_hidden_layers 2 \
#     --hidden_dim 300 \
#     --dropout_word True\
#     --dropout_hidden True\
#     --dropout_rate 0.3 \
#     --load_pretrained_embedding True \
#     --freeze_embedding True \
#     --wandb \
#     --wandb_project CSE256_PA1_DAN \
#     --run_name "Adagrad"


# python main.py \
#     --model DAN \
#     --epochs 100 \
#     --lr 1e-4 \
#     --optimizer Adam \
#     --activation ReLU \
#     --batch_size 32 \
#     --weight_decay 1e-3 \
#     --emb_dim 300 \
#     --num_hidden_layers 2 \
#     --hidden_dim 300 \
#     --dropout_word True\
#     --dropout_hidden True\
#     --dropout_rate 0.3 \
#     --load_pretrained_embedding True \
#     --freeze_embedding True \
#     --wandb \
#     --wandb_project CSE256_PA1_DAN \
#     --run_name "Adam"

# # ------------------------------- Activation Function -------------------------------
# ACTIVATION=(ReLU SiLU Tanh)
# for activation in ${ACTIVATION[@]}; do
#     python main.py \
#         --model DAN \
#         --epochs 100 \
#         --lr 1e-4 \
#         --optimizer Adam \
#         --activation ${activation} \
#         --batch_size 32 \
#         --weight_decay 1e-3 \
#         --emb_dim 300 \
#         --num_hidden_layers 2 \
#         --hidden_dim 300 \
#         --dropout_word True\
#         --dropout_hidden True\
#         --dropout_rate 0.3 \
#         --load_pretrained_embedding True \
#         --freeze_embedding True \
#         --wandb \
#         --wandb_project CSE256_PA1_DAN \
#         --run_name "${activation}"
# done

# # ------------------------------- Dropout -------------------------------
# python main.py \
#     --model DAN \
#     --epochs 100 \
#     --lr 1e-4 \
#     --optimizer Adam \
#     --activation ReLU \
#     --batch_size 32 \
#     --weight_decay 1e-3 \
#     --emb_dim 300 \
#     --num_hidden_layers 2 \
#     --hidden_dim 300 \
#     --dropout_word True\
#     --dropout_hidden False\
#     --dropout_rate 0.3 \
#     --load_pretrained_embedding True \
#     --freeze_embedding True \
#     --wandb \
#     --wandb_project CSE256_PA1_DAN \
#     --run_name "Word Dropout"

# python main.py \
#     --model DAN \
#     --epochs 100 \
#     --lr 1e-4 \
#     --optimizer Adam \
#     --activation ReLU \
#     --batch_size 32 \
#     --weight_decay 1e-3 \
#     --emb_dim 300 \
#     --num_hidden_layers 2 \
#     --hidden_dim 300 \
#     --dropout_word False\
#     --dropout_hidden True\
#     --dropout_rate 0.3 \
#     --load_pretrained_embedding True \
#     --freeze_embedding True \
#     --wandb \
#     --wandb_project CSE256_PA1_DAN \
#     --run_name "Hidden Dropout"

# python main.py \
#     --model DAN \
#     --epochs 100 \
#     --lr 1e-4 \
#     --optimizer Adam \
#     --activation ReLU \
#     --batch_size 32 \
#     --weight_decay 1e-3 \
#     --emb_dim 300 \
#     --num_hidden_layers 2 \
#     --hidden_dim 300 \
#     --dropout_word True\
#     --dropout_hidden True\
#     --dropout_rate 0.3 \
#     --load_pretrained_embedding True \
#     --freeze_embedding True \
#     --wandb \
#     --wandb_project CSE256_PA1_DAN \
#     --run_name "Both Dropout"

# # ------------------------------- Pretrained Embedding -------------------------------
# python main.py \
#     --model DAN \
#     --epochs 100 \
#     --lr 1e-4 \
#     --optimizer Adam \
#     --activation ReLU \
#     --batch_size 32 \
#     --weight_decay 1e-3 \
#     --emb_dim 300 \
#     --num_hidden_layers 2 \
#     --hidden_dim 300 \
#     --dropout_word True\
#     --dropout_hidden True\
#     --dropout_rate 0.3 \
#     --load_pretrained_embedding True \
#     --freeze_embedding True \
#     --wandb \
#     --wandb_project CSE256_PA1_DAN \
#     --run_name "Pretrained Embedding"

python main.py \
    --model DAN \
    --epochs 100 \
    --lr 1e-4 \
    --optimizer Adam \
    --activation ReLU \
    --batch_size 32 \
    --weight_decay 1e-3 \
    --emb_dim 300 \
    --num_hidden_layers 2 \
    --hidden_dim 300 \
    --dropout_word True\
    --dropout_hidden True\
    --dropout_rate 0.3 \
    --load_pretrained_embedding False \
    --freeze_embedding True \
    --wandb \
    --wandb_project CSE256_PA1_DAN \
    --run_name "Random Embedding"