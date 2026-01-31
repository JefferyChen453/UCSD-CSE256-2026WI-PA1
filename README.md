# CSE 256 PA1: Sentiment Analysis

Sentiment classification using Bag-of-Words (BOW), Deep Averaging Networks (DAN), and SubwordDAN models.

**All experiment logs available on [Weights & Biases](https://wandb.ai/chentianyi453/CSE256_PA1).**

## Quick Start
### Train Models
```bash
# Train BOW model (NN2 & NN3)
python main.py --model BOW

# Train DAN with pretrained GloVe embeddings
python main.py --model DAN

# Train SubwordDAN with BPE tokenizer
python main.py --model SUBWORDDAN --epochs 200
```
### Train BPE Tokenizer
```bash
python tokenizer/bpe.py
```

## Parameter Usage

### Training (all models)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | required | Model type: `BOW`, `DAN`, or `SUBWORDDAN` |
| `--epochs` | int | 100 | Number of training epochs |
| `--lr` | float | 1e-4 | Learning rate |
| `--optimizer` | str | Adam | Optimizer: `SGD`, `Adagrad`, or `Adam` |
| `--activation` | str | ReLU | Activation: `ReLU`, `SiLU`, or `Tanh` |
| `--batch_size` | int | 16 | Batch size |
| `--weight_decay` | float | 1e-3 | L2 regularization |
| `--use_cosine_scheduler` | flag | - | Use cosine annealing LR scheduler |
| `--wandb` | flag | - | Enable Weights & Biases logging |
| `--wandb_project` | str | CSE256_PA1 | WandB project name |
| `--run_name` | str | None | Custom run name for WandB |

### DAN / SubwordDAN

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--emb_dim` | int | 300 | Embedding dimension (50 or 300 for pretrained GloVe) |
| `--num_hidden_layers` | int | 2 | Number of hidden FFN layers |
| `--hidden_dim` | int | 100 | Hidden layer dimension |
| `--dropout_word` | bool | True | Dropout after word embedding |
| `--dropout_hidden` | bool | True | Dropout after hidden layers |
| `--dropout_rate` | float | 0.2 | Dropout probability |
| `--load_pretrained_embedding` | bool | True | Load GloVe embeddings (DAN only) |
| `--freeze_embedding` | bool | True | Freeze embedding layer |
| `--tokenizer_path` | str | tokenizer/bpe_sentiment/10000 | BPE tokenizer path (SubwordDAN only) |

### BOW

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--bow_hidden_size` | int | 100 | Hidden layer size |
| `--input_size` | int | 512 | Vocabulary size for CountVectorizer |

## Optimal Results

### DAN (Best: Dev Acc 80.7%)

| Parameter | Value |
|-----------|-------|
| epochs | 100 |
| lr | 1e-4 |
| batch_size | 32 |
| weight_decay | 1e-3 |
| optimizer | adam (default) |
| activation | ReLU (default) |
| use_cosine_scheduler | False (default) |
| emb_dim | 300 |
| num_hidden_layers | 2 |
| hidden_dim | 300 |
| dropout_word | True |
| dropout_hidden | True |
| dropout_rate | 0.3 |
| load_pretrained_embedding | True |
| freeze_embedding | False |
| **Dev Accuracy** | **0.807** |

### SubwordDAN (Best: Dev Acc 78.2%)

| Parameter | Value |
|-----------|-------|
| vocab_size | 10000 |
| tokenizer_path | tokenizer/bpe_sentiment/10000 |
| (other params similar to DAN baseline) | |
| **Dev Accuracy** | **0.782** |
