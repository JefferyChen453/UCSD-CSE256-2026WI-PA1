import argparse
from functools import partial
import os
import time
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from BOWmodels import NN2BOW, NN3BOW, SentimentDatasetBOW
from DANmodels import DAN, SentimentDatasetDAN, collate_fn
from SubwordDANmodels import SentimentDatasetSubwordDAN, SubwordDAN
from tokenizer import Tokenizer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Auto-detect device: use CUDA if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training function
def train_epoch(model_name, data_loader, model, loss_fn, optimizer, device=DEVICE):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0.0, device=device)
    grad_norm_sum = 0.0
    for batch in data_loader:
        if model_name == "BOW":
            X = batch[0].float().to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)
        elif model_name == "DAN" or model_name == "SubwordDAN":
            X = batch["input_ids"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.detach()
        correct += (pred.argmax(1) == y).type(torch.float).sum().detach()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
        grad_norm_sum += grad_norm.item()
        optimizer.step()

    average_train_loss = (train_loss / num_batches).item()
    accuracy = (correct / size).item()
    average_grad_norm = grad_norm_sum / num_batches
    return accuracy, average_train_loss, average_grad_norm


# Evaluation function
def eval_epoch(model_name, data_loader, model, loss_fn, device=DEVICE):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    if hasattr(model, "training"):
        model.training = False
    eval_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for batch in data_loader:
            if model_name == "BOW":
                X = batch[0].float().to(device, non_blocking=True)
                y = batch[1].to(device, non_blocking=True)
            elif model_name == "DAN" or model_name == "SubwordDAN":
                X = batch["input_ids"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            eval_loss += loss.detach()
            correct += (pred.argmax(1) == y).type(torch.float).sum().detach()

    average_eval_loss = (eval_loss / num_batches).item()
    accuracy = (correct / size).item()
    if hasattr(model, "training"):
        model.training = True
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model_name, model, train_loader, test_loader, epochs=100, lr=1e-4, weight_decay=0.0, optimizer_type="adam", use_wandb=False, run_name=None, device=DEVICE):
    model = model.to(device)
    loss_fn = nn.NLLLoss()
    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    all_train_accuracy = []
    all_test_accuracy = []
    all_train_loss = []
    all_test_loss = []
    
    for epoch in range(epochs):
        train_accuracy, train_loss, grad_norm = train_epoch(model_name, train_loader, model, loss_fn, optimizer, device)
        all_train_accuracy.append(train_accuracy)
        all_train_loss.append(train_loss)

        test_accuracy, test_loss = eval_epoch(model_name, test_loader, model, loss_fn, device)
        all_test_accuracy.append(test_accuracy)
        all_test_loss.append(test_loss)

        current_lr = lr

        # Log to wandb if enabled
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch + 1,
                "train/accuracy": train_accuracy,
                "train/loss": train_loss,
                "train/grad_norm": grad_norm,
                "dev/accuracy": test_accuracy,
                "dev/loss": test_loss,
                "learning_rate": current_lr,
            })

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}, lr {current_lr:.6f}')
    
    return all_train_accuracy, all_test_accuracy

def str2bool(v):
    """Parse string to bool for argparse. bool('False') is True in Python!"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    train_group.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    train_group.add_argument("--optimizer", type=str, default="Adam", choices=["SGD", "Adagrad", "Adam"], help="Optimizer: SGD, Adagrad, or Adam (default: Adam)")
    train_group.add_argument("--activation", type=str, default="ReLU", choices=["ReLU", "SiLU", "Tanh"], help="Activation function: ReLU, SiLU, or Tanh (default: ReLU)")
    train_group.add_argument("--batch_size", type=int, default=32, help="Batch size")
    train_group.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay (L2 regularization)")
    train_group.add_argument("--wandb", action='store_true', help="Log by WandB")
    train_group.add_argument("--wandb_project", type=str, default="CSE256_PA1", help="WandB project name")
    train_group.add_argument("--run_name", type=str, default=None, help="Run name")
    
    dan_group = parser.add_argument_group("DAN")
    dan_group.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
    dan_group.add_argument("--num_hidden_layers", type=int, default=2, help="Number of hidden layers")
    dan_group.add_argument("--hidden_dim", type=int, default=300, help="Hidden dimension")
    dan_group.add_argument("--dropout_word", type=str2bool, default=True, help="Use dropout after word embedding")
    dan_group.add_argument("--dropout_hidden", type=str2bool, default=True, help="Use dropout after hidden layers")
    dan_group.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate")
    dan_group.add_argument("--load_pretrained_embedding", type=str2bool, default=True, help="True to load pretrained 50d/300d embedding; False to init random embedding")
    dan_group.add_argument("--freeze_embedding", type=str2bool, default=False, help="True to freeze embedding; False to train embedding")

    subword_group = parser.add_argument_group("SubwordDAN")
    subword_group.add_argument("--tokenizer_path", type=str, default="tokenizer/bpe_sentiment/10000", help="Tokenizer path")

    bow_group = parser.add_argument_group("BOW")
    bow_group.add_argument("--bow_hidden_size", type=int, default=100, help="Hidden dimension")
    bow_group.add_argument("--input_size", type=int, default=512, help="Input dimension")

    # Parse the command-line arguments
    args = parser.parse_args()
    print(args)
    print(f"Using device: {DEVICE}")

    # Check wandb availability if enabled
    if args.wandb:
        if not WANDB_AVAILABLE:
            print("Error: wandb is not installed. Install with: pip install wandb")
            return

    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Load dataset
        start_time = time.time()

        train_data = SentimentDatasetBOW("data/train.txt")
        dev_data = SentimentDatasetBOW("data/dev.txt", vectorizer=train_data.vectorizer, train=False)
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            pin_memory=(DEVICE.type == "cuda"),
        )
        test_loader = DataLoader(
            dev_data, batch_size=args.batch_size, shuffle=False,
            pin_memory=(DEVICE.type == "cuda"),
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Initialize wandb for NN2 if enabled
        if args.wandb and WANDB_AVAILABLE:
            wandb.init(
                entity="chentianyi453",
                project=args.wandb_project,
                config={
                    "model": "BOW_NN2",
                    "num_layers": 2,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "optimizer": args.optimizer,
                    "activation": args.activation,
                    "batch_size": args.batch_size,
                    "input_size": args.input_size,
                    "bow_hidden_size": args.bow_hidden_size,
                    "weight_decay": args.weight_decay,
                },
                name=f"BOW_NN2_{args.epochs}epochs_lr{args.lr}",
                reinit=True,
            )

        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(
            "BOW", 
            NN2BOW(input_size=args.input_size, bow_hidden_size=args.bow_hidden_size, activation=args.activation), 
            train_loader, 
            test_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            optimizer_type=args.optimizer,
            use_wandb=args.wandb,
            run_name="NN2BOW"
        )

        # Finish NN2 wandb run
        if args.wandb and WANDB_AVAILABLE:
            wandb.finish()

        # Initialize wandb for NN3 if enabled
        if args.wandb and WANDB_AVAILABLE:
            wandb.init(
                entity="chentianyi453",
                project=args.wandb_project,
                config={
                    "model": "BOW_NN3",
                    "num_layers": 3,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "optimizer": args.optimizer,
                    "activation": args.activation,
                    "batch_size": args.batch_size,
                    "input_size": args.input_size,
                    "bow_hidden_size": args.bow_hidden_size,
                    "weight_decay": args.weight_decay,
                },
                name=f"BOW_NN3_{args.epochs}epochs_lr{args.lr}",
                reinit=True,
            )

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(
            "BOW", 
            NN3BOW(input_size=args.input_size, bow_hidden_size=args.bow_hidden_size, activation=args.activation), 
            train_loader, 
            test_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            optimizer_type=args.optimizer,
            use_wandb=args.wandb,
            run_name="NN3BOW"
        )

        # Finish NN3 wandb run
        if args.wandb and WANDB_AVAILABLE:
            wandb.finish()

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":
        # Load dataset
        start_time = time.time()

        train_data = SentimentDatasetDAN(
            "data/train.txt",
            emb_dim=args.emb_dim,
            load_pretrained_embedding=args.load_pretrained_embedding,
        )
        dev_data = SentimentDatasetDAN(
            "data/dev.txt",
            emb_dim=args.emb_dim,
            load_pretrained_embedding=args.load_pretrained_embedding,
        )
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
            pin_memory=(DEVICE.type == "cuda"),
        )
        test_loader = DataLoader(
            dev_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
            pin_memory=(DEVICE.type == "cuda"),
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Initialize wandb for DAN if enabled
        if args.wandb and WANDB_AVAILABLE:
            # Build detailed run name
            run_name_parts = [
                f"DAN",
                f"emb{args.emb_dim}",
                f"layers{args.num_hidden_layers}",
                f"hidden{args.hidden_dim}",
            ]
            if args.run_name:
                run_name = args.run_name
            else:
                if args.dropout_word:
                    run_name_parts.append("dropout-word")
                if args.dropout_hidden:
                    run_name_parts.append("dropout-hidden")
                if args.dropout_rate > 0:
                    run_name_parts.append(f"dropout{args.dropout_rate}")
                if args.weight_decay > 0:
                    run_name_parts.append(f"wd{args.weight_decay}")
                if not args.freeze_embedding:
                    run_name_parts.append("unfrozen-emb")
                
                run_name = "_".join(run_name_parts)
            
            wandb.init(
                entity="chentianyi453",
                project=args.wandb_project,
                config={
                    "model": "DAN",
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "optimizer": args.optimizer,
                    "activation": args.activation,
                    "batch_size": args.batch_size,
                    "weight_decay": args.weight_decay,
                    "emb_dim": args.emb_dim,
                    "num_hidden_layers": args.num_hidden_layers,
                    "hidden_dim": args.hidden_dim,
                    "dropout_word": args.dropout_word,
                    "dropout_hidden": args.dropout_hidden,
                    "dropout_rate": args.dropout_rate,
                    "load_pretrained_embedding": args.load_pretrained_embedding,
                    "freeze_embedding": args.freeze_embedding,
                },
                name=run_name,
            )

        # Train and evaluate DAN
        print(args)
        start_time = time.time()
        print('\nDAN:')
        dan_train_accuracy, dan_test_accuracy = experiment(
            "DAN",
            DAN(
                emb_dim=args.emb_dim,
                num_hidden_layers=args.num_hidden_layers,
                hidden_dim=args.hidden_dim,
                training=True,
                dropout_word=args.dropout_word,
                dropout_hidden=args.dropout_hidden,
                dropout_rate=args.dropout_rate,
                load_pretrained_embedding=args.load_pretrained_embedding,
                freeze_embedding=args.freeze_embedding,
                activation=args.activation,
            ),
            train_loader,
            test_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            optimizer_type=args.optimizer,
            use_wandb=args.wandb,
            run_name="DAN"
        )

        # Finish DAN wandb run
        if args.wandb and WANDB_AVAILABLE:
            wandb.finish()

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_test_accuracy, label='DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for DAN')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dan_dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"DAN dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()
    elif args.model == "SUBWORDDAN":
        # Load dataset
        start_time = time.time()

        tokenizer = Tokenizer.from_files(os.path.join(args.tokenizer_path, "vocab.json"), os.path.join(args.tokenizer_path, "merges.txt"))
        train_data = SentimentDatasetSubwordDAN("data/train.txt", tokenizer=tokenizer)
        dev_data = SentimentDatasetSubwordDAN("data/dev.txt", tokenizer=tokenizer)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=partial(collate_fn, pad_token_id=1), pin_memory=(DEVICE.type == "cuda"))
        test_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False, collate_fn=partial(collate_fn, pad_token_id=1), pin_memory=(DEVICE.type == "cuda"))

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Initialize wandb for SubwordDAN if enabled
        if args.wandb and WANDB_AVAILABLE:
            wandb.init(
                entity="chentianyi453",
                project=args.wandb_project,
                config={
                    "model": "SubwordDAN",
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "optimizer": args.optimizer,
                    "activation": args.activation,
                    "batch_size": args.batch_size,
                    "weight_decay": args.weight_decay,
                    "emb_dim": args.emb_dim,
                    "num_hidden_layers": args.num_hidden_layers,
                    "hidden_dim": args.hidden_dim,
                    "dropout_word": args.dropout_word,
                    "dropout_hidden": args.dropout_hidden,
                    "dropout_rate": args.dropout_rate,
                    "vocab_size": len(tokenizer.vocab),
                },
                name=args.run_name,
                reinit=True,
            )
        
        # Train and evaluate SubwordDAN
        start_time = time.time()
        print('\nSubwordDAN:')
        subword_train_accuracy, subword_test_accuracy = experiment(
            "SubwordDAN",
            SubwordDAN(
                tokenizer=tokenizer,
                emb_dim=args.emb_dim,
                num_hidden_layers=args.num_hidden_layers,
                hidden_dim=args.hidden_dim,
                training=True,
                dropout_word=args.dropout_word,
                dropout_hidden=args.dropout_hidden,
                dropout_rate=args.dropout_rate,
                activation=args.activation,
            ),
            train_loader,
            test_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            optimizer_type=args.optimizer,
            use_wandb=args.wandb,
            run_name="SubwordDAN"
        )

        # Finish SubwordDAN wandb run
        if args.wandb and WANDB_AVAILABLE:
            wandb.finish()

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(subword_test_accuracy, label='SubwordDAN')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for SubwordDAN')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'subword_dan_dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"SubwordDAN dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

if __name__ == "__main__":
    main()
