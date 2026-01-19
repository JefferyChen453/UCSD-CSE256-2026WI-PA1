import argparse
import time
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from BOWmodels import NN2BOW, NN3BOW, SentimentDatasetBOW
from DANmodels import DAN, SentimentDatasetDAN, collate_fn

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Training function
def train_epoch(model_name, data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch in data_loader:
        if model_name == "BOW":
            X = batch[0].float()
            y = batch[1]
        elif model_name == "DAN":
            X = batch["input_ids"]
            y = batch["labels"]

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(model_name, data_loader, model, loss_fn):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch in data_loader:
        if model_name == "BOW":
            X = batch[0].float()
            y = batch[1]
        elif model_name == "DAN":
            X = batch["input_ids"]
            y = batch["labels"]

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model_name, model, train_loader, test_loader, epochs=100, lr=1e-4, weight_decay=0.0, use_cosine_scheduler=False, use_wandb=False, run_name=None):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Create cosine scheduler if enabled
    scheduler = None
    if use_cosine_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.1*lr)

    all_train_accuracy = []
    all_test_accuracy = []
    all_train_loss = []
    all_test_loss = []
    
    for epoch in range(epochs):
        train_accuracy, train_loss = train_epoch(model_name, train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)
        all_train_loss.append(train_loss)

        test_accuracy, test_loss = eval_epoch(model_name, test_loader, model, loss_fn)
        all_test_accuracy.append(test_accuracy)
        all_test_loss.append(test_loss)

        # Step scheduler after each epoch if enabled
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = lr

        # Log to wandb if enabled
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch + 1,
                "train/accuracy": train_accuracy,
                "train/loss": train_loss,
                "dev/accuracy": test_accuracy,
                "dev/loss": test_loss,
                "learning_rate": current_lr,
            })

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}, lr {current_lr:.6f}')
    
    return all_train_accuracy, all_test_accuracy


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    train_group.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    train_group.add_argument("--batch_size", type=int, default=16, help="Batch size")
    train_group.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay (L2 regularization)")
    train_group.add_argument("--use_cosine_scheduler", action='store_true', help="Use cosine annealing learning rate scheduler")
    train_group.add_argument("--wandb", action='store_true', help="Log by WandB")
    train_group.add_argument("--wandb_project", type=str, default="CSE256_PA1", help="WandB project name")
    
    dan_group = parser.add_argument_group("DAN")
    dan_group.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
    dan_group.add_argument("--load_pretrained_embedding", type=bool, default=True, help="True to load pretrained 50d/300d embedding; False to init random embedding")
    dan_group.add_argument("--freeze_embedding", type=bool, default=True, help="True to freeze embedding; False to train embedding")
    dan_group.add_argument("--train_unk_token", type=bool, default=False, help="True to train unk token; False to freeze unk token")
    dan_group.add_argument("--num_hidden_layers", type=int, default=2, help="Number of hidden layers")
    dan_group.add_argument("--hidden_dim", type=int, default=100, help="Hidden dimension")
    dan_group.add_argument("--dropout", type=bool, default=False, help="Use dropout after hidden layers")
    dan_group.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")


    bow_group = parser.add_argument_group("BOW")
    bow_group.add_argument("--bow_hidden_size", type=int, default=100, help="Hidden dimension")
    bow_group.add_argument("--input_size", type=int, default=512, help="Input dimension")

    # Parse the command-line arguments
    args = parser.parse_args()

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
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Initialize wandb for NN2 if enabled
        if args.wandb and WANDB_AVAILABLE:
            wandb.init(
                project=args.wandb_project,
                config={
                    "model": "BOW_NN2",
                    "num_layers": 2,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "input_size": args.input_size,
                    "bow_hidden_size": args.bow_hidden_size,
                    "weight_decay": args.weight_decay,
                    "use_cosine_scheduler": args.use_cosine_scheduler,
                },
                name=f"BOW_NN2_{args.epochs}epochs_lr{args.lr}",
                reinit=True,
            )

        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(
            "BOW", 
            NN2BOW(input_size=args.input_size, bow_hidden_size=args.bow_hidden_size), 
            train_loader, 
            test_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_cosine_scheduler=args.use_cosine_scheduler,
            use_wandb=args.wandb,
            run_name="NN2BOW"
        )

        # Finish NN2 wandb run
        if args.wandb and WANDB_AVAILABLE:
            wandb.finish()

        # Initialize wandb for NN3 if enabled
        if args.wandb and WANDB_AVAILABLE:
            wandb.init(
                project=args.wandb_project,
                config={
                    "model": "BOW_NN3",
                    "num_layers": 3,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "input_size": args.input_size,
                    "bow_hidden_size": args.bow_hidden_size,
                    "weight_decay": args.weight_decay,
                    "use_cosine_scheduler": args.use_cosine_scheduler,
                },
                name=f"BOW_NN3_{args.epochs}epochs_lr{args.lr}",
                reinit=True,
            )

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(
            "BOW", 
            NN3BOW(input_size=args.input_size, bow_hidden_size=args.bow_hidden_size), 
            train_loader, 
            test_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_cosine_scheduler=args.use_cosine_scheduler,
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
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

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
            if args.dropout:
                run_name_parts.append(f"dropout{args.dropout_rate}")
            if args.use_cosine_scheduler:
                run_name_parts.append("cosine")
            if args.weight_decay > 0:
                run_name_parts.append(f"wd{args.weight_decay}")
            if not args.freeze_embedding:
                run_name_parts.append("unfrozen-emb")
            if args.train_unk_token:
                run_name_parts.append("train-unk")
            
            run_name = "_".join(run_name_parts)
            
            wandb.init(
                project=args.wandb_project,
                config={
                    "model": "DAN",
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "emb_dim": args.emb_dim,
                    "load_pretrained_embedding": args.load_pretrained_embedding,
                    "freeze_embedding": args.freeze_embedding,
                    "train_unk_token": args.train_unk_token,
                    "num_hidden_layers": args.num_hidden_layers,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "dropout_rate": args.dropout_rate,
                    "weight_decay": args.weight_decay,
                    "use_cosine_scheduler": args.use_cosine_scheduler,
                },
                name=run_name,
            )

        # Train and evaluate DAN
        start_time = time.time()
        print('\nDAN:')
        dan_train_accuracy, dan_test_accuracy = experiment(
            "DAN",
            DAN(
                emb_dim=args.emb_dim,
                num_hidden_layers=args.num_hidden_layers,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                dropout_rate=args.dropout_rate,
                load_pretrained_embedding=args.load_pretrained_embedding,
                freeze_embedding=args.freeze_embedding,
                train_unk_token=args.train_unk_token,
            ),
            train_loader,
            test_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_cosine_scheduler=args.use_cosine_scheduler,
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

if __name__ == "__main__":
    main()
