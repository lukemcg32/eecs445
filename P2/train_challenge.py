"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Train Challenge
    Train a convolutional neural network to classify the heldout images
    Periodically output training information, and saves model checkpoints
    Usage: python train_challenge.py
"""

import torch
import matplotlib.pyplot as plt

from challenge_dataset import get_train_val_test_loaders, get_full_target_loader #, get_pretrain_loaders
from model.challenge import Challenge
from train_common import evaluate_epoch, early_stopping, restore_checkpoint, save_checkpoint, train_epoch, count_parameters
from utils import config, set_random_seed, make_training_plot


def main():
    set_random_seed()
    
    # Target-only loaders for fine-tuning - 2 classes
    # tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
    #     task="target",
    #     batch_size=config("challenge.batch_size"),
    # )

    full_loader = get_full_target_loader(batch_size=config("challenge.batch_size"))

    # # Combined source + target loaders for pretraining - 10 classes
    # pre_tr_loader, pre_va_loader, pre_te_loader = get_pretrain_loaders(
    #     batch_size=config("challenge.batch_size"),
    # )


    # fine tune model on target
    model = Challenge(num_classes=2)

    print("Loading challenge checkpoint if it exists...")
    model, start_epoch, stats = restore_checkpoint(
        model,
        checkpoint_dir=config("challenge.checkpoint"),
        cuda=torch.cuda.is_available(),
    )


    # # if no pre trained model exist (at 0 epochs) create our source
    # if start_epoch == 0:
    #     print("Starting with 10 class pretraining...")

    #     pre_model = Challenge(num_classes=10)
    #     pre_criterion = torch.nn.CrossEntropyLoss()

    #     pre_optimizer = torch.optim.Adam(
    #         pre_model.parameters(),
    #         lr=1e-2,
    #         weight_decay=1e-4,
    #     )

    #     pre_axes = make_training_plot("Pretraining CNN (10 classes)")
    #     pre_stats = []

    #     evaluate_epoch(
    #         pre_axes,
    #         pre_tr_loader,
    #         pre_va_loader,
    #         pre_te_loader,
    #         pre_model,
    #         pre_criterion,
    #         epoch=0,
    #         stats=pre_stats,
    #         include_test=True,
    #         multiclass=True
    #     )

    #     num_pretrain_epochs = 10

    #     for pre_epoch in range(num_pretrain_epochs):
    #         print(f"[Pretrain] Epoch {pre_epoch + 1}/{num_pretrain_epochs}")

    #         train_epoch(pre_tr_loader, pre_model, pre_criterion, pre_optimizer)

    #         evaluate_epoch(
    #             pre_axes,
    #             pre_tr_loader,
    #             pre_va_loader,
    #             pre_te_loader,
    #             pre_model,
    #             pre_criterion,
    #             epoch=pre_epoch + 1,
    #             stats=pre_stats,
    #             include_test=True,
    #             multiclass=True
    #         )

    #     model.features.load_state_dict(pre_model.features.state_dict())
    #     print("Copied pretrained feature weights into 2-class challenge model.")

    #     # stats for fine-tuning phase
    #     stats = []
    # else:
    #     print(f"Checkpoint found. Resuming fine-tuning from epoch {start_epoch}.")

    # print("Fine-tuning parameters:",
    #       sum(p.numel() for p in model.parameters() if p.requires_grad))


    # # -------- fine tune with early stopping ---------

    # for param in finetune_model.features.parameters():
    #     param.requires_grad = False

    # TODO: define loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    axes = make_training_plot("Challenge CNN Training with transfer")

    if len(stats) == 0:
        print("No saved model parameters found, starting from scratch.")
        evaluate_epoch(
            axes,
            full_loader,
            full_loader,
            full_loader,
            model,
            criterion,
            epoch=0,
            stats=stats,
            include_test=True,
        )
    else:
        print(f"Restored checkpoint at epoch {start_epoch}.")


    prev_val_loss = stats[0][1]
    # patience = 20
    # curr_patience = 0
    epoch = start_epoch

    # while curr_patience < patience:
    while epoch < 37:
        print(f"Epoch {epoch + 1}")

        # One epoch of training
        train_epoch(full_loader, model, criterion, optimizer)

        # Evaluate on train/val/test
        evaluate_epoch(
            axes,
            full_loader,
            full_loader,
            full_loader,
            model,
            criterion,
            epoch=epoch + 1,
            stats=stats,
            include_test=True,
        )

        save_checkpoint(
            model,
            epoch + 1,
            checkpoint_dir=config("challenge.checkpoint"),
            stats=stats,
        )

        # curr_patience, prev_val_loss = early_stopping(stats, curr_patience, prev_val_loss)
        epoch += 1

    print("Finished Training")

    print("Finished Training")
    # Save figure and keep plot open
    plt.savefig("challenge_training_plot.png", dpi=200)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
