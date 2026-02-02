import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import *
from dataset import LungDataset
from transforms import train_transform_lung, val_transform_lung
from model import get_model
from utils import collate_fn, evaluate_segmentation, train_one_epoch_with_metrics, plot_training_metrics


def train_model(valid_pairs, num_epochs=NUM_EPOCHS, patience=7):
    train_pairs, temp_pairs = train_test_split(
        valid_pairs, test_size=0.3, random_state=42
    )

    val_pairs, test_pairs = train_test_split(
        temp_pairs, test_size=0.5, random_state=42
    )

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    train_dataset = LungDataset(train_pairs, transform=train_transform_lung)
    val_dataset = LungDataset(val_pairs, transform=val_transform_lung)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    model = get_model(NUM_CLASSES, dropout=0.3)
    model.to(DEVICE)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_val_dice = 0.0
    epochs_no_improve = 0
    early_stop = False

    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    train_ious, val_ious = [], []
    train_f1s, val_f1s = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        train_loss, train_dice, train_iou, train_f1 = train_one_epoch_with_metrics(model, train_loader, optimizer, DEVICE)
        val_loss, val_dice, val_iou, val_f1 = evaluate_segmentation(model, val_loader, DEVICE)

        scheduler.step(val_dice)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        train_ious.append(train_iou)
        val_ious.append(val_iou)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f}")
        print(f"Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f}")
        print(f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        if val_dice > best_val_dice + 1e-5:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "best_lung_segmentation_model.pth")
            print("Best model saved based on Dice!")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in Dice for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs (no Dice improvement).")
            early_stop = True
            break

    if not early_stop:
        print("\nTraining finished without early stopping.")

    print(f"\nBest Validation Dice: {best_val_dice:.4f}")

    plot_training_metrics(train_losses, val_losses, train_dices, val_dices, train_ious, val_ious, train_f1s, val_f1s)

    return model, test_pairs


if __name__ == "__main__":
    from data_preparation import prepare_data
    
    valid_pairs = prepare_data()
    model, test_pairs = train_model(valid_pairs)
