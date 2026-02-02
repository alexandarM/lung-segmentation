import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def collate_fn(batch):
    return tuple(zip(*batch))


def move_to_device(images, targets, device):
    imgs = [img.to(device) for img in images]
    targs = []
    for t in targets:
        new_t = {}
        for k, v in t.items():
            if isinstance(v, torch.Tensor):
                new_t[k] = v.to(device)
            else:
                new_t[k] = v
        targs.append(new_t)
    return imgs, targs


def evaluate_segmentation(model, data_loader, device, score_threshold=0.5):
    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = move_to_device(images, targets, device)

            predictions = model(images)

            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            model.eval()

            for pred, target in zip(predictions, targets):
                if len(target["masks"]) > 0:
                    if len(pred["scores"]) > 0:
                        best_idx = pred["scores"].argmax()
                        best_score = pred["scores"][best_idx]

                        if best_score > score_threshold:
                            pred_mask = (pred["masks"][best_idx, 0] > 0.5).float()
                            true_mask = target["masks"][0].float()

                            intersection = (pred_mask * true_mask).sum()
                            union = pred_mask.sum() + true_mask.sum()
                            dice = (2. * intersection) / (union + 1e-6)

                            iou = intersection / ((pred_mask + true_mask - pred_mask * true_mask).sum() + 1e-6)

                            precision = intersection / (pred_mask.sum() + 1e-6)
                            recall = intersection / (true_mask.sum() + 1e-6)
                            f1 = (2. * precision * recall) / (precision + recall + 1e-6)

                            total_dice += dice.item()
                            total_iou += iou.item()
                            total_f1 += f1.item()
                        else:
                            total_dice += 0.0
                            total_iou += 0.0
                            total_f1 += 0.0
                    else:
                        total_dice += 0.0
                        total_iou += 0.0
                        total_f1 += 0.0

                    num_samples += 1

    avg_loss = total_loss / len(data_loader)
    avg_dice = total_dice / num_samples if num_samples > 0 else 0.0
    avg_iou = total_iou / num_samples if num_samples > 0 else 0.0
    avg_f1 = total_f1 / num_samples if num_samples > 0 else 0.0

    return avg_loss, avg_dice, avg_iou, avg_f1


def train_one_epoch_with_metrics(model, train_loader, optimizer, device, score_threshold=0.5):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc=" Training", leave=False)

    for images, targets in progress_bar:
        images, targets = move_to_device(images, targets, device)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            preds = model(images)
            model.train()

            batch_dice = 0.0
            batch_iou = 0.0
            batch_f1 = 0.0
            batch_count = 0

            for pred, target in zip(preds, targets):
                if len(target["masks"]) > 0 and len(pred["scores"]) > 0:
                    best_idx = pred["scores"].argmax()
                    best_score = pred["scores"][best_idx]

                    if best_score > score_threshold:
                        pred_mask = (pred["masks"][best_idx, 0] > 0.5).float()
                        true_mask = target["masks"][0].float()

                        intersection = (pred_mask * true_mask).sum()
                        union = pred_mask.sum() + true_mask.sum()
                        dice = (2. * intersection) / (union + 1e-6)

                        iou = intersection / ((pred_mask + true_mask - pred_mask * true_mask).sum() + 1e-6)

                        precision = intersection / (pred_mask.sum() + 1e-6)
                        recall = intersection / (true_mask.sum() + 1e-6)
                        f1 = (2. * precision * recall) / (precision + recall + 1e-6)

                        batch_dice += dice.item()
                        batch_iou += iou.item()
                        batch_f1 += f1.item()
                        batch_count += 1

            if batch_count > 0:
                total_dice += batch_dice / batch_count
                total_iou += batch_iou / batch_count
                total_f1 += batch_f1 / batch_count

        num_batches += 1
        progress_bar.set_postfix({"Loss": f"{losses.item():.4f}"})

    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / num_batches if num_batches > 0 else 0.0
    avg_iou = total_iou / num_batches if num_batches > 0 else 0.0
    avg_f1 = total_f1 / num_batches if num_batches > 0 else 0.0

    return avg_loss, avg_dice, avg_iou, avg_f1


def plot_training_metrics(train_losses, val_losses, train_dices, val_dices, train_ious, val_ious, train_f1s=None, val_f1s=None):
    if train_f1s is not None and val_f1s is not None:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_dices, label='Train Dice')
    ax2.plot(val_dices, label='Val Dice')
    ax2.set_title('Dice Coefficient')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(train_ious, label='Train IoU')
    ax3.plot(val_ious, label='Val IoU')
    ax3.set_title('IoU')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('IoU')
    ax3.legend()
    ax3.grid(True)

    if train_f1s is not None and val_f1s is not None:
        ax4.plot(train_f1s, label='Train F1')
        ax4.plot(val_f1s, label='Val F1')
        ax4.set_title('F1 Score')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('F1')
        ax4.legend()
        ax4.grid(True)

    plt.tight_layout()
    plt.show()
