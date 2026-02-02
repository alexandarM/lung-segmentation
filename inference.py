import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image

from config import DEVICE, IMAGE_SIZE, NUM_CLASSES
from model import get_model
from transforms import transform_image


def load_trained_model(model_path="best_lung_segmentation_model.pth"):
    model = get_model(NUM_CLASSES, dropout=0.3)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict_single_image(model, image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform_image(img)

    with torch.no_grad():
        prediction = model([img_tensor.to(DEVICE)])[0]

    return img_tensor, prediction


def visualize_prediction(img_tensor, prediction, image_path):
    img_display = img_tensor.cpu().permute(1, 2, 0).numpy()
    img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_display = np.clip(img_display, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_display)
    axes[0].set_title("Originalna slika")
    axes[0].axis('off')

    axes[1].imshow(img_display)
    if len(prediction['masks']) > 0:
        best_idx = prediction['scores'].argmax()
        pred_mask = prediction['masks'][best_idx, 0].cpu().numpy() > 0.5
        confidence = prediction['scores'][best_idx].cpu().numpy()
        axes[1].imshow(pred_mask, alpha=0.5, cmap='jet')
        axes[1].set_title(f"Predikcija (Conf: {confidence:.3f})")
        if len(prediction['boxes']) > 0:
            box = prediction['boxes'][best_idx].cpu().numpy()
            rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[1].add_patch(rect)
    else:
        axes[1].set_title("Nema detekcije", color="red")
    axes[1].axis('off')

    axes[2].imshow(img_display)
    if len(prediction['masks']) > 0:
        axes[2].imshow(pred_mask, alpha=0.7, cmap='viridis')
        axes[2].set_title("Samo maska segmentacije")
    else:
        axes[2].set_title("Nema segmentacije", color="red")
    axes[2].axis('off')

    plt.suptitle(f"Analiza segmentacije: {Path(image_path).name}", fontsize=16)
    plt.tight_layout()
    plt.show()

    print("\nDETALJNA ANALIZA")
    print("-"*40)
    if len(prediction['masks']) > 0:
        print(f"Pronađena segmentacija pluća")
        print(f"   Broj detekcija: {len(prediction['masks'])}")
        print(f"   Konfidencija najbolje detekcije: {confidence:.4f}")
        mask_area = np.sum(pred_mask)
        total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
        coverage_percentage = (mask_area / total_pixels) * 100
        print(f"   Pokrivenost pluća: {coverage_percentage:.2f}% slike")
        print(f"   Raspon skorova: {prediction['scores'].min().cpu().numpy():.4f} - {prediction['scores'].max().cpu().numpy():.4f}")
    else:
        print("Nema segmentacije pluća na slici.")


def predict_batch(model, image_folder):
    test_folder = Path(image_folder)
    if not test_folder.exists():
        raise FileNotFoundError(f"Folder {image_folder} ne postoji.")

    test_images = sorted(test_folder.glob("*.png"))
    if len(test_images) == 0:
        raise FileNotFoundError("Nema PNG slika u folderu.")

    print(f"Pronađeno {len(test_images)} test slika.\n")

    for idx, img_path in enumerate(test_images):
        print("="*60)
        print(f"Testiranje slike {idx+1}/{len(test_images)}: {img_path.name}")
        print("="*60)

        img_tensor, prediction = predict_single_image(model, img_path)
        visualize_prediction(img_tensor, prediction, img_path)
        print()


if __name__ == "__main__":
    model = load_trained_model()
    
    image_path = "path/to/your/image.png"
    img_tensor, prediction = predict_single_image(model, image_path)
    visualize_prediction(img_tensor, prediction, image_path)
