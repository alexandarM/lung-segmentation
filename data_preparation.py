import shutil
import random
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from config import IMG_ROOT, MASK_ROOT


def prepare_data():
    no_mask_folder = Path("Data_bdbm/Data_bdbm/no_mask")
    no_mask_folder.mkdir(exist_ok=True)

    images = sorted(IMG_ROOT.glob("*.png"))
    masks = sorted(MASK_ROOT.glob("*.png"))

    mask_dict = {m.stem.replace("_mask", ""): m for m in masks}

    valid_pairs = []
    no_mask_pairs = []

    for img in images:
        key = img.stem
        if key in mask_dict:
            mask_path = mask_dict[key]
            img_size = Image.open(img).size
            mask_size = Image.open(mask_path).size
            if img_size == mask_size:
                valid_pairs.append((img, mask_path))
            else:
                no_mask_pairs.append((img, mask_path))
                shutil.move(str(img), no_mask_folder / img.name)
                print(f"Dimenzije se ne poklapaju, premještam {img.name}")
        else:
            no_mask_pairs.append((img, None))
            shutil.move(str(img), no_mask_folder / img.name)
            print(f"Nema masku, premještam {img.name}")

    print(f"Pronađeno {len(valid_pairs)} validnih parova")
    print(f"{len(no_mask_pairs)} slika bez maski premješteno u {no_mask_folder}")

    return valid_pairs


def visualize_samples(valid_pairs, num_samples=5):
    num_samples = min(num_samples, len(valid_pairs))
    sample_pairs = random.sample(valid_pairs, num_samples)

    plt.figure(figsize=(15, 6))
    for i, (img_path, mask_path) in enumerate(sample_pairs):
        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        plt.subplot(2, num_samples, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"RTG {img_path.stem}")
        plt.axis("off")

        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(mask, cmap="gray")
        plt.title("Maska pluća")
        plt.axis("off")

    plt.suptitle("Nasumični RTG snimci i njihove maske", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    valid_pairs = prepare_data()
    visualize_samples(valid_pairs)
