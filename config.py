import torch
from pathlib import Path

IMG_ROOT = Path("Data_bdbm/Data_bdbm/images")
MASK_ROOT = Path("Data_bdbm/Data_bdbm/masks")

NUM_CLASSES = 2
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 5e-5

IMAGE_SIZE = (256, 256)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {1: "lung"}
CLASS_COLORS = {1: '#00FF00'}
