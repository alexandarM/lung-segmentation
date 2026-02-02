import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors


class LungDataset(Dataset):
    def __init__(self, img_mask_pairs, transform=None):
        self.img_mask_pairs = img_mask_pairs
        self.transform = transform

    def __len__(self):
        return len(self.img_mask_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.img_mask_pairs[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        mask_array = np.array(mask)
        mask_tensor = torch.as_tensor(mask_array, dtype=torch.uint8)
        binary_mask = (mask_tensor > 0).to(torch.uint8)

        if binary_mask.sum() > 0:
            masks = binary_mask.unsqueeze(0)
        else:
            masks = torch.zeros((0, mask_array.shape[0], mask_array.shape[1]), dtype=torch.uint8)

        boxes = masks_to_boxes(masks)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]
        masks = masks[valid]

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, mask_array.shape[0], mask_array.shape[1]), dtype=torch.uint8)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            labels_tensor = torch.ones((boxes.shape[0],), dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(mask_array.shape[0], mask_array.shape[1])),
            "masks": tv_tensors.Mask(masks),
            "labels": labels_tensor,
            "image_id": idx,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target
