import torch
import random
from torchvision.transforms.v2 import functional as F
from config import IMAGE_SIZE


def train_transform_lung(img, target):
    orig_w, orig_h = img.size

    if random.random() > 0.3:
        angle = random.uniform(-15, 15)
        img = F.rotate(img, angle)
        target["masks"] = F.rotate(target["masks"], angle)

    if random.random() > 0.5:
        img = F.hflip(img)
        target["masks"] = F.hflip(target["masks"])
        boxes = target["boxes"]
        width = img.size[0]
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        target["boxes"] = boxes

    if random.random() > 0.3:
        img = F.adjust_brightness(img, random.uniform(0.8, 1.2))
    if random.random() > 0.3:
        img = F.adjust_contrast(img, random.uniform(0.8, 1.2))

    if random.random() > 0.5:
        noise = torch.randn_like(F.to_tensor(img)) * 0.03
        img_tensor = F.to_tensor(img) + noise
        img = F.to_pil_image(img_tensor.clamp(0, 1))

    img = F.resize(img, IMAGE_SIZE, antialias=True)
    target["masks"] = F.resize(target["masks"], IMAGE_SIZE)

    scale_w = IMAGE_SIZE[0] / orig_w
    scale_h = IMAGE_SIZE[1] / orig_h
    boxes = target["boxes"]
    boxes[:, [0, 2]] *= scale_w
    boxes[:, [1, 3]] *= scale_h
    target["boxes"] = boxes

    img = F.to_tensor(img)
    img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return img, target


def val_transform_lung(img, target):
    orig_w, orig_h = img.size

    img = F.resize(img, IMAGE_SIZE, antialias=True)
    target["masks"] = F.resize(target["masks"], IMAGE_SIZE)

    scale_w = IMAGE_SIZE[0] / orig_w
    scale_h = IMAGE_SIZE[1] / orig_h
    boxes = target["boxes"]
    boxes[:, [0, 2]] *= scale_w
    boxes[:, [1, 3]] *= scale_h
    target["boxes"] = boxes

    img = F.to_tensor(img)
    img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return img, target


def transform_image(img):
    img = F.resize(img, IMAGE_SIZE, antialias=True)
    img = F.to_tensor(img)
    img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img
