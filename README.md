# Lung Segmentation with Mask R-CNN

Deep learning project for automatic lung segmentation from chest X-ray images using Mask R-CNN with ResNet50-FPN backbone.

## Live Demo

**Want to try the model without installing anything?**

**[Try it now on Hugging Face Space](https://huggingface.co/spaces/lxndr1337/segmentacija_pluca)** 

Upload your chest X-ray image and get instant lung segmentation results!

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/lxndr1337/segmentacija_pluca)

> **Note**: The Space may take 10-15 seconds to wake up if it's been inactive. This is normal behavior for free tier Spaces.

## Project Structure

```
.
├── config.py                 # Configuration and hyperparameters
├── dataset.py               # Custom Dataset class for lung images
├── transforms.py            # Image augmentation and transformations
├── model.py                 # Model architecture definition
├── utils.py                 # Helper functions for training and evaluation
├── data_preparation.py      # Data loading and preprocessing
├── train.py                 # Training script
├── inference.py             # Inference and visualization
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## Features

- Mask R-CNN architecture with ResNet50-FPN backbone
- Custom data augmentation pipeline for medical images
- Training with Dice coefficient, IoU, and F1 metrics
- Early stopping and learning rate scheduling
- Comprehensive evaluation and visualization tools

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Organize your data in the following structure:
```
Data_bdbm/
├── images/
│   └── *.png
└── masks/
    └── *_mask.png
```

2. Run data preparation:
```bash
python data_preparation.py
```

## Training

Configure hyperparameters in `config.py`, then run:

```bash
python train.py
```

Training parameters:
- Batch size: 8
- Epochs: 30
- Learning rate: 5e-5
- Weight decay: 5e-5
- Image size: 256x256

## Inference

Run inference on new images:

```python
from inference import load_trained_model, predict_single_image, visualize_prediction

model = load_trained_model("best_lung_segmentation_model.pth")
img_tensor, prediction = predict_single_image(model, "path/to/image.png")
visualize_prediction(img_tensor, prediction, "path/to/image.png")
```

For batch inference:
```python
from inference import predict_batch

predict_batch(model, "path/to/image/folder")
```

## Model Architecture

- Backbone: ResNet50 with Feature Pyramid Network (FPN)
- Head: Mask R-CNN with custom box and mask predictors
- Classes: 2 (background + lungs)
- Dropout: 0.3 for regularization

## Evaluation Metrics

- Dice Coefficient
- Intersection over Union (IoU)
- F1 Score
- Segmentation Loss

## Results

The model is evaluated on validation set with the following metrics:
- Dice coefficient tracking for model selection
- Early stopping based on Dice improvement
- Comprehensive visualization of predictions vs ground truth

## Acknowledgments

- PyTorch and torchvision for deep learning framework
- Mask R-CNN implementation from torchvision
- Hugging Face Space for hosting the demo application

### Hugging Face Space

The model is deployed as an interactive Gradio application on Hugging Face Spaces:
- **Live Demo**: https://huggingface.co/spaces/lxndr1337/segmentacija_pluca
- **Framework**: Gradio
- **Backend**: Python with PyTorch
- **Features**: Real-time lung segmentation with visualization

## License

MIT License