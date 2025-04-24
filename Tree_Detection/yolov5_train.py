# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score
import numpy as np
import os
import sys

# Add YOLOv5 repository to Python path
sys.path.append(os.path.abspath('.'))  # Assumes script is run from yolov5/ directory

from utils.torch_utils import select_device
from utils.callbacks import Callbacks
from train import train

# Define paths and hyperparameters
model_path = "yolov5n-cls.pt"  # Pretrained YOLOv5n classification model
data_path = "/home/bygpu/Desktop/dataset"
data_yaml = "/home/bygpu/Desktop/dataset/data.yaml"
epochs = 100
imgsz = 224
batch_size = 16
project = "runs/train_yolov5n"
name = "test"
workers = 4

# Create data.yaml if it doesn¡¯t exist
if not os.path.exists(data_yaml):
    with open(data_yaml, 'w', encoding='utf-8') as f:
        f.write(f"""
train: {os.path.join(data_path, 'train')}
val: {os.path.join(data_path, 'val')}
nc: 2
names: ['class1', 'class2']  # Replace with your class names
""")

# Initialize device
device = select_device('')  # Automatically select GPU if available

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model = model.to(device)

# Define transforms for validation dataset (matching YOLOv5 preprocessing)
transform = transforms.Compose([
    transforms.Resize((imgsz, imgsz)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])  # YOLOv5 uses [0,1] range
])

# Load validation dataset
val_dataset = ImageFolder(root=os.path.join(data_path, "val"), transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

# Initialize log file
log_file = os.path.join(project, name, "metrics_log.txt")
os.makedirs(os.path.dirname(log_file), exist_ok=True)
with open(log_file, 'w', encoding='utf-8') as f:
    f.write("Training Metrics Log\n")
    f.write("====================\n")

# Initialize callbacks
callbacks = Callbacks()

# Track best model
best_accuracy = 0.0

# Training and validation loop
for epoch in range(epochs):
    # Train for one epoch
    train(
        hyp='data/hyps/hyp.scratch-low.yaml',  # Default hyperparameters
        opt={
            'epochs': 1,  # Train one epoch at a time
            'batch_size': batch_size,
            'imgsz': imgsz,
            'workers': workers,
            'project': project,
            'name': name,
            'exist_ok': True,
            'weights': model_path,
            'data': data_yaml,
            'single_cls': False,
            'verbose': True,
            'save_period': -1,  # Disable auto-saving
            'cache': False,  # Disable disk caching
            'device': device,  # Pass device in opt
        },
        device=device,  # Pass device explicitly
        callbacks=callbacks  # Pass callbacks explicitly
    )

    # Perform custom validation
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions
            outputs = model(images)  # YOLOv5 returns a tensor of logits
            preds = torch.argmax(outputs, dim=1)  # Get predicted classes
            
            # Compute accuracy
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = 100 * correct / total if total > 0 else 0.0  # Accuracy in percentage
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Print metrics
    print(f"Epoch {epoch + 1}/{epochs}:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Log metrics to file
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"Epoch {epoch + 1}/{epochs}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write("--------------------\n")

    # Save model if this epoch has the best accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), os.path.join(project, name, "weights", "best.pt"))
        print(f"Best model saved with accuracy: {accuracy:.2f}%")

print("Training completed!")
print(f"Results and logs saved in: {project}/{name}")
print(f"Metrics log: {log_file}")