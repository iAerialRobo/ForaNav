# -*- coding: utf-8 -*-
from ultralytics import YOLO
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score
import torch
import numpy as np
import os

# Define paths and hyperparameters (matching your CLI command)
model_path = "yolov8n-cls.pt"
data_path = "/home/bygpu/Desktop/dataset"
epochs = 30
imgsz = 224
batch_size = 16
project = "runs/train_yolov8n"
name = "test"
workers = 4

# Initialize the model
model = YOLO(model_path)

# Define transforms for validation dataset (matching YOLOv8 preprocessing)
transform = transforms.Compose([
    transforms.Resize((imgsz, imgsz)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])  # YOLOv8 uses [0,1] range
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

# Track best model
best_accuracy = 0.0

# Train the model
for epoch in range(epochs):
    # Train for one epoch
    model.train(
        data=data_path,
        epochs=1,  # Train one epoch at a time
        imgsz=imgsz,
        batch=batch_size,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
        workers=workers,
        save=False  # Save manually after validation
    )

    # Perform validation
    model.model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(model.device)
            labels = labels.to(model.device)
            
            # Get predictions using model.predict
            results = model.predict(images, verbose=False)
            
            # Extract probability tensor and get predicted classes
            batch_preds = []
            for result in results:
                probs = result.probs.data  # Get the probability tensor
                pred = torch.argmax(probs, dim=0)  # Get the predicted class (scalar)
                batch_preds.append(pred)
            
            batch_preds = torch.tensor(batch_preds, device=model.device)
            
            all_preds.extend(batch_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Run official validation for accuracy
    results = model.val(data=data_path, imgsz=imgsz, batch=batch_size, workers=workers, verbose=False)

    # Compute metrics
    accuracy = results.top1 * 100  # Top-1 accuracy in percentage
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
        model.save(os.path.join(project, name, "weights", "best.pt"))
        print(f"Best model saved with accuracy: {accuracy:.2f}%")

print("Training completed!")
print(f"Results and logs saved in: {project}/{name}")
print(f"Metrics log: {log_file}")
