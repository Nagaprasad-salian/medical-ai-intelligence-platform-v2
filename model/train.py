"""
Training script for Chest X-Ray Pneumonia Classification.

Dataset expected structure (Kaggle "Chest X-Ray Images (Pneumonia)"):
    data/
        train/
            NORMAL/
            PNEUMONIA/
        val/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/

Usage:
    python train.py --data_dir ../data --epochs 10 --batch_size 32
"""

import argparse
import os
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def get_dataloaders(data_dir, batch_size, img_size=224):
    train_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    eval_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
    val_dir = os.path.join(data_dir, "val")
    if not os.path.isdir(val_dir) or len(os.listdir(val_dir)) == 0:
        # some versions of this dataset ship a tiny/empty val split -> carve one out of train
        val_dir = os.path.join(data_dir, "test")
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_ds.classes


def build_model(num_classes=2, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    # Replace final FC layer -- this stays trainable regardless
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, classes = get_dataloaders(args.data_dir, args.batch_size, args.img_size)
    print(f"Classes: {classes}")

    model = build_model(num_classes=len(classes), freeze_backbone=args.freeze_backbone).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total

        # validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "val_acc": val_acc,
                "img_size": args.img_size,
            }, os.path.join(args.output_dir, "best_model.pt"))
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(args.output_dir, "classes.json"), "w") as f:
        json.dump(classes, f)

    print(f"Training complete. Best val_acc={best_val_acc:.4f}")
    print(f"Model saved to {os.path.join(args.output_dir, 'best_model.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="../saved_models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    args = parser.parse_args()

    train(args)
