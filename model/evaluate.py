"""
Evaluate the trained model on the test set (624 images) -- a much more
reliable performance estimate than the tiny 16-image official val split.

Usage:
    python evaluate.py --data_dir ../data --model_path ../saved_models/best_model.pt
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import build_model


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model_path, map_location=device)
    classes = checkpoint["classes"]

    model = build_model(num_classes=len(classes), freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_ds = datasets.ImageFolder(os.path.join(args.data_dir, "test"), transform=tfms)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    print(f"Evaluating on {len(test_ds)} test images. Classes: {classes}")

    tp = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for p, l in zip(preds.tolist(), labels.tolist()):
                pred_name, true_name = classes[p], classes[l]
                if p == l:
                    tp[pred_name] += 1
                else:
                    fp[pred_name] += 1
                    fn[true_name] += 1

    accuracy = correct / total
    print(f"\nOverall Test Accuracy: {accuracy:.4f} ({correct}/{total})\n")

    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    for c in classes:
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"{c:<12} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--model_path", type=str, default="../saved_models/best_model.pt")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    evaluate(args)