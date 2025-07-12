from header import * 



def evaluate(model, dataloader, device, return_preds=True):
    model.eval()
    total_correct = 0
    total_points = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for points, labels in dataloader:
            points = points.to(device)  # (B, N, 3)
            labels = labels.to(device).long()  # (B, N)

            logits = model(points)  # (B, N, C)
            preds = torch.argmax(logits, dim=-1)  # (B, N)

            total_correct += (preds == labels).sum().item()
            total_points += labels.numel()

            if return_preds:
                all_preds.extend(preds.view(-1).cpu().numpy())
                all_labels.extend(labels.view(-1).cpu().numpy())

    acc = total_correct / total_points if total_points > 0 else 0.0

    if return_preds:
        return acc, all_labels, all_preds
    return acc





def evaluate_segmentation(model, dataloader, device, num_classes=10):
    model.eval()
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for points, labels in dataloader:
            points = points.to(device)           # (B, N, 3)
            labels = labels.to(device).long()    # (B, N)

            #points = normalize_points(points)

            logits = model(points)               # (B, N, num_classes)
            preds = torch.argmax(logits, dim=-1) # (B, N)
            #import ipdb 
            #ipdb.set_trace()
            print("Unique preds:", torch.unique(preds))
            print("Unique gts:  ", torch.unique(labels))

            all_preds.append(preds.view(-1))
            all_gts.append(labels.view(-1))


    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)

    print("Unique preds:", torch.unique(all_preds))
    print("Unique gts:  ", torch.unique(all_gts))
    
    # Confusion matrix using CPU for display
    cm = confusion_matrix(all_gts.cpu().numpy(), all_preds.cpu().numpy(), labels=list(range(num_classes)))
    print("Confusion Matrix (point-level):\n", cm)

    # Classification report for detailed inspection
    print("\nSegmentation Report (per point):\n", classification_report(
        all_gts.cpu().numpy(), all_preds.cpu().numpy(), labels=list(range(num_classes)), digits=4
    ))

    # Compute torchmetrics-based scores
    acc = accuracy(all_preds, all_gts, task="multiclass", num_classes=num_classes).item()
    prec = precision(all_preds, all_gts, average='macro', num_classes=num_classes).item()
    rec = recall(all_preds, all_gts, average='macro', num_classes=num_classes).item()
    f1 = f1_score(all_preds, all_gts, average='macro', num_classes=num_classes).item()
    ious = jaccard_index(all_preds, all_gts, task="multiclass", num_classes=num_classes)  # shape: (num_classes,)
    miou = ious.mean().item()

    print(f"IoU per class: {[f'{iou:.4f}' for iou in ious.tolist()]}")
    print(f"Mean IoU (mIoU): {miou:.4f}")

    return acc, prec, rec, f1, ious.tolist(), miou, cm



def evaluate_pointcloud_segmentation(model, val_loader, num_classes=10, device="cuda"):
    model.eval()

    total_correct = 0
    total_seen = 0
    class_correct = torch.zeros(num_classes).to(device)
    class_seen = torch.zeros(num_classes).to(device)
    iou_inter = torch.zeros(num_classes).to(device)
    iou_union = torch.zeros(num_classes).to(device)

    with torch.no_grad():
        for input_data, labels in tqdm(val_loader, desc="Evaluating"):
            input_data = input_data.to(device)  # (B, N, F)
            labels = labels.to(device)          # (B, N)

            # Normalize per sample (across point dimension)
            input_data = (input_data - input_data.mean(dim=1, keepdim=True)) / \
                         (input_data.std(dim=1, keepdim=True) + 1e-5)

            logits = model(input_data)          # (B, N, C)
            preds = logits.argmax(dim=2)        # (B, N)

            total_correct += (preds == labels).sum().item()
            total_seen += labels.numel()

            for cls in range(num_classes):
                cls_mask = (labels == cls)
                class_seen[cls] += cls_mask.sum()
                class_correct[cls] += ((preds == cls) & cls_mask).sum()
                iou_inter[cls] += ((preds == cls) & (labels == cls)).sum()
                iou_union[cls] += ((preds == cls) | (labels == cls)).sum()

    overall_acc = total_correct / total_seen
    class_acc = (class_correct / (class_seen + 1e-6)).mean().item()
    iou = (iou_inter / (iou_union + 1e-6))
    mean_iou = iou.mean().item()

    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Mean Class Accuracy: {class_acc:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    return overall_acc, class_acc, mean_iou


import torch
from tqdm import tqdm

def validate(model, val_loader, num_classes=10, device='cuda'):
    model.eval()
    total_correct = 0
    total_seen = 0
    class_correct = torch.zeros(num_classes).to(device)
    class_seen = torch.zeros(num_classes).to(device)
    iou_inter = torch.zeros(num_classes).to(device)
    iou_union = torch.zeros(num_classes).to(device)

    with torch.no_grad():
        for input_data, labels in tqdm(val_loader, desc="Validation"):
            input_data = input_data.to(device)
            labels = labels.to(device)

            # Normalize input same as training
            input_data = (input_data - input_data.mean(dim=1, keepdim=True)) / \
                         (input_data.std(dim=1, keepdim=True) + 1e-5)

            logits = model(input_data)  # (B, N, C) where N = number of points
            preds = logits.argmax(dim=2)  # (B, N)

            total_correct += (preds == labels).sum().item()
            total_seen += labels.numel()

            for cls in range(num_classes):
                cls_mask = (labels == cls)
                class_seen[cls] += cls_mask.sum()
                class_correct[cls] += ((preds == cls) & cls_mask).sum()
                iou_inter[cls] += ((preds == cls) & cls_mask).sum()
                iou_union[cls] += ((preds == cls) | cls_mask).sum()

    overall_acc = total_correct / total_seen
    mean_class_acc = (class_correct / (class_seen + 1e-6)).mean().item()
    mean_iou = (iou_inter / (iou_union + 1e-6)).mean().item()

    print(f"Validation Overall Accuracy: {overall_acc:.4f}")
    print(f"Validation Mean Class Accuracy: {mean_class_acc:.4f}")
    print(f"Validation Mean IoU: {mean_iou:.4f}")

    return overall_acc, mean_class_acc, mean_iou


from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(predictions, true_labels, num_classes=10, ignore_index=0):
    # Flatten predictions and labels
    all_preds = torch.cat([p.reshape(-1).cpu() for p in predictions], dim=0)
    all_labels = torch.cat([l.reshape(-1).cpu() for l in true_labels], dim=0)

    # Remove ignored labels
    valid_mask = all_labels != ignore_index
    all_preds = all_preds[valid_mask]
    all_labels = all_labels[valid_mask]

    print("\n=== Classification Report (excluding ignore_index) ===")
    print(classification_report(
        all_labels,
        all_preds,
        labels=list(range(1, num_classes)),  # skip class 0
        zero_division=0,  # avoid warnings for empty predictions
        digits=4
    ))

    print("\n=== Confusion Matrix (excluding ignore_index) ===")
    print(confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(1, num_classes))
    ))