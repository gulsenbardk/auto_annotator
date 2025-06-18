import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch.nn as nn


# ====== Model Definition ======
class PointNetSeg(nn.Module):
    def __init__(self, num_classes):
        super(PointNetSeg, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU()
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, num_classes, 1)
        )

    def forward(self, x):
        B, N, _ = x.shape
        x = x.transpose(1, 2)
        feat = self.mlp1(x)
        global_feat = torch.max(feat, 2, keepdim=True)[0]
        global_feat = global_feat.repeat(1, 1, N)
        concat = torch.cat([feat, global_feat], 1)
        out = self.mlp2(concat)
        return out.transpose(1, 2)


# ====== Helper Functions ======
def get_colormap(num_classes=21):
    cmap = plt.get_cmap('tab20', num_classes)
    return [cmap(i)[:3] for i in range(num_classes)]


def visualize_pointcloud(points, gt_labels, pred_labels, colormap):
    points = points.cpu().numpy().astype(np.float64)
    gt_labels = gt_labels.cpu().numpy().astype(np.int32)
    pred_labels = pred_labels.cpu().numpy().astype(np.int32)

    gt_colors = np.array([colormap[l % len(colormap)] for l in gt_labels], dtype=np.float64)
    pred_colors = np.array([colormap[l % len(colormap)] for l in pred_labels], dtype=np.float64)
    pred_points = points + np.array([5.0, 0.0, 0.0])

    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(points)
    gt_pcd.colors = o3d.utility.Vector3dVector(gt_colors)

    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_points)
    pred_pcd.colors = o3d.utility.Vector3dVector(pred_colors)

    print("Left: Ground Truth | Right: Prediction")
    o3d.visualization.draw_geometries([gt_pcd, pred_pcd])


def compute_metrics(preds, labels, num_classes):
    preds = preds.flatten()
    labels = labels.flatten()
    valid = (labels >= 0) & (labels < num_classes)
    preds = preds[valid]
    labels = labels[valid]

    conf_matrix = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    intersection = np.diag(conf_matrix)
    union = conf_matrix.sum(1) + conf_matrix.sum(0) - intersection
    iou = intersection / (union + 1e-6)
    mean_iou = np.mean(iou)
    overall_acc = np.sum(intersection) / np.sum(conf_matrix)

    print("\nEvaluation Metrics:")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    for i, iou_val in enumerate(iou):
        print(f"  Class {i}: IoU = {iou_val:.4f}")


# ====== Main Evaluation Script ======
def main():
    # --- Config ---
    model_path = "pointnet_seg.pth"  # Replace if needed
    data_path = "/mnt/data/bard_gu/pcd/Datasets/SemanticKITTI/dataset/sequences/01/train"
    num_classes = 21
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load Data ---
    data = torch.load(data_path)
    points = data['points'].float().unsqueeze(0).to(device)  # [1, N, 3]
    labels = data['labels'].long().unsqueeze(0).to(device)   # [1, N]

    # --- Load Model ---
    model = PointNetSeg(num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # --- Inference ---
    with torch.no_grad():
        outputs = model(points)
        preds = outputs.argmax(dim=2)

    # --- Evaluation ---
    compute_metrics(preds.cpu().numpy(), labels.cpu().numpy(), num_classes)

    # --- Visualization ---
    colormap = get_colormap(num_classes)
    visualize_pointcloud(points[0], labels[0], preds[0], colormap)


if __name__ == '__main__':
    main()
