from train import train 
from dataset import * 
from utils import *
from models import *
from evaluation import *
from loss import *
from visualization import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from header import * 

def save_point_clouds(points, labels, est_labels):
    # Convert tensors to numpy arrays
    points_np = points.detach().cpu().numpy()
    preds_np = labels.detach().cpu().numpy()
    gts_np = est_labels.detach().cpu().numpy()

    # Get all unique labels from both predictions and ground truth
    all_labels = np.concatenate([preds_np, gts_np])
    unique_labels = np.unique(all_labels)

    # Create a consistent color map
    cmap = plt.get_cmap('tab20')
    label_to_color = {label: np.array(cmap(i % 20)[:3]) for i, label in enumerate(unique_labels)}

    def create_o3d_pc(points, labels, label_to_color):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        colors = np.array([label_to_color[label] for label in labels])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    # Use the same color mapping for both predicted and ground truth labels
    pred_pcd = create_o3d_pc(points_np, preds_np, label_to_color)
    gt_pcd = create_o3d_pc(points_np, gts_np, label_to_color)

    # Save the point clouds
    o3d.io.write_point_cloud("predictedv2.ply", pred_pcd, write_ascii=True)
    o3d.io.write_point_cloud("ground_truthv2.ply", gt_pcd, write_ascii=True)


def save_colored_by_accuracy(points, predicted_labels, ground_truth_labels):
    # Convert tensors to numpy arrays
    points_np = points.detach().cpu().numpy()
    preds_np = predicted_labels.detach().cpu().numpy()
    gts_np = ground_truth_labels.detach().cpu().numpy()

    # Initialize point colors
    colors = np.zeros((points_np.shape[0], 3))

    # Set green for correct predictions, red for incorrect
    correct = preds_np == gts_np
    colors[correct] = [0, 1, 0]   # Green for correct
    colors[~correct] = [1, 0, 0]  # Red for incorrect

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save point cloud
    o3d.io.write_point_cloud("accuracy_colored.ply", pcd, write_ascii=True)



def save_torch_point_cloud_as_ply(points, labels, filename):
    """
    Save point cloud with labels to a .ply file.
    
    Args:
        points: Tensor of shape [B, N, 3] or [N, 3]
        labels: Tensor of shape [B, N] or [N]
        filename: Output path
    """
    # Detach and convert to numpy
    points_np = points.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # Handle batch dimension
    if points_np.ndim == 3:
        points_np = points_np.reshape(-1, 3)
        labels_np = labels_np.reshape(-1)

    cmap = plt.get_cmap('tab20')
    unique_labels = np.unique(labels_np)
    label_to_color = {label: cmap(i % 20)[:3] for i, label in enumerate(unique_labels)}
    colors = np.array([label_to_color[lbl] for lbl in labels_np])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved: {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=420)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--loss', type=str, default='entropy', choices=['focal', 'dice', 'bce', 'entropy'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_raw = PointCloudDataset(args.train_dir)
    test_raw = PointCloudDataset(args.test_dir)
    val_raw = PointCloudDataset(args.val_dir)

    train_dataset = SelectedPointDataset(train_raw)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn, drop_last=True)

    model = GeoPointNet(input_dim=6, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    weights = compute_class_weights(train_loader, num_classes=10).to(device)
    start_epoch = 0
    checkpoint_path = 'modelFocalMulti_SampledV4.pth'

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        count = 0

        for input_data, labels in train_loader:
            input_data, labels = input_data.to(device), labels.to(device)
            input_data = normalize_input(input_data)

            logits = model(input_data)
            logits = logits.reshape(-1, 10)
            labels = labels.reshape(-1)

            #loss = F.cross_entropy(logits, labels, weight=weights, ignore_index=0)
            loss_fn = FocalLoss(alpha=weights.to(device), gamma=2.0, ignore_index=0)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            count += 1
           
        avg_loss = epoch_loss / count if count > 0 else float('nan')
        print(f"Epoch {epoch + 1}/{args.epochs}, Avg Loss: {avg_loss:.4f}")
        checkpoint_path = 'modelFocalMulti_SampledV4.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)

    # Evaluation
    print("Evaluating on validation set...")
    model.eval()
    predictions = []
    true_labels = []
    """
    with torch.no_grad():
        for input_data, labels in train_loader:
            points = input_data
            input_data, labels = input_data.to(device), labels.to(device)
            input_data = normalize_input(input_data)

            logits = model(input_data)
            preds = torch.argmax(logits, dim=2)
            import ipdb
            ipdb.set_trace()
            save_point_clouds(input_data, labels, preds)
            predictions.append(preds.cpu())
            true_labels.append(labels.cpu())

    evaluate_model(predictions, true_labels, num_classes=10, ignore_index=0)
    #points = train_loader.dataset.dataset.data[0][0]
    #preds = predictions[0][0]
    #gt = true_labels[0][0]
    
    #import ipdb
    #ipdb.set_trace()
    """
    """
    all_points = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for input_data, labels in train_loader:

            xyz = input_data[:, :, :3] 

            input_data, labels = input_data.to(device), labels.to(device)
            input_data = normalize_input(input_data)

            logits = model(input_data)
            preds = torch.argmax(logits, dim=2)

            all_points.append(xyz.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
    import ipdb 
    ipdb.set_trace()
    points_concat = torch.cat(all_points, dim=0)   # Shape: (4*100000, 3)
    labels_concat = torch.cat(all_labels, dim=0)   # Shape: (4*100000,)
    preds_concat = torch.cat(all_preds, dim=0)     # Shape: (4*100000,)

    
    points_flat = points_concat.view(-1, 3)
    labels_flat = labels_concat.view(-1)
    preds_flat = preds_concat.view(-1)
    
    evaluate_model(preds_flat, labels_flat, num_classes=10, ignore_index=0)
    
    save_point_clouds(points_flat, preds_flat, labels_flat)
    """
    train_dataset = SelectedPointDataset(train_raw,  max_points= 100000)
    full_train_loader = DataLoader(train_dataset,  batch_size=4, shuffle=False, collate_fn=pad_collate_fn)
    all_points = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_data, labels in full_train_loader:
            
            xyz = input_data[:, :, :3]  # Get only XYZ

            input_data, labels = input_data.to(device), labels.to(device)
            input_data = normalize_input(input_data)

            logits = model(input_data)
            preds = torch.argmax(logits, dim=2)

            all_points.append(xyz.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    points_concat = torch.cat(all_points, dim=0)     # (total_batches, N, 3)
    preds_concat = torch.cat(all_preds, dim=0)       # (total_batches, N)
    labels_concat = torch.cat(all_labels, dim=0)     # (total_batches, N)

    points_flat = points_concat.view(-1, 3)
    preds_flat = preds_concat.view(-1)
    labels_flat = labels_concat.view(-1)

    evaluate_model(preds_flat, labels_flat, num_classes=10, ignore_index=0)
    save_point_clouds(points_flat, preds_flat, labels_flat)
    save_colored_by_accuracy(points_flat, preds_flat, labels_flat)


if __name__ == "__main__":
    main()