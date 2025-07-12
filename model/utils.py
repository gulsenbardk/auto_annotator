
from header import *

def collate_fn(batch):
    points = [item[0] for item in batch]  # list of (N_i, 3) tensors
    labels = [item[1] for item in batch]  # list of (N_i,) tensors
    return points, labels


def pad_collate_fn(batch):
    import torch.nn.functional as F
    points_list, labels_list = zip(*batch)
    max_len = max(p.shape[0] for p in points_list)

    padded_points = []
    padded_labels = []

    for points, labels in batch:
        pad_len = max_len - points.shape[0]
        padded_points.append(F.pad(points, (0, 0, 0, pad_len)))  # pad N dim
        padded_labels.append(F.pad(labels, (0, pad_len), value=-1))  # optional: pad label with -1 (ignore)

    return torch.stack(padded_points), torch.stack(padded_labels)


def get_color_map_binary(preds):
    colors = np.zeros((len(preds), 3))
    colors[preds == 1] = [0.0, 1.0, 0.0]  
    colors[preds == 0] = [0.7, 0.7, 0.7]  
    return colors

def visualize_prediction(points, labels, preds):

    points_np = points.cpu().numpy()
    preds_np = preds.cpu().numpy()
    colors = get_color_map_binary(preds_np)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])



def remove_ground_points(dataset, z_threshold=0.5):
    """
    Removes ground points from a PointCloudDataset based on z-height.

    Args:
        dataset: PointCloudDataset object with .data being list of arrays or tuples
        z_threshold: Ground height cutoff

    Returns:
        PointCloudDataset with ground points removed
    """
    filtered_data = []
    for item in dataset.data:
        # Handle tuple case: (points, label) or (points, features, label), etc.
        if isinstance(item, tuple):
            points = item[0]  # Assume first element is (N, 3) coordinates
            other_data = item[1:]  # e.g., label, features

            mask = points[:, 2] > z_threshold
            filtered_points = points[mask]

            if filtered_points.shape[0] > 0:
                filtered_data.append((filtered_points, *other_data))

        else:
            # Assume raw point cloud only
            points = item
            mask = points[:, 2] > z_threshold
            filtered_points = points[mask]

            if filtered_points.shape[0] > 0:
                filtered_data.append(filtered_points)

    dataset.data = filtered_data
    return dataset

def downsample_points(points, labels, max_points):
    N = points.shape[0]
    if N <= max_points:
        return points, labels
    else:
        idx = torch.randperm(N)[:max_points]
        return points[idx], labels[idx]

def downsample_points(points, labels, max_points):
    N = points.shape[0]
    if N <= max_points:
        return points, labels
    else:
        idx = torch.randperm(N)[:max_points]
        return points[idx], labels[idx]


def downsample_dominant_classes(points, labels, max_per_class=1_000_000):
    new_points = []
    new_labels = []
    
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        count = len(idx)
        
        if count > max_per_class:
            sampled_idx = np.random.choice(idx, max_per_class, replace=False)
        else:
            sampled_idx = idx  # Keep as is
        
        new_points.append(points[sampled_idx])
        new_labels.append(labels[sampled_idx])
    
    return np.vstack(new_points), np.concatenate(new_labels)


from sklearn.utils.class_weight import compute_class_weight

def compute_class_weights(dataloader, num_classes):
    all_labels = []

    for _, labels in dataloader:
        for lbl in labels:
            all_labels.append(lbl.view(-1).cpu().numpy())

    all_labels = np.concatenate(all_labels)

    unique_classes = np.unique(all_labels)
    if len(unique_classes) < num_classes:
        print(f"⚠️ Warning: Only {len(unique_classes)} of {num_classes} classes found in training data: {unique_classes}")

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=all_labels
    )

    return torch.tensor(class_weights, dtype=torch.float32)


def compute_normals(points, k=20):
    tree = KDTree(points)
    normals = []
    for i, p in enumerate(points):
        _, idx = tree.query(p.reshape(1, -1), k=k)
        neighbors = points[idx[0]]
        cov = np.cov(neighbors.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        normals.append(normal)
    return np.array(normals)

def estimate_normals(points, k=20):
    points_np = points.cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k).fit(points_np)
    _, indices = nbrs.kneighbors(points_np)

    normals = []
    for i in range(points_np.shape[0]):
        neighbors = points_np[indices[i]]
        cov = np.cov(neighbors.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        if normal.dot(points_np[i]) > 0:  # flip toward origin
            normal = -normal
        normals.append(normal)
    normals = torch.tensor(normals, dtype=torch.float32)
    return normals

def normalize_input(x):
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-5)


def get_weighted_sampler(dataset, num_classes):
    all_labels = []
    for _, labels in dataset:
        all_labels.append(labels.view(-1).numpy())
    all_labels = np.concatenate(all_labels)
    class_sample_counts = np.bincount(all_labels)
    weights = 1. / class_sample_counts
    sample_weights = weights[all_labels]
    return torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))