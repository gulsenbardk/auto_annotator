from header import * 
from utils import *


class PointCloudDataset(Dataset):
    def __init__(self, dir_path, downsample_dominant_classes=True, max_per_label=1_000_000):
        self.files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".pth")]
        self.data = []
        
        self.num_points_per_cloud = []
        self.label_distributions = []
        
        all_labels = []
        self.global_label_distribution = {}

        # First pass to count global label frequency if downsampling is enabled
        if downsample_dominant_classes:
            label_counter = {}
            for file in self.files:
                item = torch.load(file)
                labels = item['labels']
                unique_labels, counts = torch.unique(labels, return_counts=True)
                for lbl, cnt in zip(unique_labels, counts):
                    lbl = int(lbl)
                    cnt = int(cnt)
                    label_counter[lbl] = label_counter.get(lbl, 0) + cnt
            self.global_label_distribution = label_counter
        else:
            self.global_label_distribution = {}

        # Second pass to load and optionally downsample
        for file in self.files:
            item = torch.load(file)
            points = item['points'][:, :3]
            labels = item['labels']

            if downsample_dominant_classes:
                # Get per-label indices
                kept_idx = []
                for lbl in labels.unique():
                    lbl = int(lbl)
                    idx = (labels == lbl).nonzero(as_tuple=True)[0]
                    count = idx.shape[0]
                    
                    # Downsample if label is globally dominant
                    if self.global_label_distribution.get(lbl, 0) > max_per_label:
                        sample_size = min(count, max_per_label)
                        sampled_idx = idx[torch.randperm(count)[:sample_size]]
                    else:
                        sampled_idx = idx
                    
                    kept_idx.append(sampled_idx)

                kept_idx = torch.cat(kept_idx)
                points = points[kept_idx]
                labels = labels[kept_idx]

            self.data.append((points.float(), labels.long()))
            self.num_points_per_cloud.append(points.shape[0])
            
            # Per-sample label distribution
            unique_labels, counts = torch.unique(labels, return_counts=True)
            label_count_dict = {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, counts)}
            self.label_distributions.append(label_count_dict)
            
            all_labels.extend(labels.tolist())

        # Final global distribution if not done earlier
        if not self.global_label_distribution:
            unique_all, counts_all = torch.unique(torch.tensor(all_labels), return_counts=True)
            self.global_label_distribution = {int(lbl): int(cnt) for lbl, cnt in zip(unique_all, counts_all)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_point_count_stats(self):
        np_points = np.array(self.num_points_per_cloud)
        return {
            "min": int(np_points.min()),
            "max": int(np_points.max()),
            "mean": float(np_points.mean()),
            "std": float(np_points.std())
        }
    
    def get_global_label_distribution(self):
        return self.global_label_distribution
    
    def print_sample_info(self, idx):
        points, labels = self.data[idx]
        print(f"Sample {idx}:")
        print(f"  Number of points: {points.shape[0]}")
        print(f"  Label distribution: {self.label_distributions[idx]}")

SELECTED_LABELS = {
    2: 1,  # obstacle → 1
    3: 2,  # vegetation → 2
    4: 3,  # tree → 3
    5: 4,  # streetLamp → 4
    6: 5,  # barrier → 5
    7: 6,  # trafficSign → 6
    8: 7,  # building → 7
    9:  8,  # pole → 8
    10: 9  # trafficLight → 9
}
class SelectedPointDataset(Dataset):
    def __init__(self, dataset, max_points=10000):
        self.dataset = dataset
        self.max_points = max_points

        # Remap selected labels to 1-based (start from 1)
        self.label_map = {k: i + 1 for i, k in enumerate(SELECTED_LABELS.keys())}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        points, labels = self.dataset[idx] 

        remapped_labels = torch.zeros_like(labels, dtype=torch.long)
        for orig_label, new_label in self.label_map.items():
            remapped_labels[labels == orig_label] = new_label

        points, remapped_labels = downsample_points(points, remapped_labels, self.max_points)

        normals = estimate_normals(points, k=20)

        points_with_normals = torch.cat([points, normals], dim=1)

        return points_with_normals, remapped_labels
