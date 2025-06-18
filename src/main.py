import argparse
import os
import numpy as np
import open3d as o3d

from processV3 import *

def process_point_clouds(xodr_path, pcd_dir, output_dir, nn, max_iter, radius, max_nn):
    #getsingleBuild(xodr_path, pcd_dir, output_dir)
    getsingleXODR_asil(xodr_path, pcd_dir, output_dir)
    
# Function to load point cloud and labels
def load_point_cloud_and_labels(bin_path, label_path):
    # Load point cloud (X, Y, Z, Intensity)
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # Load labels (class labels for each point)
    labels = np.fromfile(label_path, dtype=np.uint32)

    # Extract X, Y, Z points
    xyz_points = points[:, :3]

    return xyz_points, labels

# Function to visualize the point cloud with color based on labels
def visualize_point_cloud_with_labels(xyz_points, labels, num_classes=21):
    # Generate a color map based on unique labels
    label_colors = np.random.rand(num_classes, 3)  # Random colors for each class (you can customize this)
    import ipdb 
    ipdb.set_trace()
    # Map labels to colors
    colors = label_colors[labels]

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()

    # Set point cloud data
    pcd.points = o3d.utility.Vector3dVector(xyz_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize using Open3D
    o3d.visualization.draw_geometries([pcd])   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process point clouds using XODR and save synthetic labels.')
    parser.add_argument('--xodr_path', type=str, required=True, help='Path to XODR file')
    parser.add_argument('--pcd_dir', type=str, required=True, help='Directory containing point cloud files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed files')
    #parser.add_argument('--kitti_dir', type=str, required=True, help='Directory to save processed files')
    parser.add_argument('--nn', type=float, default=5, help='Voxel size for downsampling')
    parser.add_argument('--max_iter', type=int, default=100, help='Max iterations for ICP')
    parser.add_argument('--radius', type=float, default=0.1, help='Radius for KDTree')
    parser.add_argument('--max_nn', type=int, default=30, help='MaxNN for KDTree')
    
    args = parser.parse_args()
    
    process_point_clouds(args.xodr_path, args.pcd_dir, args.output_dir, args.nn, args.max_iter, args.radius, args.max_nn)
    """import numpy as np

    # Path to your .label file
    label_file_path = "/mnt/data/bard_gu/pcd/Datasets/SemanticKITTI/dataset/sequences/00/labels/00000_00.label"

    # Read the .label file as uint32
    labels = np.fromfile(label_file_path, dtype=np.uint32)

    print(f"Total labels read: {len(labels)}")
    print(f"First 10 labels: {labels[:10]}")

    # If you want to see unique label values
    unique_labels = np.unique(labels)
    print(f"Unique labels: {unique_labels}")"""
    bin_dir = '/mnt/data/bard_gu/pcd/Datasets/SemanticKITTI/dataset/sequences/01/velodyne'
    label_dir = '/mnt/data/bard_gu/pcd/Datasets/SemanticKITTI/dataset/sequences/01/labels'




    # Load a sample point cloud and label file (you can loop through a batch or entire dataset)
    #bin_path = bin_dir + '/00000_00.bin'  # Change this to a valid .bin file path
    #label_path = label_dir + '/00000_00.label'  # Change this to a valid .label file path
#
    ## Load the point cloud and labels
    #xyz_points, labels = load_point_cloud_and_labels(bin_path, label_path)
    #import ipdb
    #ipdb.set_trace()
    ## Visualize the point cloud with segmentation labels
    #visualize_point_cloud_with_labels(xyz_points, labels)