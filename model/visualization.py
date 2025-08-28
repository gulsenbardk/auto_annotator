import tkinter as tk
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np 


def TARL_visualize_compare_pcd_clusters(points1, label1, points2, label2, label1_name="Ground Truth", label2_name="Prediction"):
    # Setup screen geometry using tkinter
    root = tk.Tk()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()

    def create_colored_pcd(points, labels):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        num_classes = len(np.unique(labels))
        colors = np.zeros((len(labels), 4))
        cmap = plt.get_cmap("tab20b")
        class_colors = cmap(np.linspace(0, 1, num_classes))

        for idx, class_id in enumerate(np.unique(labels)):
            colors[labels == class_id] = class_colors[idx]

        colors[labels == 0] = [0.7, 0.7, 0.7, 0.1]  # ignore_index (class 0) as gray
        return pcd, colors[:, :3]

    # Create colored point clouds
    pcd1, color1 = create_colored_pcd(points1, label1)
    pcd1.colors = o3d.utility.Vector3dVector(color1)

    pcd2, color2 = create_colored_pcd(points2, label2)
    pcd2.colors = o3d.utility.Vector3dVector(color2)

    # Open3D visualizers
    vis1 = o3d.visualization.Visualizer()
    vis2 = o3d.visualization.Visualizer()

    vis1.create_window(window_name=label1_name, width=int(w / 2), height=h, left=0, top=0)
    vis2.create_window(window_name=label2_name, width=int(w / 2), height=h, left=int(w / 2), top=0)

    vis1.add_geometry(pcd1)
    vis2.add_geometry(pcd2)

    # Sync both windows
    while True:
        vis1.update_geometry(pcd1)
        vis2.update_geometry(pcd2)

        if not vis1.poll_events() or not vis2.poll_events():
            break

        vis1.update_renderer()
        vis2.update_renderer()

        # Synchronize view
        ctr1 = vis1.get_view_control()
        ctr2 = vis2.get_view_control()
        params = ctr1.convert_to_pinhole_camera_parameters()
        ctr2.convert_from_pinhole_camera_parameters(params)

    vis1.destroy_window()
    vis2.destroy_window()