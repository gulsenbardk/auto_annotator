from header import *
from collections import defaultdict
def get_point_cloud_bounds(points):
	min_bounds = points.min(axis=0)
	max_bounds = points.max(axis=0)
	return min_bounds, max_bounds


def filter_points_in_bounds(points, min_bounds, max_bounds):
	
		filtered_points = []
		for point in points:
			x, y, z, intensity = point
			if (min_bounds[0] < x < max_bounds[0] and
				min_bounds[1] < y < max_bounds[1] and
				min_bounds[2] < z < max_bounds[2]):
				filtered_points.append(point)
		return filtered_points

def fill_bounding_box(corners, resolution=0.2): 

	corners = np.array(corners)

	x_min, x_max = np.min(corners[:, 0]), np.max(corners[:, 0])
	y_min, y_max = np.min(corners[:, 1]), np.max(corners[:, 1])
	z_min, z_max = np.min(corners[:, 2]), np.max(corners[:, 2])

	x_vals = np.arange(x_min, x_max, resolution)
	y_vals = np.arange(y_min, y_max, resolution)
	z_vals = np.arange(z_min, z_max, resolution)

	xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
	grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
	
	return grid_points

import numpy as np

def sample_bounding_box(corners, typ, points_per_m3=100):
    corners = np.array(corners)
    x_min, x_max = np.min(corners[:, 0]), np.max(corners[:, 0])
    y_min, y_max = np.min(corners[:, 1]), np.max(corners[:, 1])
    z_min, z_max = np.min(corners[:, 2]), np.max(corners[:, 2])


    volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    num_points = max(1, int(volume * points_per_m3))

    if typ == "gaussian":
        x_mean, x_std = (x_max + x_min) / 2, (x_max - x_min) / 6
        y_mean, y_std = (y_max + y_min) / 2, (y_max - y_min) / 6
        z_mean, z_std = (z_max + z_min) / 2, (z_max - z_min) / 6

        sampled_points = np.column_stack((
            np.random.normal(loc=x_mean, scale=x_std, size=num_points),
            np.random.normal(loc=y_mean, scale=y_std, size=num_points),
            np.random.normal(loc=z_mean, scale=z_std, size=num_points),
            np.zeros(num_points)  
        ))

    elif typ == "uniform":
        sampled_points = np.column_stack((
            np.random.uniform(x_min, x_max, num_points),
            np.random.uniform(y_min, y_max, num_points),
            np.random.uniform(z_min, z_max, num_points),
            np.zeros(num_points)
        ))

    elif typ == "random":
        x = x_min + (x_max - x_min) * np.random.rand(num_points)
        y = y_min + (y_max - y_min) * np.random.rand(num_points)
        z = z_min + (z_max - z_min) * np.random.rand(num_points)
        sampled_points = np.column_stack((x, y, z, np.zeros(num_points)))

    else:
        raise ValueError(f"Unknown sampling type: {typ}")

    return sampled_points


def dynamic_icp_finetune(original_pcd, synthetic_pcd, synthetic_lbl, initial_config=None):
    import numpy as np

    # Default initial parameters if not provided
    config = {
        'downsample_ratio': 100,
        'max_nn_kdtree': 10,
        'radius_kdtree': 0.5,
        'icp_iteration': 30,
        'trans_type': 'plane'
    }

    if initial_config:
        config.update(initial_config)

    # Criteria for improvement
    max_attempts = 5
    best_rmse = float("inf")
    best_output = None
    best_config = config.copy()

    for attempt in range(max_attempts):
        print(f"\n[Attempt {attempt+1}] Using config:", config)

        try:
            org_pts, org_lbls, syn_pts, rmse, fitness = icp_syn(
                original_pcd,
                synthetic_pcd,
                synthetic_lbl,
                config['downsample_ratio'],
                config['max_nn_kdtree'],
                config['radius_kdtree'],
                config['icp_iteration'],
                config['trans_type']
            )

            # Evaluate performance
            print(f" → RMSE: {rmse:.4f}, Fitness: {fitness:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_output = (org_pts, org_lbls, syn_pts, rmse, fitness)
                best_config = config.copy()

            # Adaptive adjustments
            if rmse > 1.0:
                config['downsample_ratio'] = max(2, config['downsample_ratio'] - 1)  # finer
                config['radius_kdtree'] = max(0.1, config['radius_kdtree'] * 0.8)
            if fitness < 0.5:
                config['icp_iteration'] += 20
                config['max_nn_kdtree'] += 5
            if attempt == 2:
                config['trans_type'] = 'point' if config['trans_type'] == 'plane' else 'plane'

        except Exception as e:
            print("❌ Error during ICP:", e)
            break

    print("\n✅ Best RMSE:", best_rmse)
    print("⚙️  Best Config:", best_config)
    return best_output, best_config

def extract_corners_from_geometry(geometry):

    corners = []
    
    if geometry.GetGeometryType() == ogr.wkbTINZ:
        for i in range(geometry.GetGeometryCount()):
            triangle = geometry.GetGeometryRef(i)
            ring = triangle.GetGeometryRef(0)
            corners.extend([ring.GetPoint(j) for j in range(ring.GetPointCount())])
    
    elif geometry.GetGeometryType() in [ogr.wkbMultiPolygon, ogr.wkbMultiPolygon25D]:
        for i in range(geometry.GetGeometryCount()):
            polygon = geometry.GetGeometryRef(i)
            for j in range(polygon.GetGeometryCount()):
                ring = polygon.GetGeometryRef(j)
                corners.extend([ring.GetPoint(k) for k in range(ring.GetPointCount())])
    
    elif geometry.GetGeometryType() in [ogr.wkbPolygon, ogr.wkbPolygon25D]:
        for i in range(geometry.GetGeometryCount()):
            ring = geometry.GetGeometryRef(i)
            corners.extend([ring.GetPoint(j) for j in range(ring.GetPointCount())])
    
    corners = np.array(corners)
    
    if corners.size == 0:
        return np.empty((0, 3))
    
    return corners
"""
def icp(original_pcd, synthetic_pcd, synthetic_lbl, downsample_ratio, max_nn_kdtree, radius_kdtree, icp_iteration):
    original_pcd = np.asarray(original_pcd)
    synthetic_pcd = np.asarray(synthetic_pcd)
    
    org_pcd = o3d.geometry.PointCloud()
    org_pcd.points = o3d.utility.Vector3dVector(original_pcd[:, :3])
    org_intensity = original_pcd[:, 3]
    downsampled_org_pcd = org_pcd.uniform_down_sample(every_k_points=downsample_ratio)
    
    syn_pcd = o3d.geometry.PointCloud()
    syn_pcd.points = o3d.utility.Vector3dVector(synthetic_pcd[:, :3])
    syn_intensity = synthetic_pcd[:, 3]
    unique_labels = np.unique(synthetic_lbl)
    label_to_color = {label: np.random.rand(3) for label in unique_labels}
    colors = np.array([label_to_color[label] for label in synthetic_lbl])
    syn_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    init_transformation = np.eye(4)
    syn_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_kdtree, max_nn=max_nn_kdtree))
    downsampled_org_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_kdtree, max_nn=max_nn_kdtree))
    result_icp = o3d.pipelines.registration.registration_icp(
        syn_pcd, 
        downsampled_org_pcd, 
        0.1,
        init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_iteration)
    )
    syn_pcd.transform(result_icp.transformation)
    evaluation = o3d.pipelines.registration.evaluate_registration(syn_pcd, downsampled_org_pcd, 10)
    correspondences = np.asarray(evaluation.correspondence_set)
    
    downsampled_labels = np.full(len(downsampled_org_pcd.points), -1, dtype=np.int32)
    downsampled_intensity = np.zeros(len(downsampled_org_pcd.points))

    for i, j in tqdm(correspondences, desc="Transferring labels (downsampled)", leave=False):
        downsampled_labels[j] = synthetic_lbl[i]
        downsampled_intensity[j] = synthetic_pcd[i, 3]

    
    kd_tree = o3d.geometry.KDTreeFlann(downsampled_org_pcd)
    original_labels = np.full(len(org_pcd.points), -1, dtype=np.int32)
    original_intensity = np.zeros(len(org_pcd.points))

    for i, point in enumerate(tqdm(org_pcd.points, desc="Upsampling labels to original cloud")):
        _, idx, _ = kd_tree.search_knn_vector_3d(point, 2)  # Nearest neighbor
        if downsampled_labels[idx[0]] != -1:  # If a valid label exists
            original_labels[i] = downsampled_labels[idx[0]]
            original_intensity[i] = downsampled_intensity[idx[0]]
        else:
            # Find the nearest valid label
            _, idx, _ = kd_tree.search_knn_vector_3d(point, downsample_ratio) 
            for neighbor in idx:
                if downsampled_labels[neighbor] != -1:
                    original_labels[i] = downsampled_labels[neighbor]
                    original_intensity[i] = downsampled_intensity[neighbor]
                    break  # Stop once we find a valid label

    # Ensure all labels are assigned (fallback to mode of existing labels)
    valid_labels = original_labels[original_labels != -1]
    if len(valid_labels) > 0:
        most_common_label = np.bincount(valid_labels).argmax()
        original_labels[original_labels == -1] = most_common_label

    # Return aligned point cloud with correct labels
    original_points = np.asarray(org_pcd.points)
    return np.hstack((original_points, original_intensity[:, None])), original_labels


"""
def icp_syn(original_pcd, synthetic_pcd, synthetic_lbl, downsample_ratio, max_nn_kdtree, radius_kdtree, icp_iteration, trans_type):
    import open3d as o3d
    import numpy as np
    from tqdm import tqdm
    from collections import defaultdict

    print("Converting point clouds to Open3D format...")
    original_pcd = np.asarray(original_pcd)
    synthetic_pcd = np.asarray(synthetic_pcd)

    assert original_pcd.shape[1] >= 4, "Original point cloud must include intensity in the 4th column"
    assert synthetic_pcd.shape[1] >= 3, "Synthetic point cloud must be at least XYZ"

    org_pcd = o3d.geometry.PointCloud()
    org_pcd.points = o3d.utility.Vector3dVector(original_pcd[:, :3])
    org_intensity = original_pcd[:, 3]

    print("Downsampling original point cloud...")
    downsampled_org_pcd = org_pcd.uniform_down_sample(downsample_ratio)
    downsampled_org_points = np.asarray(downsampled_org_pcd.points)

    print("Interpolating intensity for downsampled points...")
    kd_tree_org = o3d.geometry.KDTreeFlann(org_pcd)
    downsampled_intensity = np.zeros(len(downsampled_org_points))
    for i, pt in enumerate(tqdm(downsampled_org_points, desc="Intensity interpolation")):
        _, idx, _ = kd_tree_org.search_knn_vector_3d(pt, downsample_ratio)
        downsampled_intensity[i] = org_intensity[idx[0]]

    print("Preparing synthetic point cloud...")
    syn_points = synthetic_pcd[:, :3]
    syn_pcd = o3d.geometry.PointCloud()
    syn_pcd.points = o3d.utility.Vector3dVector(syn_points)

    print("Estimating synthetic intensity...")
    estimated_syn_intensity = np.zeros(len(syn_points))
    for i, pt in enumerate(tqdm(syn_points, desc="Synthetic intensity estimation")):
        _, idx, dist = kd_tree_org.search_knn_vector_3d(pt, max_nn_kdtree)
        weights = 1 / (np.array(dist) + 1e-8)
        weights /= np.sum(weights)
        estimated_syn_intensity[i] = np.sum(weights * org_intensity[idx])

    print("Assigning synthetic colors based on labels...")
    unique_labels = np.unique(synthetic_lbl)
    label_to_color = {label: np.random.rand(3) for label in unique_labels}
    colors = np.array([label_to_color[label] for label in synthetic_lbl])
    syn_pcd.colors = o3d.utility.Vector3dVector(colors)

    print("Estimating normals...")
    syn_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_kdtree, max_nn=max_nn_kdtree))
    downsampled_org_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_kdtree, max_nn=max_nn_kdtree))

    print("Running ICP registration...")
    init_transform = np.eye(4)
    if trans_type == "plane":
        result_icp = o3d.pipelines.registration.registration_icp(
            downsampled_org_pcd,
            syn_pcd,
            0.1,
            init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_iteration)
        )
    elif trans_type == "point":
        result_icp = o3d.pipelines.registration.registration_icp(
            downsampled_org_pcd,
            syn_pcd,
            0.1,
            init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_iteration)
        )

    print("Applying transformation...")
    downsampled_org_pcd.transform(result_icp.transformation)

    print("Evaluating registration...")
    evaluation = o3d.pipelines.registration.evaluate_registration(downsampled_org_pcd, syn_pcd, 10.0)
    correspondences = np.asarray(evaluation.correspondence_set)

    print("Transferring labels...")
    downsampled_labels = np.full(len(downsampled_org_pcd.points), -1, dtype=np.int32)
    for i, j in tqdm(correspondences, desc="Label transfer to downsampled"):
        downsampled_labels[i] = synthetic_lbl[j]

    print("Computing class-based RMSE...")
    class_errors = defaultdict(list)
    downsampled_points = np.asarray(downsampled_org_pcd.points)
    synthetic_points = np.asarray(syn_pcd.points)
    for i, j in correspondences:
        class_label = synthetic_lbl[j]
        error = np.linalg.norm(downsampled_points[i] - synthetic_points[j])
        class_errors[class_label].append(error ** 2)

    class_rmse = {
        label: np.sqrt(np.mean(errors)) if errors else 0.0
        for label, errors in class_errors.items()
    }

    print("Upsampling labels to original resolution...")
    kd_tree = o3d.geometry.KDTreeFlann(downsampled_org_pcd)
    original_labels = np.full(len(org_pcd.points), -1, dtype=np.int32)
    for i, point in enumerate(tqdm(org_pcd.points, desc="Upsampling labels")):
        _, idx, _ = kd_tree.search_knn_vector_3d(point, downsample_ratio)
        nearest = idx[0]
        if downsampled_labels[nearest] != -1:
            original_labels[i] = downsampled_labels[nearest]
        else:
            for neighbor in idx:
                if downsampled_labels[neighbor] != -1:
                    original_labels[i] = downsampled_labels[neighbor]
                    break

    print("Filling missing labels...")
    valid_labels = original_labels[original_labels != -1]
    if len(valid_labels) > 0:
        most_common_label = np.bincount(valid_labels).argmax()
        original_labels[original_labels == -1] = most_common_label

    original_points = np.asarray(org_pcd.points)
    synthetic_with_intensity = np.hstack((syn_points, estimated_syn_intensity[:, None]))

    print("✓ ICP processing complete.")

    return (
        np.hstack((original_points, org_intensity[:, None])),
        original_labels,
        synthetic_with_intensity,
        evaluation.inlier_rmse,
        evaluation.fitness,
        class_rmse
    )


def tune_icp(original_pcd, synthetic_pcd, synthetic_lbl):
    best_rmse_point = float("inf")
    best_rmse_plane = float("inf")
    best_config_point = None
    best_config_plane = None

    results_point = []
    results_plane = []

    downsample_ratios = [2, 5, 10]
    radii = [0.05, 0.2]
    icp_iters = [100]

    for ds in downsample_ratios:
        for rad in radii:
            for iters in icp_iters:
                # --- Point-to-Point ICP ---
                try:
                    _, _, _, rmse_pt, fitness_pt = icp_syn(original_pcd, synthetic_pcd, synthetic_lbl, ds, 30, rad, iters, "point")
                    results_point.append(((ds, rad, iters), rmse_pt, fitness_pt))
                    if rmse_pt < best_rmse_point:
                        best_rmse_point = rmse_pt
                        best_config_point = (ds, rad, iters)
                except Exception as e:
                    print(f"[Point] Failed: ds={ds}, rad={rad}, iter={iters} | Error: {e}")

                # --- Point-to-Plane ICP ---
                try:
                    _, _, _, rmse_plane, fitness_plane = icp_syn(original_pcd, synthetic_pcd, synthetic_lbl,ds, 30, rad,iters, "plane")
                    results_plane.append(((ds, rad, iters), rmse_plane, fitness_plane))
                    if rmse_plane < best_rmse_plane:
                        best_rmse_plane = rmse_plane
                        best_config_plane = (ds, rad, iters)
                except Exception as e:
                    print(f"[Plane] Failed: ds={ds}, rad={rad}, iter={iters} | Error: {e}")

    # --- Print Best Configs ---
    print("\nBest ICP Config - Point-to-Point:")
    if best_config_point:
        print(f"Downsample Ratio: {best_config_point[0]}, Radius: {best_config_point[1]}, Iter: {best_config_point[2]}")
        print(f"Best RMSE: {best_rmse_point:.4f}")
    else:
        print("No successful point-to-point results.")

    print("\nBest ICP Config - Point-to-Plane:")
    if best_config_plane:
        print(f"Downsample Ratio: {best_config_plane[0]}, Radius: {best_config_plane[1]}, Iter: {best_config_plane[2]}")
        print(f"Best RMSE: {best_rmse_plane:.4f}")
    else:
        print("No successful point-to-plane results.")

    # --- Print All Results ---
    print("\nAll Point-to-Point Results:")
    for (ds, rad, iters), rmse, fitness in results_point:
        print(f"[Point] ds={ds}, radius={rad}, iter={iters} -> RMSE={rmse:.4f}, Fitness={fitness:.3f}")

    print("\nAll Point-to-Plane Results:")
    for (ds, rad, iters), rmse, fitness in results_plane:
        print(f"[Plane] ds={ds}, radius={rad}, iter={iters} -> RMSE={rmse:.4f}, Fitness={fitness:.3f}")

    return {
        "best_point": (best_config_point, best_rmse_point),
        "best_plane": (best_config_plane, best_rmse_plane),
        "results_point": results_point,
        "results_plane": results_plane
    }