from tools import *
import json



color_map = {
  0: [0, 0, 0],
  1: [245, 150, 100],
  2: [245, 230, 100],
  3: [150, 60, 30],
  4: [180, 30, 80],
  5: [255, 0, 0],
  6: [30, 30, 255],
  7: [200, 40, 255],
  8: [90, 30, 150],
} 
types = {}

def get_label_count(feature_type_str):
    for label, label_type in types.items():
        if label_type == feature_type_str:
            return label
    
    new_label = max(types.keys(), default=-1) + 1 
    types[new_label] = feature_type_str
    return new_label

def visualizePCD(xyz, labels):
    xyz = np.asarray(xyz, dtype=np.float64)
    

    unique_labels = np.unique(labels)
    colormap = plt.get_cmap('viridis')
    colors = colormap(np.linspace(0, 1, len(unique_labels)))[:, :3]
    
   
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    

    point_colors = np.array([label_to_color[label] for label in labels])
    
   
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(point_colors)
    

    o3d.visualization.draw_geometries([point_cloud])


def visualize_pcd_with_intensity(pcd_array):

    pcd_array = np.asarray(pcd_array, dtype=np.float32)


    points = pcd_array[:, :3]

    intensity = pcd_array[:, 3]
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)  # Avoid div by zero

    colors = np.stack([intensity, intensity, intensity], axis=1)  # Grayscale RGB


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud with Intensity")

def split_and_save_pointcloud(pcd, lbl, idx, output_dir, k=10):
    syn_pcd_dir = os.path.join(output_dir, 'velodyne')
    syn_labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(syn_pcd_dir, exist_ok=True)
    os.makedirs(syn_labels_dir, exist_ok=True)
    

    min_coords = np.min(pcd, axis=0)
    max_coords = np.max(pcd, axis=0)
    

    bbox_size = (max_coords - min_coords) / k
    
    for i in range(k):

        bbox_min = min_coords + i * bbox_size
        bbox_max = bbox_min + bbox_size

        mask = np.all((pcd >= bbox_min) & (pcd <= bbox_max), axis=1)
        pcd_subset = pcd[mask]
        lbl_subset = lbl[mask]
        
        if len(pcd_subset) == 0:
            continue
        

        tensor_pcd = torch.from_numpy(pcd_subset)
        inverse_pcd = torch.unique(tensor_pcd, return_inverse=True)[1]
        pcd_encoded = inverse_pcd.cpu().numpy()
        

        tensor_lbl = torch.from_numpy(lbl_subset)
        inverse_lbl = torch.unique(tensor_lbl, return_inverse=True)[1]
        labels_encoded = inverse_lbl.cpu().numpy()
        
        lower_bits = labels_encoded & 0xFFFF
        upper_bits = labels_encoded >> 16
        switched_value = (lower_bits << 16) | upper_bits
        import ipdb 
        ipdb.set_trace()
        output_pcd_path = os.path.join(syn_pcd_dir, f'{idx:05d}_{i:02d}.bin')
        output_lbl_path = os.path.join(syn_labels_dir, f'{idx:05d}_{i:02d}.label')
        
        pcd_encoded.astype('float32').tofile(output_pcd_path)
        switched_value.astype('uint32').tofile(output_lbl_path)

def save_point_cloud_npz(points, labels, filename):
    points = np.array(points, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    np.savez(filename, points=points, labels=labels) 

def save_ascii_for_cloudcompare(filename, points, labels=None):

    if labels is not None:
        assert len(points) == len(labels)
        data = np.hstack((points, labels.reshape(-1, 1)))
        np.savetxt(filename, data, fmt="%.6f %.6f %.6f %d")
    else:
        np.savetxt(filename, points, fmt="%.6f %.6f %.6f")


def getsingleXODR(xodr_path, pcd_dir, output_dir):
    driver = ogr.GetDriverByName('XODR')
    options = ["DISSOLVE_TIN=True", "EPSILON=0.2"]
    source = gdal.OpenEx(xodr_path, gdal.OF_VECTOR, open_options=options)
    
    if source is None:
        raise Exception(f"Cannot open file: {xodr_path}")
    
    layers = source.GetLayerCount()
    pcd_files = [f for f in os.listdir(pcd_dir) if f.endswith('.las')]
   
    for idx, pcd_path in enumerate(pcd_files):
        pcd_path = os.path.join(pcd_dir, pcd_path)
        
        gauss_pcd = []
        gauss_lbl = []

        random_pcd = []
        random_lbl = []

        uniform_pcd = []
        uniform_lbl = []

        original_pcd = las2array(pcd_path)

        min_bounds, max_bounds = get_point_cloud_bounds(original_pcd)

        for i in tqdm(range(layers), desc="Processing Layers"):
            layer = source.GetLayerByIndex(i)
            layer_name = layer.GetName()
            if layer_name in ["Lane", "RoadObject", "RoadMark"]:
                features = list(layer)
                
                
                for feature in tqdm(features, desc=f"Processing Features in {layer_name}", leave=False):
           
                    geometry = feature.GetGeometryRef()
                    label_ = -1
                    if layer_name == "RoadObject":
                        type_ = feature.type 
                        name = feature.name
                        if type_ == "pole":
                            label_ = get_label_count(name)
                        elif type_ == roadMark:
                            label_ = 0
                    
                        else:
                            label_ = get_label_count(type_)
                    elif layer_name == "Lane":
                        type_ = feature.type
                        if type_ in ["driving", "border", "bidirectional"]:
                            label_ = 1
                        elif type_ == "parking":
                            label_ = 15
                        else:
                            label_ = get_label_count(type_)
                    else: 
                        label_ = 0
                
                    geometry_type = geometry.GetGeometryType()
                    if geometry_type == ogr.wkbPoint:
            
                        point = geometry.GetPoint()
                        poi = (point[0], point[1], point[2] if len(point) > 2 else 0)
                        filtered_pois = filter_points_in_bounds([poi], min_bounds, max_bounds)
                        for poi in filtered_pois:
                            synthetic_pcd.append(poi)
                            synthetic_lbl.append(label_)

                    elif geometry_type == ogr.wkbMultiPoint:
                        num_points = geometry.GetGeometryCount()
                        for i in range(num_points):
                            point_geom = geometry.GetGeometryRef(i)
                            point = point_geom.GetPoint()
                            poi = (point[0], point[1], point[2] if len(point) > 2 else 0)
                            filtered_pois = filter_points_in_bounds([poi], min_bounds, max_bounds)
                            for poi in filtered_pois:
                                synthetic_pcd.append(poi)
                                synthetic_lbl.append(label_)

                    elif geometry_type == ogr.wkbLineString:
                        num_points = geometry.GetPointCount()
                        corners = [geometry.GetPoint(i) for i in range(num_points)]
                        inter_type = "gaussian"
                        pois = sample_bounding_box(corners, inter_type)
                        filtered_pois = filter_points_in_bounds(pois, min_bounds, max_bounds)
                        for poi in filtered_pois:
                            gauss_pcd.append(poi)
                            gauss_lbl.append(label_)
                        
                        inter_type = "random"
                        pois = sample_bounding_box(corners, inter_type)
                        filtered_pois = filter_points_in_bounds(pois, min_bounds, max_bounds)
                        for poi in filtered_pois:
                            random_pcd.append(poi)
                            random_lbl.append(label_)

                        inter_type = "uniform"
                        pois = sample_bounding_box(corners, inter_type)
                        filtered_pois = filter_points_in_bounds(pois, min_bounds, max_bounds)
                        for poi in filtered_pois:
                            uniform_pcd.append(poi)
                            uniform_lbl.append(label_)

                    elif geometry_type == ogr.wkbMultiLineString:
                        num_lines = geometry.GetGeometryCount()
                        for i in range(num_lines):
                            line = geometry.GetGeometryRef(i)
                            num_points = line.GetPointCount()
                            corners = [line.GetPoint(j) for j in range(num_points)]
                            inter_type = "gaussian"
                            pois = sample_bounding_box(corners, inter_type)
                            filtered_pois = filter_points_in_bounds(pois, min_bounds, max_bounds)
                            for poi in filtered_pois:
                                gauss_pcd.append(poi)
                                gauss_lbl.append(label_)
                            
                            inter_type = "random"
                            pois = sample_bounding_box(corners, inter_type)
                            filtered_pois = filter_points_in_bounds(pois, min_bounds, max_bounds)
                            for poi in filtered_pois:
                                random_pcd.append(poi)
                                random_lbl.append(label_)

                            inter_type = "uniform"
                            pois = sample_bounding_box(corners, inter_type)
                            filtered_pois = filter_points_in_bounds(pois, min_bounds, max_bounds)
                            for poi in filtered_pois:
                                uniform_pcd.append(poi)
                                uniform_lbl.append(label_)



                    elif geometry_type == ogr.wkbPolygon or geometry_type == ogr.wkbPolygon25D:
                        num_rings = geometry.GetGeometryCount()
                        for i in range(num_rings):
                            ring = geometry.GetGeometryRef(i)
                            corners = [(ring.GetPoint(j)[0], ring.GetPoint(j)[1], ring.GetPoint(j)[2]) for j in range(ring.GetPointCount())]
                            inter_type = "gaussian"
                            pois = sample_bounding_box(corners, inter_type)
                            filtered_pois = filter_points_in_bounds(pois, min_bounds, max_bounds)
                            for poi in filtered_pois:
                                gauss_pcd.append(poi)
                                gauss_lbl.append(label_)
                            
                            inter_type = "random"
                            pois = sample_bounding_box(corners, inter_type)
                            filtered_pois = filter_points_in_bounds(pois, min_bounds, max_bounds)
                            for poi in filtered_pois:
                                random_pcd.append(poi)
                                random_lbl.append(label_)
                            
                            inter_type = "uniform"
                            pois = sample_bounding_box(corners, inter_type)
                            filtered_pois = filter_points_in_bounds(pois, min_bounds, max_bounds)
                            for poi in filtered_pois:
                                uniform_pcd.append(poi)
                                uniform_lbl.append(label_)

                    elif geometry_type == ogr.wkbMultiPolygon or geometry_type == ogr.wkbMultiPolygon25D:
                        num_polygons = geometry.GetGeometryCount()
                        for i in range(num_polygons):
                            polygon = geometry.GetGeometryRef(i)
                            num_rings = polygon.GetGeometryCount()
                            for j in range(num_rings):
                                ring = polygon.GetGeometryRef(j)
                                corners = [(ring.GetPoint(k)[0], ring.GetPoint(k)[1], ring.GetPoint(k)[2]) for k in range(ring.GetPointCount())]
                                inter_type = "gaussian"
                                pois = sample_bounding_box(corners, inter_type)
                                filtered_pois = filter_points_in_bounds(pois, min_bounds, max_bounds)
                                for poi in filtered_pois:
                                    gauss_pcd.append(poi)
                                    gauss_lbl.append(label_)
                                
                                inter_type = "random"
                                pois = sample_bounding_box(corners, inter_type)
                                filtered_pois = filter_points_in_bounds(pois, min_bounds, max_bounds)
                                for poi in filtered_pois:
                                    random_pcd.append(poi)
                                    random_lbl.append(label_)
                                
                                inter_type = "uniform"
                                pois = sample_bounding_box(corners, inter_type)
                                filtered_pois = filter_points_in_bounds(pois, min_bounds, max_bounds)
                                for poi in filtered_pois:
                                    uniform_pcd.append(poi)
                                    uniform_lbl.append(label_)

                    elif geometry_type == ogr.wkbTIN or geometry_type == ogr.wkbTINZ:
                        num_triangles = geometry.GetGeometryCount()
                        for i in range(num_triangles):
                            triangle = geometry.GetGeometryRef(i)
                            ring = triangle.GetGeometryRef(0)
                            corners = [ring.GetPoint(j) for j in range(ring.GetPointCount())]

                            inter_type = "gaussian"
                            pois = sample_bounding_box(corners, inter_type)
                            filtered_pois = filter_points_in_bounds(pois, min_bounds, max_bounds)
                            for poi in filtered_pois:
                                gauss_pcd.append(poi)
                                gauss_lbl.append(label_)
                            
                            inter_type = "random"
                            pois = sample_bounding_box(corners, inter_type)
                            filtered_pois = filter_points_in_bounds(pois, min_bounds, max_bounds)
                            for poi in filtered_pois:
                                random_pcd.append(poi)
                                random_lbl.append(label_)
                            
                            inter_type = "uniform"
                            pois = sample_bounding_box(corners, inter_type)
                            filtered_pois = filter_points_in_bounds(pois, min_bounds, max_bounds)
                            for poi in filtered_pois:
                                uniform_pcd.append(poi)
                                uniform_lbl.append(label_)
        
        uniform_points = np.array(uniform_pcd, dtype=np.float32)[:,:3]
        uniform_labels = np.array(uniform_lbl, dtype=np.int32)
        import ipdb 
        ipdb.set_trace()
        """
        gauss_points = np.array(gauss_pcd, dtype=np.float32)[:,:3]
        gauss_labels = np.array(gauss_lbl, dtype=np.int32)

        random_points = np.array(random_pcd, dtype=np.float32)[:,:3]
        random_labels = np.array(random_lbl, dtype=np.int32)

        uniform_points = np.array(uniform_pcd, dtype=np.float32)[:,:3]
        uniform_labels = np.array(uniform_lbl, dtype=np.int32)


        #save raw dataset for 3 different sampling types 
        save_ascii_for_cloudcompare(f'output/polygon/{idx:05d}_random.txt', random_points, random_labels)
        save_ascii_for_cloudcompare(f'output/polygon/{idx:05d}_uniform.txt', uniform_points, uniform_labels)
        save_ascii_for_cloudcompare(f'output/polygon/{idx:05d}_gaussian.txt', gausss_points, gauss_labels)
        import ipdb 
        ipdb.set_trace()
        
        """
        #
        
        
        
        """#gauss_pcd, gauss_lbl, gauss_syn_pcd, gauss_rmse, gauss_fitness = icp(original_pcd, gauss_points, gauss_lbl, 1, 15, 0.2, 100)
        #print(gauss_rmse, "for 1")
        #gauss_pcd, gauss_lbl, gauss_syn_pcd, gauss_rmse, gauss_fitness = icp(original_pcd, gauss_points, gauss_lbl, 2, 15, 0.2, 100)
        #print(gauss_rmse, "for 2")
        #gauss_pcd, gauss_lbl, gauss_syn_pcd, gauss_rmse, gauss_fitness = icp(original_pcd, gauss_points, gauss_lbl, 3, 15, 0.2, 100)
        #print(gauss_rmse, "for 3")
        random_pcd, random_lbl, random_syn_pcd, random_rmse, random_fitness = icp_align_to_synthetic(original_pcd, random_points, random_lbl, 2, 30, 0.2, 100)
        print("random rmse", random_rmse, "fitness", random_fitness, "for 2 downsample ratio")
        random_pcd, random_lbl, random_syn_pcd, random_rmse, random_fitness = icp_align_to_synthetic(original_pcd, random_points, random_lbl, 5, 30, 0.2, 100)
        print("random rmse", random_rmse, "fitness", random_fitness, "for 5 downsample ratio")
        #save_ascii_for_cloudcompare(f'output/new/{idx:05d}_gauss.txt', gauss_points, gauss_labels)
        #save_ascii_for_cloudcompare(f'output/new/{idx:05d}_random.txt', random_points, random_labels)
        #save_ascii_for_cloudcompare(f'output/new/{idx:05d}_uniform.txt', uniform_points, uniform_labels)
        import ipdb 
        ipdb.set_trace()
        #best_config, results = tune_icp(original_pcd, gauss_points, gauss_labels)
        print("Original point cloud has ", original_pcd.shape[0], " points")
        print("Gaussian-based synthetic point cloud has ", gauss_points.shape[0], " points")
        print("Random-sampled synthetic point cloud has ", random_points.shape[0], " points")
        print("Uniform-sampled synthetic point cloud has ", uniform_points.shape[0], " points")
        gauss_pcd, gauss_lbl, gauss_syn_pcd, gauss_rmse, gauss_fitness = icp(original_pcd, gauss_points, gauss_lbl, 5, 30, 0.1, 100)
        random_pcd, random_lbl, random_syn_pcd, random_rmse, random_fitness = icp(original_pcd, random_points, random_lbl, 5, 30, 0.1, 100)
        uniform_pcd, uniform_lbl, uniform_syn_pcd, uniform_rmse, uniform_fitness = icp(original_pcd,uniform_points, uniform_lbl, 5, 30, 0.1, 100) 
        print(f'After registration, Gaussian sampling has {gauss_rmse} matching')
        print(f'After registration, Random sampling has {random_rmse} matching')
        print(f'After registration,Uniform sampling has {uniform_rmse} matching')"""
        

        
        #pcd, lbl, syn_pcd, rmse, fitness = icp(original_pcd, synthetic_pcd, synthetic_lbl, 5, 30, 0.1, 100) 
   
        #save_ascii_for_cloudcompare(f'output/new/{idx:05d}_gauss.txt', gauss_pcd, gauss_lbl)
        #save_ascii_for_cloudcompare(f'output/new/random/{idx:05d}_random.txt', random_points, random_labels)
        #save_ascii_for_cloudcompare(f'output/new/uniform/{idx:05d}_uniform.txt', uniform_points, uniform_labels)
        
        #random_pcd, random_lbl, random_syn_pcd, random_rmse, random_fitness = icp(original_pcd, random_points, random_labels, 5, 30, 0.1, 100, "plane")
        #uniform_pcd, uniform_lbl, uniform_syn_pcd, uniform_rmse, uniform_fitness = icp(original_pcd,uniform_points, uniform_labels, 5, 30, 0.1, 100,"plane") 
        #print("random pcd for point2plane", random_rmse, " ", random_fitness)
        #print("uniform pcd for point2plane", uniform_rmse, " ", uniform_fitness)

        
        #random_pcd, random_lbl, random_syn_pcd, random_rmse, random_fitness = icp_syn(original_pcd, random_points, random_labels, 5, 30, 0.1, 100, "point")
        
        #uniform_pcd, uniform_lbl, uniform_syn_pcd, uniform_rmse, uniform_fitness = icp_syn(original_pcd, uniform_points, uniform_labels, 5, 30, 0.1, 100, "point")
        
        #gauss_pcd, gauss_lbl, gauss_syn_pcd, gauss_rmse, gauss_fitness = icp_syn(original_pcd, gauss_points, gauss_labels, 5, 30, 0.1, 100, "point")
        
        #tune_icp(original_pcd, uniform_points, uniform_labels)
        #import ipdb
        #ipdb.set_trace
        #save_ascii_for_cloudcompare(f'output/new/{idx:05d}_gauss.txt', gauss_pcd, gauss_lbl)
        
        save_ascii_for_cloudcompare(f'output/new/registered/{idx:05d}_uniform_registered.txt', uniform_pcd, uniform_lbl)
        
        #save_ascii_for_cloudcompare(f'output/new/uniform/{idx:05d}_uniform.txt', uniform_points, uniform_labels)
        print("random pcd for synthetic point cloud as reference", random_rmse, " ", random_fitness)

def save_bin_file(filepath, points):
    """
    Saves a point cloud to a .bin file in float32 format (x, y, z, intensity).

    Args:
        filepath (str): Output path for the .bin file.
        points (np.ndarray): Nx3 or Nx4 array of point cloud data.
    """
    # Ensure points are Nx4: if only XYZ, add dummy intensity
    if points.shape[1] == 3:
        intensity = np.zeros((points.shape[0], 1), dtype=np.float32)
        points = np.hstack((points, intensity))

    points.astype(np.float32).tofile(filepath)





def getsingleBuild(xodr_path, pcd_dir, output_dir):
    driver = ogr.GetDriverByName('XODR')
    options = ["DISSOLVE_TIN=True", "EPSILON=0.2"]
    source = gdal.OpenEx(xodr_path, gdal.OF_VECTOR, open_options=options)
    
    if source is None:
        raise Exception(f"Cannot open file: {xodr_path}")
    
    layers = source.GetLayerCount()
    pcd_files = [f for f in os.listdir(pcd_dir) if f.endswith('.las')]
   
    for idx, pcd_path in enumerate(pcd_files):
        pcd_path = os.path.join(pcd_dir, pcd_path)
        
        gauss_pcd = []
        gauss_lbl = []

        uniform_pcd = []
        uniform_lbl = []

        random_pcd = []
        random_lbl = []
        

        original_pcd = las2array(pcd_path)

        min_bounds, max_bounds = get_point_cloud_bounds(original_pcd)

        for i in tqdm(range(layers), desc="Processing Layers"):
            layer = source.GetLayerByIndex(i)
            layer_name = layer.GetName()
            """if layer_name in [ "RoadObject"]:#["Lane", "RoadObject", "RoadMark"]:
                features = list(layer)
                for feature in tqdm(features, desc=f"Processing Features in {layer_name}", leave=False):
                    geometry = feature.GetGeometryRef()
                    envelope = geometry.GetEnvelope()
                    x_min, x_max = envelope[0], envelope[1]
                    y_min, y_max = envelope[2], envelope[3]

                    z_min, z_max = min_bounds[2], max_bounds[2]
                    
                    mask = (
                        (original_pcd[:, 0] >= min_bounds[0]) & (original_pcd[:, 0] <= max_bounds[0]) &
                        (original_pcd[:, 1] >= min_bounds[1]) & (original_pcd[:, 1] <= max_bounds[1]) &
                        (original_pcd[:, 2] >= min_bounds[2]) & (original_pcd[:, 2] <= max_bounds[2])
                    )
                    building_points = original_pcd[mask]

                    if layer_name == "RoadObject":
                        type_ = feature.type 
                        name = feature.name
                        if type_ == "building":
                            label_ = 1

                            geometry_type = geometry.GetGeometryType()

                            if geometry_type == ogr.wkbTIN or geometry_type == ogr.wkbTINZ:
                                num_triangles = geometry.GetGeometryCount()
                                for i in range(num_triangles):
                                    triangle = geometry.GetGeometryRef(i)
                                    ring = triangle.GetGeometryRef(0)
                                    corners = [ring.GetPoint(j) for j in range(ring.GetPointCount())]

                                    inter_type = "gaussian"
                                    pois = sample_bounding_box(corners, inter_type)
                                    filtered_pois = filter_points_in_bounds(pois, min_bounds, max_bounds)
                                    for poi in filtered_pois:
                                        gauss_pcd.append(poi)
                                        gauss_lbl.append(label_)

                                    inter_type = "uniform"
                                    pois = sample_bounding_box(corners, inter_type)
                                    filtered_pois = filter_points_in_bounds(pois, min_bounds, max_bounds)
                                    for poi in filtered_pois:
                                        uniform_pcd.append(poi)
                                        uniform_lbl.append(label_)

                                    inter_type = "random"
                                    pois = sample_bounding_box(corners, inter_type)
                                    filtered_pois = filter_points_in_bounds(pois, min_bounds, max_bounds)
                                    for poi in filtered_pois:
                                        random_pcd.append(poi)
                                        random_lbl.append(label_)"""
            if layer_name == "RoadObject":
                features = list(layer)
                for feature in tqdm(features, desc="Processing Building Features", leave=False):
                    if feature.type != "building":
                        continue 

                    geometry = feature.GetGeometryRef()
                    if geometry is None:
                        continue

                    geometry_type = geometry.GetGeometryType()
                    if geometry_type not in [ogr.wkbTIN, ogr.wkbTINZ]:
                        continue 


                    envelope = geometry.GetEnvelope()
                    x_min, x_max = envelope[0], envelope[1]
                    y_min, y_max = envelope[2], envelope[3]
                    z_min, z_max = min_bounds[2], max_bounds[2]  

                    local_mask = (
                        (original_pcd[:, 0] >= x_min) & (original_pcd[:, 0] <= x_max) &
                        (original_pcd[:, 1] >= y_min) & (original_pcd[:, 1] <= y_max) &
                        (original_pcd[:, 2] >= z_min) & (original_pcd[:, 2] <= z_max)
                    )
                    building_points_local = original_pcd[local_mask]

                    if 'building_points_all' not in locals():
                        building_points_all = []
                    building_points_all.append(building_points_local)

                    label_ = 1 

                    num_triangles = geometry.GetGeometryCount()
                    for i in range(num_triangles):
                        triangle = geometry.GetGeometryRef(i)
                        ring = triangle.GetGeometryRef(0)
                        corners = [ring.GetPoint(j) for j in range(ring.GetPointCount())]

                        for inter_type, point_list, label_list in zip(
                            ["gaussian", "uniform", "random"],
                            [gauss_pcd, uniform_pcd, random_pcd],
                            [gauss_lbl, uniform_lbl, random_lbl]
                        ):
                            sampled = sample_bounding_box(corners, inter_type)
                            filtered = filter_points_in_bounds(sampled, [x_min, y_min, z_min], [x_max, y_max, z_max])
                            point_list.extend(filtered)
                            label_list.extend([label_] * len(filtered))

        gauss_points = np.array(gauss_pcd, dtype=np.float32)
        gauss_points = gauss_points[:,:3]
        gauss_labels = np.array(gauss_lbl, dtype=np.int32)
        
        uniform_points = np.array(uniform_pcd, dtype=np.float32)
        uniform_points = uniform_points[:,:3]
        uniform_labels = np.array(uniform_lbl, dtype=np.int32)

        random_points = np.array(random_pcd, dtype=np.float32)
        random_points = random_points[:,:3]
        random_labels = np.array(random_lbl, dtype=np.int32)
        import ipdb 
        ipdb.set_trace()
        #save_bin_file(f'output/build/{idx:05d}_gauss.bin', gauss_points)
        #save_bin_file(f'output/build/{idx:05d}_uniform.bin', uniform_points)
        #save_bin_file(f'output/build/{idx:05d}_random.bin', random_points)
        #save_bin_file(f'output/build/{idx:05d}_building_original.bin', building_points[:, :3])
        #np.savez(f'output/build/{idx:05d}_gauss.npz', points=gauss_points, labels=gauss_labels)
        #np.savez(f'output/build/{idx:05d}_uniform.npz', points=uniform_points, labels=uniform_labels)
        #np.savez(f'output/build/{idx:05d}_random.npz', points=random_points, labels=random_labels)
        #np.savez( f'output/build/{idx:05d}_building_original.npz', points=building_points_local[:, :3], labels=np.full(len(building_points_local), 1, dtype=np.int32))
        #save_ascii_for_cloudcompare(f'output/build/{idx:05d}_gauss.txt', gauss_points, gauss_labels)
        #save_ascii_for_cloudcompare(f'output/build/{idx:05d}_uniform.txt', uniform_points, uniform_labels)
        #save_ascii_for_cloudcompare(f'output/build/{idx:05d}_random.txt', random_points, random_labels)
        #save_ascii_for_cloudcompare(f'output/build/{idx:05d}_building_original.txt', building_points[:, :3], building_labels)

        import ipdb 
        ipdb.set_trace()
        