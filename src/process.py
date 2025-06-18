

from data import *
def las2array(file_path):
    las = laspy.read(file_path)
    points = np.column_stack((
        np.asarray(las.x, dtype=np.float64),
        np.asarray(las.y, dtype=np.float64),
        np.asarray(las.z, dtype=np.float64),
        np.asarray(las.intensity, dtype=np.float64)
    ))
    return points
def save_icp_metrics_txt(
    filepath,
    num_points,
    total_rmse,
    total_fitness,
    class_rmse
):
    with open(filepath, "w") as f:
        f.write("==== ICP Registration Metrics ====\n")
        f.write(f"Total Points in Result: {num_points}\n")
        f.write(f"Total RMSE: {total_rmse:.4f}\n")
        f.write(f"Total Fitness: {total_fitness:.4f}\n\n")

        f.write("Class-based RMSE:\n")
        for label in sorted(class_rmse.keys()):
            f.write(f"  Class {label}: RMSE = {class_rmse[label]:.4f}\n")

    print(f"\nMetrics saved to {filepath}")

def getsingleXODR_asil(xodr_path, pcd_dir, output_dir):
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
                        """
                        {
                            "-1": "unknown",
                            "0": "roadmark",
                            "1": "parking",
                            "2": "obstacle",
                            "3": "vegetation",
                            "4": "tree",
                            "5": "streetLamp",
                            "6": "barrier",
                            "7": "trafficSign",
                            "8": "building",
                            "9": "pole",
                            "10": "trafficLight",
                            "11": "driving",
                            "19": "restricted"

                        }
                        """
                        if type_ == "barrier":
                            label_ = 6
                        elif type_ == "pole":
                            if name == "trafficSign":
                                label_ = 7 
                            elif name == "trafficLight":
                                label_ = 10 
                            elif name == "streetLamp":
                                label_ = 5
                            else:
                                label_ = 9
                        elif type_ == "roadMark":
                            label_ = 0
                        elif type_ == "tree":
                            label_ = 4
                        elif type_ == "vegetation":
                            label_ = 3
                        elif type_ == "parkingSpace":
                            label_ = 1
                        elif type_ == "obstacle":
                            label_ = 2
                        elif type_ == "building":
                            label_ = 8
                        else:
                            label_ = -1
                    elif layer_name == "Lane":
                        type_ = feature.type
                        #if type_ == "biking":
                        #    label_ 17 
                        #elif type_ == "tram":
                        #    label_ = 18
                        #elif type_ == "sidewalk":
                        #    label_ == 12
                        if type_ == "birectional" or type_ == "border" or type_ == "curb" or type_ == "driving":
                            label_ = 11
                        elif type_ == "parking":
                            label_ = 1
                        else: # undriving
                            label_ = 19 
                    elif layer_name == "RoadMark":   
                        label_ = 0
                    else: 
                        label_ = -1
                        
                
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

        random_points = np.array(random_pcd, dtype=np.float32)[:,:3]
        random_labels = np.array(random_lbl, dtype=np.int32)

        gauss_points = np.array(gauss_pcd, dtype=np.float32)[:,:3]
        gauss_labels = np.array(gauss_lbl, dtype=np.int32)

        #save_ascii_for_cloudcompare(f'output/process/{idx:05d}_raw_random.txt', random_points, random_labels)
        #save_ascii_for_cloudcompare(f'output/process/{idx:05d}_raw_uniform.txt', uniform_points, uniform_labels)
        #save_ascii_for_cloudcompare(f'output/process/{idx:05d}_raw_gaussian.txt', gauss_points, gauss_labels)
        
        #dynamic_icp_finetune(original_pcd, uniform_points, uniform_labels)
        #import ipdb 
        #ipdb.set_trace()
        uniform_pcd_poly, uniform_lbl_poly, uniform_syn_poly, uniform_rmse_poly, uniform_fitness_poly = icp_syn(original_pcd, uniform_points, uniform_labels, 2, 30, 0.2, 10, "point")
        #uniform_pcd_tin, uniform_lbl_tin, uniform_syn_pcd_tin, uniform_rmse_tin, uniform_fitness_tin = icp_syn(original_pcd, uniform_points, uniform_labels, 2, 30, 0.2, 10, "point")
        #
        #uniform_pcd_poly_plane, uniform_lbl_poly_plane, uniform_syn_poly_plane, uniform_rmse_poly_plane, uniform_fitness_poly_plane = icp_syn(original_pcd, uniform_points, uniform_labels, 2, 30, 0.2, 10, "plane")
        #uniform_pcd_tin_plane, uniform_lbl_tin_plane, uniform_syn_pcd_tin_plane, uniform_rmse_tin_plane, uniform_fitness_tin_plane = icp_syn(original_pcd, uniform_points, uniform_labels, 2, 30, 0.2, 10, "plane")
        import ipdb 
        ipdb.set_trace()
        random_pcd, random_lbl, random_syn_pcd, random_rmse, random_fitness = icp_syn(original_pcd, random_points, random_labels, 2, 30, 0.2, 10, "point")
        gauss_pcd, gauss_lbl, gauss_syn_pcd, gauss_rmse, gauss_fitness = icp_syn(original_pcd, gauss_points, gauss_labels, 2, 30, 0.2, 10, "point")
        import ipdb 
        ipdb.set_trace()
        save_ascii_for_cloudcompare(f'output/process/{idx:05d}_icp_random.txt', random_pcd, random_lbl)
        save_ascii_for_cloudcompare(f'output/process/{idx:05d}_icp_uniform.txt', uniform_pcd, uniform_lbl)
        save_ascii_for_cloudcompare(f'output/process/{idx:05d}_icp_random.txt', gauss_pcd, gauss_lbl)
        
        data = np.hstack((uniform_pcd_poly, uniform_lbl_poly.reshape(-1, 1)))
        np.savetxt('/mnt/data/bard_gu/pcd/00000_icp_parking_uniform.txt', data, fmt='%.6f %.6f %.6f %.6f %d')