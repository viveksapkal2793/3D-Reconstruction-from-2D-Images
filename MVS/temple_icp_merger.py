#!/usr/bin/env python3
# temple_icp_merger.py - Script to align and merge temple point clouds using ICP

import numpy as np
import open3d as o3d
import os
import glob
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import cv2
import argparse

def load_point_cloud(filepath):
    """Load a point cloud from a PLY file."""
    print(f"Loading point cloud: {filepath}")
    try:
        pcd = o3d.io.read_point_cloud(filepath)
        print(f"  Loaded {len(np.asarray(pcd.points))} points")
        return pcd
    except Exception as e:
        print(f"Error loading point cloud {filepath}: {e}")
        return None

def align_point_clouds(source, target, threshold=0.02, initial_transformation=None):
    """Align source point cloud to target using ICP."""
    if initial_transformation is None:
        # Use identity matrix as initial guess
        initial_transformation = np.eye(4)
    
    # Run point-to-point ICP
    print(f"Running ICP with threshold {threshold}...")
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    print(f"ICP completed with fitness: {result.fitness}, RMSE: {result.inlier_rmse}")
    return result.transformation

def merge_point_clouds(point_clouds, transformations):
    """Merge multiple point clouds using provided transformations."""
    merged_pcd = o3d.geometry.PointCloud()
    
    for i, pcd in enumerate(point_clouds):
        # Transform point cloud to global coordinate system
        transformed_pcd = pcd.transform(transformations[i])
        # Add to merged point cloud
        merged_pcd += transformed_pcd
    
    print(f"Merged point cloud has {len(np.asarray(merged_pcd.points))} points")
    return merged_pcd

def filter_point_cloud(pcd, voxel_size=0.005, remove_outliers=True):
    """Filter and clean up point cloud."""
    # Voxel downsampling to reduce density and redundancy
    print(f"Downsampling with voxel size {voxel_size}...")
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    
    # Statistical outlier removal
    if remove_outliers and len(np.asarray(downsampled_pcd.points)) > 100:
        print("Removing outliers...")
        try:
            cleaned_pcd, _ = downsampled_pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0)
            return cleaned_pcd
        except Exception as e:
            print(f"Outlier removal failed: {e}")
            return downsampled_pcd
    
    return downsampled_pcd

def create_mesh_from_point_cloud(pcd, depth=9):
    """Create a surface mesh from point cloud using Poisson reconstruction."""
    # Estimate normals if they don't exist
    if not pcd.has_normals():
        print("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=20)
    
    # Poisson surface reconstruction
    print(f"Performing Poisson reconstruction with depth {depth}...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=0, scale=1.1, linear_fit=False)
    
    # Filter low-density vertices
    print("Filtering low-density vertices...")
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Smooth the mesh
    print("Smoothing mesh...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=20)
    
    return mesh

def save_visualization(geometry, output_path, is_mesh=False):
    """Save a visualization of the geometry (point cloud or mesh)."""
    try:
        # Create off-screen renderer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1920, height=1080)
        vis.add_geometry(geometry)
        
        # Configure rendering settings
        opt = vis.get_render_option()
        if opt is not None:
            opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
            
            if is_mesh:
                opt.mesh_show_back_face = True
                opt.mesh_show_wireframe = True
                opt.line_width = 0.5
            else:
                opt.point_size = 2.0
        
        # Setup camera view
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        
        # Capture and save image
        vis.capture_screen_image(output_path, do_render=True)
        vis.destroy_window()
        print(f"Visualization saved to {output_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")

def save_multiview_visualizations(pcd, output_folder, color_by='height'):
    """Save multiple views of the point cloud with coloring."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract points data
    points = np.asarray(pcd.points)
    
    # Determine coloring
    if color_by == 'height':
        colors = points[:, 2]  # Z-coordinate
        cmap_name = 'viridis'
        title = 'Height (Z)'
    elif color_by == 'distance':
        centroid = np.mean(points, axis=0)
        colors = np.linalg.norm(points - centroid, axis=1)
        cmap_name = 'plasma'
        title = 'Distance from Center'
    else:
        # Default to height
        colors = points[:, 2]
        cmap_name = 'viridis'
        title = 'Height (Z)'
    
    # Normalize colors for better visualization
    colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))
    
    # Apply colormap to get RGB values
    cmap = plt.get_cmap(cmap_name)
    color_values = cmap(colors)[:, :3]
    
    # Assign colors to point cloud
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(points)
    colored_pcd.colors = o3d.utility.Vector3dVector(color_values)
    
    # Define view angles for different perspectives
    views = {
        'perspective': {'front': [0.5, 0.5, -0.5], 'lookat': [0, 0, 0], 'up': [0, -1, 0], 'zoom': 0.7},
        'front': {'front': [0, 0, -1], 'lookat': [0, 0, 0], 'up': [0, -1, 0], 'zoom': 0.7},
        'side': {'front': [1, 0, 0], 'lookat': [0, 0, 0], 'up': [0, -1, 0], 'zoom': 0.7},
        'top': {'front': [0, -1, 0], 'lookat': [0, 0, 0], 'up': [0, 0, -1], 'zoom': 0.7},
        'isometric1': {'front': [0.7, -0.7, -0.7], 'lookat': [0, 0, 0], 'up': [0, -1, 0], 'zoom': 0.6},
        'isometric2': {'front': [-0.7, -0.7, -0.7], 'lookat': [0, 0, 0], 'up': [0, -1, 0], 'zoom': 0.6}
    }
    
    # Save visualizations for each view
    for view_name, view_params in views.items():
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=1920, height=1080)
            vis.add_geometry(colored_pcd)
            
            # Set rendering options
            opt = vis.get_render_option()
            if opt is not None:
                opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
                opt.point_size = 2.0
            
            # Set camera view
            ctr = vis.get_view_control()
            ctr.set_front(view_params['front'])
            ctr.set_lookat(view_params['lookat'])
            ctr.set_up(view_params['up'])
            ctr.set_zoom(view_params['zoom'])
            
            # Save image
            output_path = os.path.join(output_folder, f'temple_{view_name}_{color_by}.png')
            vis.capture_screen_image(output_path, do_render=True)
            vis.destroy_window()
            print(f"Saved {view_name} view to {output_path}")
        except Exception as e:
            print(f"Error saving {view_name} view: {e}")

def main():
    parser = argparse.ArgumentParser(description='Align and merge temple point clouds using ICP')
    parser.add_argument('--input_dir', type=str, default='Data/Output', 
                        help='Directory containing input PLY files')
    parser.add_argument('--output_dir', type=str, default='Data/Output/Merged', 
                        help='Directory to save output files')
    parser.add_argument('--voxel_size', type=float, default=0.005, 
                        help='Voxel size for downsampling')
    parser.add_argument('--threshold', type=float, default=0.02, 
                        help='Distance threshold for ICP')
    parser.add_argument('--create_mesh', action='store_true', 
                        help='Create mesh from point cloud')
    parser.add_argument('--mesh_depth', type=int, default=9, 
                        help='Depth for Poisson reconstruction')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to {args.output_dir}")
    
    # Find all PLY files in the input directory
    ply_files = sorted(glob.glob(os.path.join(args.input_dir, "temple_*.ply")))
    if not ply_files:
        print(f"No PLY files found in {args.input_dir}")
        return
    
    print(f"Found {len(ply_files)} PLY files")
    for f in ply_files:
        print(f"  {os.path.basename(f)}")
    
    # Load point clouds
    point_clouds = []
    for ply_file in ply_files:
        pcd = load_point_cloud(ply_file)
        if pcd is not None:
            point_clouds.append(pcd)
    
    if len(point_clouds) < 2:
        print("Need at least 2 point clouds for alignment")
        return
    
    # Pre-process point clouds to improve alignment
    processed_clouds = []
    for i, pcd in enumerate(point_clouds):
        print(f"Processing point cloud {i+1}/{len(point_clouds)}...")
        # Downsample first to speed up processing
        processed = pcd.voxel_down_sample(args.voxel_size)
        processed_clouds.append(processed)
    
    # Align point clouds sequentially
    # Use the first cloud as reference
    transformations = [np.eye(4)]  # Identity matrix for the first point cloud
    
    # Align each subsequent cloud to the growing merged cloud
    merged_cloud = processed_clouds[0]
    
    for i in range(1, len(processed_clouds)):
        print(f"Aligning point cloud {i+1} to the merged cloud...")
        # Align current cloud to the growing merged cloud
        transformation = align_point_clouds(
            processed_clouds[i], merged_cloud, 
            threshold=args.threshold)
        
        transformations.append(transformation)
        
        # Add aligned cloud to the merged cloud
        transformed_cloud = processed_clouds[i].transform(transformation)
        merged_cloud += transformed_cloud
        
        # Downsample merged cloud to keep it manageable
        merged_cloud = merged_cloud.voxel_down_sample(args.voxel_size)
    
    # Now merge the original (unprocessed) point clouds using the calculated transformations
    final_merged_cloud = merge_point_clouds(point_clouds, transformations)
    
    # Clean up the final merged cloud
    cleaned_cloud = filter_point_cloud(
        final_merged_cloud, 
        voxel_size=args.voxel_size, 
        remove_outliers=True)
    
    # Save the merged point cloud in ASCII format
    output_path = os.path.join(args.output_dir, "temple_merged.ply")
    print(f"Saving merged point cloud to {output_path} in ASCII format...")
    o3d.io.write_point_cloud(output_path, cleaned_cloud, write_ascii=True)
    print(f"Merged point cloud saved to {output_path}")
    
    # Try to save visualizations with error handling
    try:
        save_multiview_visualizations(cleaned_cloud, args.output_dir, color_by='height')
        save_multiview_visualizations(cleaned_cloud, args.output_dir, color_by='distance')
    except Exception as e:
        print(f"Error saving visualizations: {e}")
        print("Continuing with mesh creation...")
    
    # Create and save mesh if requested
    if args.create_mesh:
        print("Creating mesh from point cloud...")
        try:
            mesh = create_mesh_from_point_cloud(cleaned_cloud, depth=args.mesh_depth)
            
            # Save mesh
            mesh_path = os.path.join(args.output_dir, "temple_mesh.obj")
            o3d.io.write_triangle_mesh(mesh_path, mesh)
            print(f"Mesh saved to {mesh_path}")
            
            # Save mesh visualization
            try:
                mesh_vis_path = os.path.join(args.output_dir, "temple_mesh_visualization.png")
                save_visualization(mesh, mesh_vis_path, is_mesh=True)
            except Exception as e:
                print(f"Error saving mesh visualization: {e}")
        except Exception as e:
            print(f"Error creating mesh: {e}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()