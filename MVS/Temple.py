from load_camera_info import load_intrinsics, load_extrinsics
from load_camera_info_temple import load_all_camera_parameters_temple
from load_ply import save_ply
import numpy as np
from pathlib import Path
# from IPython.display import display
# import inspect
import cv2
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import numpy as np
import os
from PIL import Image
import collections
from utils import *
# import glob
# from pyntcloud import PyntCloud
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
# from ipywidgets import interact, interactive, fixed

def compute_disparity(image, img_pair, num_disparities=6*16, block_size=11, window_size=6, uniqueness_ratio=0, speckleWindowSize=200, matcher="stereo_sgbm", show_disparity=True):
    if matcher == "stereo_bm":
        new_image = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
        new_image.setPreFilterType(1)
        new_image.setUniquenessRatio(uniqueness_ratio)
        new_image.setSpeckleRange(2)
        new_image.setSpeckleWindowSize(speckleWindowSize)
    elif matcher == "stereo_sgbm":
        new_image = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disparities, blockSize=block_size,
                                         uniquenessRatio=uniqueness_ratio, speckleWindowSize=speckleWindowSize, speckleRange=2, disp12MaxDiff=1,
                                         P1=8 * 1 * window_size **2, P2=32 * 1 * window_size **2)

    new_image = new_image.compute(image, img_pair).astype(np.float32) / 16

    if (show_disparity == True):
        plt.figure(figsize=(20, 10))
        plt.imshow(new_image, cmap="plasma")
        plt.show()
    return new_image

def rotate_images_anticlockwise(folder_path):
    # Get a list of all files in the folder
    all_files = os.listdir(folder_path)

    # Filter only the image files
    image_files = [file for file in all_files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Rotate each image 90 degrees anticlockwise and save it back
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        try:
            image = Image.open(image_path)
            rotated_image = image.transpose(Image.ROTATE_270)
            rotated_image.save(image_path)
        except Exception as e:
            print(f"Error rotating the image {image_file}: {e}")

def get_disparity_temple(options):
    """
    Create a stereo matcher based on the provided options.
    
    Args:
        options (dict): Dictionary containing matcher parameters.
        
    Returns:
        cv2.StereoMatcher: Configured stereo matcher.
    """
    # Extract parameters
    matcher_params = options['StereoMatcher']
    
    # Create SGBM matcher (Semi-Global Block Matching)
    matcher = cv2.StereoSGBM_create(
        minDisparity=matcher_params['MinDisparity'],
        numDisparities=matcher_params['NumDisparities'],
        blockSize=matcher_params['BlockSize'],
        P1=options['StereoSGBM']['P1'] * matcher_params['BlockSize']**2,
        P2=options['StereoSGBM']['P2'] * matcher_params['BlockSize']**2,
        disp12MaxDiff=matcher_params['Disp12MaxDiff'],
        preFilterCap=matcher_params['PreFilterCap'],
        uniquenessRatio=matcher_params['UniquenessRatio'],
        speckleWindowSize=matcher_params['SpeckleWindowSize'],
        speckleRange=matcher_params['SpeckleRange']
    )
    
    return matcher

class OpenCVStereoMatcher():
    global FinalOptions, folder_path
    def __init__(self, options=None, calibration_path=None):
            """
            This class initializes an OpenCV stereo matcher for multi-camera stereo vision using provided options and calibration data.

            Parameters:
            - options: A dictionary containing various options for stereo matching and rectification (default is FinalOptions).
            - calibration_path: The path to the calibration data containing camera parameters (optional).

            Initialization Steps:
            1. Load all camera parameters from the calibration data (if provided).
            2. Initialize arrays to store stereo calibration results, rectification maps, and disparity-to-depth mapping matrices (Q) for each camera pair.
            3. Loop through each camera pair based on the specified topology and perform the following steps:
                a. Get camera intrinsic and extrinsic parameters for the left and right cameras.
                b. Perform stereo calibration and rectification to obtain Q, extrinsics_left_rectified_to_global, left_maps, and right_maps.
                c. Get the stereo matcher based on the provided options.
            """
            self.options = options if options is not None else FinalOptions
            self.calibration_path = folder_path
            self.num_cameras = options['CameraArray']['num_cameras']
            self.topology = options['CameraArray']['topology']
            self.all_camera_parameters = load_all_camera_parameters_temple(folder_path)

            self.left_maps_array = []
            self.right_maps_array = []
            self.Q_array = []
            self.extrinsics_left_rectified_to_global_array = []

            for pair_index, (left_index, right_index) in enumerate(topologies[self.topology]):
                # 1 — Get R, T, W, H for each camera
                left_K, left_R, left_T, left_width, left_height = [self.all_camera_parameters[left_index][key] for key
                                                                   in ('camera_matrix', 'R', 'T', 'image_width',
                                                                       'image_height')]
                right_K, right_R, right_T, right_width, right_height = [self.all_camera_parameters[right_index][key] for
                                                                        key in (
                                                                        'camera_matrix', 'R', 'T', 'image_width',
                                                                        'image_height')]
                h, w = left_height, left_width

                # 2 — Stereo Calibrate & Rectify
                Q, extrinsics_left_rectified_to_global, left_maps, right_maps = calibrate_and_rectify(options, left_K,
                                                                                                      right_K,
                                                                                                      left_R, right_R,
                                                                                                      left_T, right_T)
                self.Q_array.append(Q)
                self.extrinsics_left_rectified_to_global_array.append(extrinsics_left_rectified_to_global)
                self.left_maps_array.append(left_maps)
                self.right_maps_array.append(right_maps)

                # 3 — Get Matcher
                self.matcher = get_disparity_temple(options)

    def load_images(self, folder_path):
        """
        Load temple images based on the names in camera parameters.
        
        Args:
            folder_path (Path): Path to the directory containing images.
        """
        # Resolve the provided path to an absolute path
        imagesPath = folder_path.resolve()
        
        # Get all image files in the folder
        image_files = list(imagesPath.glob('*.png')) + list(imagesPath.glob('*.jpg'))
        
        # Sort image files to ensure consistent order
        image_files.sort()
        
        print(f'Found {len(image_files)} images in {imagesPath}')
        
        # Assert number of cameras matches
        if len(image_files) != self.num_cameras:
            print(f"Warning: Expected {self.num_cameras} cameras, but found {len(image_files)} images.")
            # Adjust number of cameras based on actual images found
            self.num_cameras = min(self.num_cameras, len(image_files))
        
        # Load images
        images = []
        for i in range(self.num_cameras):
            # Try to match the image name from camera parameters
            expected_name = self.all_camera_parameters[i]['name']
            image_path = imagesPath / expected_name
            
            # If file not found by name, use position
            if not image_path.exists() and i < len(image_files):
                image_path = image_files[i]
                print(f"Using {image_path.name} for camera {i} (expected {expected_name})")
            
            print(f"Loading image {image_path}")
            
            # Load and convert image
            if image_path.exists():
                colorImage = cv2.imread(str(image_path))
                if colorImage is None:
                    print(f"Warning: Failed to load {image_path}")
                    # Create blank image as placeholder
                    colorImage = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                print(f"Warning: Image {image_path} not found")
                # Create blank image as placeholder
                colorImage = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Convert to grayscale
            grayImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
            
            # Add to list
            images.append(grayImage)
        
        # Store loaded images
        self.images = images

    def run(self):
        # Check if camera parameters have been loaded
        assert self.all_camera_parameters is not None, 'Camera parameters not loaded yet; You should run load_all_camera_parameters first!'

        # Create an array to store the 3D points for each pair of images
        xyz_global_array = [None] * len(topologies[self.topology])

        def run_pair(pair_idx, left_idx, right_idx):
            # Load the proper images and rectification maps
            left_img, right_img = self.images[left_idx], self.images[right_idx]
            left_maps = self.left_maps_array[pair_idx]
            right_maps = self.right_maps_array[pair_idx]

            # Rectify the images
            remap_interpolation = self.options['Remap']['interpolation']
            left_image_rectified = cv2.remap(left_img, left_maps[0], left_maps[1], remap_interpolation)
            right_image_rectified = cv2.remap(right_img, right_maps[0], right_maps[1], remap_interpolation)

            # Load & Find Disparity
            disparity_image = self.matcher.compute(left_image_rectified, right_image_rectified)

            # Convert the disparity image to floating-point format
            if disparity_image.dtype == np.int16:
                disparity_image = disparity_image.astype(np.float32)
                disparity_image /= 16

            # Reproject 3D points from the disparity map using the Q-matrix
            Q = self.Q_array[pair_idx]
            threedeeimage = cv2.reprojectImageTo3D(disparity_image, Q, handleMissingValues=True, ddepth=cv2.CV_32F)
            threedeeimage = np.array(threedeeimage)

            # Postprocess the 3D points
            xyz = threedeeimage.reshape((-1, 3))  # x, y, z now in three columns, in left rectified camera coordinates
            z = xyz[:, 2]
            goodz = z < 9999.0
            xyz_filtered = xyz[goodz, :]

            # Transform the 3D points to global coordinates
            R_left_rectified_to_global, T_left_rectified_to_global = self.extrinsics_left_rectified_to_global_array[
                pair_idx]
            xyz_global = np.dot(xyz_filtered, R_left_rectified_to_global.T) + T_left_rectified_to_global.T

            # Save PLY file for the pair of images
            #save_ply(xyz_global, 'pair_' + str(left_index) + '_' + str(right_index) + '.ply', output_folder)
            xyz_global_array[pair_index] = xyz_global

        # Process each pair of images
        for pair_index, (left_index, right_index) in enumerate(topologies[self.topology]):
            run_pair(pair_index, left_index, right_index)

        # Stack all the 3D points from different pairs into one array
        xyz = np.vstack(xyz_global_array)
        return xyz


def rectify_and_show_results(opencv_matcher, image_index=0, show_image=True, output_folder=None):
    """Rectifies the input stereo images and displays the results.

    Args:
        opencv_matcher (OpenCVStereoMatcher): The OpenCVStereoMatcher instance.
        image_index (int, optional): The index of the stereo image pair to rectify and show results. Defaults to 0.
        show_image (bool, optional): Whether to display the results or not. Defaults to True.
        output_folder (str, optional): Folder to save visualization images if not showing them. Defaults to None.

    Returns:
        tuple: A tuple containing the rectified left and right images.
    """

    # Get the Images
    img0 = opencv_matcher.images[image_index]
    img1 = opencv_matcher.images[image_index + 1]

    # Get the Maps
    left_maps = opencv_matcher.left_maps_array[image_index]
    right_maps = opencv_matcher.right_maps_array[image_index]

    # Rectify with the parameters
    remap_int = opencv_matcher.options['Remap']['interpolation']
    left_image_rectified = cv2.remap(img0, left_maps[0], left_maps[1], remap_int)
    right_image_rectified = cv2.remap(img1, right_maps[0], right_maps[1], remap_int)

    if show_image:
        # Show Results
        f, (f0, f1, f2) = plt.subplots(1, 3, figsize=(20, 10))
        f0.imshow(img0)
        f0.set_title('Original Left Image')
        f1.imshow(left_maps[1])
        f1.set_title('Left Disparity Map')
        f2.imshow(left_image_rectified)
        f2.set_title('Rectified Left Image')
        
        if output_folder:
            plt.savefig(os.path.join(output_folder, 'rectification_left.png'), dpi=300)
        else:
            plt.savefig('rectification_left.png', dpi=300)

        f, (f3, f4, f5) = plt.subplots(1, 3, figsize=(20, 10))
        f3.imshow(img1)
        f3.set_title('Original Right Image')
        f4.imshow(right_maps[1])
        f4.set_title('Right Disparity Map')
        f5.imshow(right_image_rectified)
        f5.set_title('Rectified Right Image')
        
        if output_folder:
            plt.savefig(os.path.join(output_folder, 'rectification_right.png'), dpi=300)
        else:
            plt.savefig('rectification_right.png', dpi=300)

    return left_image_rectified, right_image_rectified



def compute_and_show_disparity(opencv_matcher, left_image_rectified, right_image_rectified, show_image=True):
    """Computes the disparity map from rectified left and right images using the stereo matcher and displays it.

    Args:
        left_image_rectified (numpy.ndarray): The rectified left image.
        right_image_rectified (numpy.ndarray): The rectified right image.
        show_image (bool, optional): Whether to display the disparity map or not. Defaults to True.

    Returns:
        numpy.ndarray: The computed disparity map.
    """

    # Compute disparity
    matcher = opencv_matcher.matcher
    disparity_img = matcher.compute(left_image_rectified, right_image_rectified)

    if disparity_img.dtype == np.int16:
        disparity_img = disparity_img.astype(np.float32)
        disparity_img /= 16

    if show_image:
        # Show Results
        plt.imshow(disparity_img)
        plt.title('Disparity Map')
        plt.colorbar()  # Add colorbar to show the disparity values
        plt.show()
        # plt.savefig('disparity_map.png', dpi=300, bbox_inches='tight')

    return disparity_img


def reproject_and_save_ply(disparity_img, opencv_matcher, index, output_folder):
    """Reprojects the 3D points from disparity image, transforms them into global coordinates, and saves as a PLY file.

    Args:
        disparity_img (numpy.ndarray): The disparity image.
        opencv_matcher (OpenCVStereoMatcher): The OpenCVStereoMatcher instance containing calibration parameters.
        index (int): The index of the stereo pair to process.
        output_folder (str): The folder path where the PLY file should be saved.

    Returns:
        str: The file path of the saved PLY file.
    """

    # Get Q-matrix
    Q = opencv_matcher.Q_array[index]

    # Reproject 3D
    threedeeimage = cv2.reprojectImageTo3D(disparity_img, Q, handleMissingValues=True, ddepth=cv2.CV_32F)
    threedeeimage = np.array(threedeeimage)

    # Postprocess
    xyz = threedeeimage.reshape((-1, 3))  # x,y,z now in three columns, in left rectified camera coordinates
    z = xyz[:, 2]
    goodz = z < 9999.0
    xyz_filtered = xyz[goodz, :]

    # Global Coordinates
    R_left_rectified_to_global, T_left_rectified_to_global = opencv_matcher.extrinsics_left_rectified_to_global_array[index]
    xyz_global = np.dot(xyz_filtered, R_left_rectified_to_global.T) + T_left_rectified_to_global.T

    # Save PLY
    filename = "temple_0"
    output_file_path = os.path.join(output_folder, filename + '.ply')
    save_ply(xyz_global, filename, output_folder)
    print("Saving: ", filename)

    return output_file_path

def read_and_rotate_images(image_paths):
    """
    Read images from the provided paths and convert them to OpenCV format.
    
    Args:
        image_paths (list): List of paths to image files.
        
    Returns:
        list: List of images in OpenCV format (numpy arrays).
    """
    images_cv = []
    for img_path in image_paths:
        # Read image using OpenCV
        img = cv2.imread(img_path)
        
        # Check if image was loaded successfully
        if img is None:
            print(f"Warning: Failed to load image {img_path}")
            continue
            
        # Add image to the list
        images_cv.append(img)
    
    return images_cv

def calibrate_and_rectify(options, left_K, right_K, left_R, right_R, left_T, right_T):
    """
    Perform stereo calibration and rectification.
    
    Args:
        options (dict): Dictionary containing rectification parameters.
        left_K (numpy.ndarray): Left camera intrinsic matrix.
        right_K (numpy.ndarray): Right camera intrinsic matrix.
        left_R (numpy.ndarray): Left camera rotation matrix.
        right_R (numpy.ndarray): Right camera rotation matrix.
        left_T (numpy.ndarray): Left camera translation vector.
        right_T (numpy.ndarray): Right camera translation vector.
        
    Returns:
        tuple: (Q, extrinsics_left_rectified_to_global, left_maps, right_maps)
            - Q: Disparity-to-depth mapping matrix
            - extrinsics_left_rectified_to_global: Transformation from rectified left camera to global coordinate system
            - left_maps: Rectification maps for left camera
            - right_maps: Rectification maps for right camera
    """
    # Convert rotation matrices to rotation vectors
    left_rvec, _ = cv2.Rodrigues(left_R)
    right_rvec, _ = cv2.Rodrigues(right_R)
    
    # Calculate relative pose between cameras
    R_left_to_right = np.dot(right_R, left_R.T)
    T_left_to_right = right_T - np.dot(R_left_to_right, left_T)
    
    # Extract parameters from options
    stereo_rectify_options = options['StereoRectify']
    image_size = stereo_rectify_options['imageSize']
    flags = stereo_rectify_options['flags']
    alpha = stereo_rectify_options['alpha']
    
    # Perform stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        left_K, None, right_K, None, 
        image_size, R_left_to_right, T_left_to_right,
        flags=flags, alpha=alpha
    )
    
    # Calculate rectification maps
    left_maps = cv2.initUndistortRectifyMap(
        left_K, None, R1, P1, image_size, cv2.CV_32FC1
    )
    
    right_maps = cv2.initUndistortRectifyMap(
        right_K, None, R2, P2, image_size, cv2.CV_32FC1
    )
    
    # Calculate transformation from rectified left camera to global coordinates
    R_left_rectified_to_global = np.dot(left_R, R1.T)
    T_left_rectified_to_global = left_T
    extrinsics_left_rectified_to_global = (R_left_rectified_to_global, T_left_rectified_to_global)
    
    return Q, extrinsics_left_rectified_to_global, left_maps, right_maps

def visualization_draw_geometry(geometry, headless=False):
    """
    Visualize a 3D geometry using Open3D.
    
    Args:
        geometry: Open3D geometry object to visualize.
        headless: Whether to run in headless mode (no GUI)
    """

    if headless or geometry is None:
        print("Skipping visualization (headless mode or empty geometry)")
        return

    try:
        # Create a visualization window
        vis = o3d.visualization.Visualizer()
        success = vis.create_window()
        if not success:
            print("Failed to create visualization window. Running in headless mode.")
            return
            
        # Add the geometry to the visualization
        vis.add_geometry(geometry)
        
        # Set rendering options
        opt = vis.get_render_option()
        if opt is not None:
            opt.background_color = np.asarray([0.5, 0.5, 0.5])  # Gray background
            opt.point_size = 1.0
        
        # Update the view
        vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()
        
        # Run the visualization
        vis.run()
        vis.destroy_window()
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Saving screenshot instead...")
        try:
            # Save an image of the point cloud as an alternative
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 
                                     '3D-Reconstruction-with-Uncalibrated-Stereo-main', 'Data', 'Output')
            os.makedirs(output_dir, exist_ok=True)
            image_path = os.path.join(output_dir, "temple_pointcloud.png")
            o3d.io.write_point_cloud(os.path.join(output_dir, "temple_pointcloud.ply"), geometry)
            print(f"Saved point cloud to {os.path.join(output_dir, 'temple_pointcloud.ply')}")
        except Exception as e2:
            print(f"Failed to save point cloud: {e2}")

def save_point_cloud_multiview(points, output_folder):
    """
    Save multiple views of a 3D point cloud visualization using matplotlib.
    
    Args:
        points (numpy.ndarray): Array of 3D points.
        output_folder (str): Path where to save the images.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Limit the number of points to avoid overloading
    max_points = 10000
    if points.shape[0] > max_points:
        indices = np.random.choice(points.shape[0], max_points, replace=False)
        points_subset = points[indices]
    else:
        points_subset = points
    
    # Calculate centroid and limits for consistent view
    centroid = np.mean(points_subset, axis=0)
    max_range = np.max(np.abs(points_subset - centroid)) * 1.2
    
    # Define elevation and azimuth angles for different views
    views = [
        (30, 45),   # Default view (30° elevation, 45° azimuth)
        (0, 0),     # Front view
        (0, 90),    # Side view
        (90, 0)     # Top view
    ]
    
    view_names = ['perspective', 'front', 'side', 'top']
    
    # Save each view
    for i, (elev, azim) in enumerate(views):
        fig = plt.figure(figsize=(10, 10), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the points with coloring based on height (z-coordinate)
        sc = ax.scatter(
            points_subset[:, 0],
            points_subset[:, 1],
            points_subset[:, 2],
            s=1,
            c=points_subset[:, 2],  # Color by Z-coordinate
            cmap='viridis',
            alpha=0.8
        )
        
        # Add a color bar
        plt.colorbar(sc, ax=ax, shrink=0.5)
        
        # Set consistent viewing limits
        ax.set_xlim([centroid[0] - max_range, centroid[0] + max_range])
        ax.set_ylim([centroid[1] - max_range, centroid[1] + max_range])
        ax.set_zlim([centroid[2] - max_range, centroid[2] + max_range])
        
        # Set the view angle
        ax.view_init(elev=elev, azim=azim)
        
        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Point Cloud - {view_names[i]} view')
        
        # Save the figure
        output_path = os.path.join(output_folder, f'pointcloud_{view_names[i]}.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved point cloud {view_names[i]} view to: {output_path}")
    
if __name__ == '__main__':

    # Get the current directory
    current_directory = os.getcwd()

    # Go back to the parent directory
    parent_directory = os.path.dirname(current_directory)

    # Set input directory
    rock_folder = os.path.join(parent_directory, '3D-Reconstruction-with-Uncalibrated-Stereo-main', 'Data', 'rock', 'undistorted')
    temple_folder = os.path.join(parent_directory, '3D-Reconstruction-with-Uncalibrated-Stereo-main', 'Data', 'temple', 'undistorted')
    output_folder = os.path.join(parent_directory, '3D-Reconstruction-with-Uncalibrated-Stereo-main', 'Data', 'Output')

    # # Call the function to rotate images in the temple_folder
    # rotate_images_anticlockwise(temple_folder)


    # Choose index of image
    index = 0

    # Choose folder
    input_folder = temple_folder

    # Folder path
    folder_path = Path(input_folder)

    # Get a list of all files in the rock_folder directory
    all_files = os.listdir(input_folder)

    # Filter only the image files (e.g., PNG or JPG)
    image_files = [file for file in all_files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Create a list of image paths by joining the filenames with the rock_folder path
    images = [os.path.join(input_folder, image_file) for image_file in image_files]

    # Call the function to read, rotate, and convert the images
    images_cv = read_and_rotate_images(images)

    # Get parameters of image
    h, w, d = images_cv[index].shape
    print(h, w, d)

    ### -------------------- TOPOLOGIES ---------------------------- ###

    topologies = collections.OrderedDict()
    topologies['360'] = tuple(zip((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
                                  (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0)))

    topologies['overlapping'] = tuple(zip((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                                          (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)))

    topologies['adjacent'] = tuple(zip((0, 2, 4, 6, 8, 10),
                                       (1, 3, 5, 7, 9, 11)))
    topologies['skipping_1'] = tuple(zip((0, 3, 6, 9),
                                         (1, 4, 7, 10)))
    topologies['skipping_2'] = tuple(zip((0, 4, 8),
                                         (1, 5, 9)))





    ### -------------------- CAMERA CALIOBRATION AND RECTIFICATION ---------------------------- ###

    # StereoRectifyOptions
    StereoRectifyOptions = {
        'imageSize': (w, h),
        # Specifies the desired size of the rectified stereo images. 'w' and 'h' are width and height, respectively.
        'flags': (0, cv2.CALIB_ZERO_DISPARITY)[0],
        # Flag for stereo rectification. 0: Disparity map is not modified. cv2.CALIB_ZERO_DISPARITY: Zero disparity at all pixels.
        'newImageSize': (w, h),  # Size of the output rectified images after the rectification process.
        'alpha': 0.5
        # Balance between preserving all pixels (alpha = 0.0) and completely rectifying the images (alpha = 1.0).
    }

    # RemapOptions
    RemapOptions = {
        'interpolation': cv2.INTER_LINEAR
        # Interpolation method used during the remapping process. Bilinear interpolation for smoother results.
    }

    # CameraArrayOptions
    CameraArrayOptions = {
        'channels': 3,
        # Number of color channels in the camera images. 1 for grayscale images, 3 for RGB color channels.
        'num_cameras': 12,  # Total number of cameras in the camera array.
        'topology': 'skipping_2'
        # Spatial arrangement or topology of the camera array ('adjacent', 'circular', 'linear', 'grid', etc.).
    }




    ### -------------------- DISPARITY ESTIMATION ---------------------------- ###

    # StereoMatcherOptions
    StereoMatcherOptions = {
        'MinDisparity': 0,
        'NumDisparities': 64,
        'BlockSize': 7,
        'Disp12MaxDiff': 0,
        'PreFilterCap': 0,
        'UniquenessRatio': 15,
        'SpeckleWindowSize': 50,
        'SpeckleRange': 1
    }

    # StereoSGBMOptions
    StereoSGBMOptions = {
        'PreFilterCap': 0,
        'UniquenessRatio': 0,
        'P1': 8,  # "Depth Change Cost
        'P2': 32,  # "Depth Step Cost
    }

    # FinalOptions
    FinalOptions = {
        'StereoRectify': StereoRectifyOptions,  # Options for stereo rectification.
        'StereoMatcher': StereoMatcherOptions,  # Options for the stereo matcher (either StereoBM or StereoSGBM).
        'StereoSGBM': StereoSGBMOptions,  # Options for StereoBM (set to StereoSGBMOptions if needed).
        'CameraArray': CameraArrayOptions,  # Options for the camera array configuration.
        'Remap': RemapOptions  # Options for remapping.
    }

    # Initialize Class
    opencv_matcher = OpenCVStereoMatcher(options=FinalOptions, calibration_path=folder_path)

    # Print Q-matrix to check
    print(opencv_matcher.Q_array[0])

    # Load images
    opencv_matcher.load_images(folder_path)

    # Check images
    plt.imshow(opencv_matcher.images[index])
    plt.show()

    # 1. Rectification
    print("\nRectification")
    left_image_rectified, right_image_rectified = rectify_and_show_results(opencv_matcher, image_index=index, show_image=False, output_folder=output_folder)

    # 2. Disparity
    print("\nDisparity")
    disparity_img = compute_and_show_disparity(opencv_matcher, left_image_rectified, right_image_rectified)
    plt.savefig(os.path.join(output_folder, 'disparity_map.png'), dpi=300, bbox_inches='tight')


    # num_d = (0, 512, 16)
    # b_s = (1, 31, 2)
    # window_s = (1, 13, 2)
    # uniqueness_r = (0, 10, 1)
    # speckle_w = (0, 250, 50)
    #
    # # Replace the display() function with plt.show()
    # disparity_left = interactive(compute_disparity, image=fixed(left_image_rectified),
    #                              img_pair=fixed(right_image_rectified), num_disparities=num_d, block_size=b_s,
    #                              window_size=window_s, matcher=["stereo_sgbm", "stereo_bm"],
    #                              uniqueness_ratio=uniqueness_r, speckleWindowSize=speckle_w)
    # plt.show()  # Instead of display(disparity_left)

    # 3. Project to 3D
    print("\nProject to 3D")
    output_file_path = reproject_and_save_ply(disparity_img, opencv_matcher, index, output_folder)

    # 4. Visualize 3D image
    # print("\nVisualize 3D image")
    # object_3d = PyntCloud.from_file(output_file_path)
    # object_3d.plot()

    # 5. Run in all images
    xyz = opencv_matcher.run()
    save_ply(xyz, "temple_skipping_2", output_folder)
    output_file_path = os.path.join(parent_directory, '3D-Reconstruction-with-Uncalibrated-Stereo-main', 'Data', 'Output', 'temple_skipping_2.ply')
    print("Saving: ", output_file_path)
    # # object_3d = PyntCloud.from_file(output_file_path)
    # # object_3d.plot()

    # # Load the PLY file
    # pcd = o3d.io.read_point_cloud(output_file_path)

    # # Visualize the point cloud
    # visualization_draw_geometry(pcd)

    # After saving the PLY file:
    print(f"Saving point cloud images to {output_folder}")
    try:
        # This might work in some environments:
        pcd = o3d.io.read_point_cloud(output_file_path)
        visualization_draw_geometry(pcd, headless=True)
    except Exception as e:
        print(f"Open3D visualization failed: {e}")

    # Always save the point cloud visualizations using matplotlib as a backup
    try:
        # Load the saved point cloud
        with open(output_file_path, 'r') as f:
            lines = f.readlines()
            
        # Skip header and read points
        header_end = 0
        for i, line in enumerate(lines):
            if "end_header" in line:
                header_end = i + 1
                break
                
        points_data = []
        for i in range(header_end, len(lines)):
            if lines[i].strip():
                coords = lines[i].strip().split()
                points_data.append([float(coords[0]), float(coords[1]), float(coords[2])])
                
        points = np.array(points_data)
        
        # Save multi-view visualizations
        save_point_cloud_multiview(points, output_folder)
    except Exception as e:
        print(f"Failed to save point cloud visualizations: {e}")


