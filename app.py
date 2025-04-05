import streamlit as st
import os
import subprocess
import sys
import time
import base64
from pathlib import Path
import shutil
import tempfile
import logging
import uuid
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from plyfile import PlyData
IS_DEPLOYMENT = os.getenv('STREAMLIT_SHARING', '') or os.getenv('STREAMLIT_CLOUD', '')
# Set page config
st.set_page_config(
    page_title="3D Reconstruction from 2D Images",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import OpenCV
try:
    import cv2
    st.sidebar.success("✅ OpenCV installed successfully")
except ImportError:
    st.sidebar.error("❌ OpenCV not found, certain functionality will not work")


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path(__file__).parent.resolve()
SFM_DIR = BASE_DIR / "SFM"
DATA_DIR = SFM_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets" / "sample_outputs"

# Create a temporary directory for results - this is key for deployment
TEMP_DIR = Path(tempfile.mkdtemp())
RESULTS_DIR = TEMP_DIR / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Check environment
def check_environment():
    env_status = {}
    env_status["base_dir"] = os.path.exists(BASE_DIR)
    env_status["sfm_dir"] = os.path.exists(SFM_DIR)
    env_status["data_dir"] = os.path.exists(DATA_DIR)
    env_status["script_dir"] = os.path.exists(SFM_DIR / "script")
    env_status["temp_writable"] = os.access(TEMP_DIR, os.W_OK)
    return env_status

def get_environment_info():
    """Get information about the execution environment for debugging"""
    info = {
        "python_version": sys.version,
        "current_directory": os.getcwd(),
        "path_exists": {
            "BASE_DIR": os.path.exists(BASE_DIR),
            "SFM_DIR": os.path.exists(SFM_DIR),
            "DATA_DIR": os.path.exists(DATA_DIR),
            "TEMP_DIR": os.path.exists(TEMP_DIR),
            "script_dir": os.path.exists(SFM_DIR / "script")
        },
        "directory_contents": {
            "SFM/script": os.listdir(SFM_DIR / "script") if os.path.exists(SFM_DIR / "script") else "Not found",
            "SFM/data": os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else "Not found"
        }
    }
    return info

def load_ply_file(file_path):
    """Load PLY file and extract points and colors"""
    try:
        plydata = PlyData.read(file_path)
        vertices = plydata['vertex']
        
        # Extract coordinates
        x = vertices['x']
        y = vertices['y']
        z = vertices['z']
        
        # Extract colors if available, otherwise use default
        if 'red' in vertices:
            r = vertices['red']
            g = vertices['green']
            b = vertices['blue']
            has_colors = True
        else:
            # Generate a default color (white)
            r = np.ones_like(x) * 255
            g = np.ones_like(x) * 255
            b = np.ones_like(x) * 255
            has_colors = False
        
        return x, y, z, r, g, b, has_colors
    except Exception as e:
        logger.error(f"Error loading PLY file: {e}")
        raise

def create_3d_point_cloud_plot(x, y, z, r, g, b, subsample=1.0):
    """Create a 3D scatter plot from point cloud data"""
    # Subsample points if needed (for better performance)
    if subsample < 1.0:
        total_points = len(x)
        sample_size = int(total_points * subsample)
        indices = np.random.choice(total_points, size=sample_size, replace=False)
        x = x[indices]
        y = y[indices]
        z = z[indices]
        r = r[indices]
        g = g[indices]
        b = b[indices]
    
    # Create color strings for plotly
    colors = [f'rgb({int(r[i])},{int(g[i])},{int(b[i])})' for i in range(len(x))]
    
    # Create the scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=colors,
                opacity=0.8
            )
        )
    ])
    
    # Set layout options
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'  # preserve the data aspect ratio
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="3D Point Cloud"
    )
    
    return fig

def main():
    st.title("3D Reconstruction from 2D Images")
    
    # Environment check
    env_status = check_environment()
    if not all(env_status.values()):
        st.sidebar.error("⚠️ Environment Issue Detected")
        with st.sidebar.expander("Environment Details"):
            for key, val in env_status.items():
                status = "✅" if val else "❌"
                st.write(f"{status} {key}")
    
    st.sidebar.title("Navigation")
    approach = st.sidebar.selectbox(
        "Select Approach",
        ["Structure from Motion (SFM)", "Other Approaches"]
    )

    with st.sidebar.expander("Environment Info"):
        env_info = get_environment_info()
        st.json(env_info)
    
    if approach == "Structure from Motion (SFM)":
        render_sfm_page()
    else:
        render_other_approaches()

def render_sfm_page():
    st.header("Structure from Motion")
    
    # Add tabs for built-in datasets and custom uploads
    tabs = st.tabs(["Built-in Datasets", "Upload Your Own Images"])
    
    with tabs[0]:
        render_builtin_datasets()
    
    with tabs[1]:
        render_custom_upload()

# Extract the existing dataset rendering code
def render_builtin_datasets():
    # Description
    st.markdown("""
    Structure from Motion (SFM) is a photogrammetric technique that estimates 
    three-dimensional structures from two-dimensional image sequences. This implementation 
    uses feature matching and triangulation to reconstruct 3D point clouds from multiple views.
    """)
    
    # Display project workflow
    st.subheader("How it works")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**1. Feature Extraction & Matching**")
        st.markdown("Detects keypoints and matches them across images")
    with cols[1]:
        st.markdown("**2. Camera Pose Estimation**")
        st.markdown("Estimates camera positions using epipolar geometry")
    with cols[2]:
        st.markdown("**3. 3D Reconstruction**")
        st.markdown("Triangulates 3D points from matched features")
    
    # Dataset selection
    st.subheader("Select Dataset")
    dataset = st.selectbox(
        "Choose a sample dataset",
        ["fountain-P11", "entry-P10", "Herz-Jesus-P8", "castle-P19"],
        help="These are pre-calibrated datasets with known camera parameters"
    )
    
    # Show sample images from dataset
    st.subheader("Sample Images")
    st.markdown(f"Images from the {dataset} dataset:")
    
    # Display 3 sample images from the dataset
    image_dir = DATA_DIR / dataset / "images"
    if image_dir.exists():
        image_files = sorted(list(image_dir.glob("*.jpg")))[:3]
        cols = st.columns(3)
        for i, img_path in enumerate(image_files):
            if i < len(cols):
                cols[i].image(str(img_path), caption=f"Image {i+1}")
    else:
        st.warning(f"Dataset images for {dataset} not found. Please make sure the dataset is available in the data directory.")
    
    # Display sample output
    st.subheader("Expected Output")
    sample_output_path = ASSETS_DIR / dataset / "cloud_final.png"
    if sample_output_path.exists():
        st.image(str(sample_output_path), caption=f"Sample 3D reconstruction of {dataset}")
    else:
        st.info(f"Sample output image for {dataset} not found. You can add one at {sample_output_path}")
    
    # Configuration options
    st.subheader("Configuration")
    with st.expander("Advanced Settings"):
        feature_type = st.selectbox("Feature Detector", ["ORB", "SIFT"], index=0)
        matcher_type = st.selectbox("Feature Matcher", ["BFMatcher"], index=0)
        
    # Process button
    if st.button("Run Reconstruction", type="primary", key="run_builtin"):
        run_sfm_pipeline(dataset, feature_type, matcher_type)

# Add new function for custom image uploads
def render_custom_upload():
    st.markdown("""
    ## Upload Your Own Images
    
    Upload multiple images to create a 3D reconstruction. You'll need to provide the 
    intrinsic camera matrix parameters.
    
    **Note:** Since extrinsic parameters are unknown for your images, reprojection error 
    cannot be calculated.
    """)
    
    # Create a unique session ID for this upload to avoid conflicts
    if 'upload_session_id' not in st.session_state:
        st.session_state.upload_session_id = str(uuid.uuid4())
    
    session_id = st.session_state.upload_session_id
    
    # File uploaders
    st.subheader("Upload Images")
    uploaded_files = st.file_uploader("Choose multiple image files", accept_multiple_files=True, 
                                     type=["jpg", "jpeg", "png"])
    
    # Check if we have at least 2 images
    if uploaded_files and len(uploaded_files) < 2:
        st.warning("Please upload at least 2 images for reconstruction.")
    
    # Display uploaded images
    if uploaded_files:
        st.subheader("Uploaded Images")
        cols = st.columns(min(3, len(uploaded_files)))
        for i, img_file in enumerate(uploaded_files[:3]):
            cols[i % 3].image(img_file, caption=f"Image {i+1}")
    
    # Camera intrinsic matrix input
    st.subheader("Camera Intrinsics")
    st.info("Enter your camera's intrinsic matrix parameters. If unknown, you can use approximate values.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Focal Length")
        fx = st.number_input("fx (focal length in x direction)", 
                            value=1000.0, step=10.0)
        fy = st.number_input("fy (focal length in y direction)", 
                            value=1000.0, step=10.0)
    
    with col2:
        st.markdown("#### Principal Point")
        cx = st.number_input("cx (principal point x coordinate)", 
                            value=960.0, step=10.0)
        cy = st.number_input("cy (principal point y coordinate)", 
                            value=540.0, step=10.0)
    
    # Advanced options
    st.subheader("Configuration")
    with st.expander("Advanced Settings"):
        feature_type = st.selectbox("Feature Detector", ["ORB", "SIFT"], index=0, key="custom_feature")
        matcher_type = st.selectbox("Feature Matcher", ["BFMatcher"], index=0, key="custom_matcher")
    
    # Process button
    if st.button("Run Reconstruction", type="primary", key="run_custom") and uploaded_files and len(uploaded_files) >= 2:
        # Create custom dataset from uploads
        custom_dataset_name = f"custom_{session_id}"
        custom_dataset = create_custom_dataset(uploaded_files, custom_dataset_name, fx, fy, cx, cy)
        
        # Run the pipeline with the custom dataset
        run_sfm_pipeline(custom_dataset, feature_type, matcher_type, custom_intrinsics=True)

# Add function to create a custom dataset from uploads
def create_custom_dataset(uploaded_files, dataset_name, fx, fy, cx, cy):
    # Create dataset directory structure in TEMP_DIR instead of DATA_DIR
    dataset_dir = TEMP_DIR / "data" / dataset_name
    images_dir = dataset_dir / "images"
    calibration_dir = dataset_dir / "calibration"
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(calibration_dir, exist_ok=True)
    
    # Save uploaded images to temporary location
    for i, file in enumerate(uploaded_files):
        img = Image.open(file)
        img_path = images_dir / f"image_{i:03d}.jpg"
        img.save(img_path)
    
    # Save intrinsic matrix
    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    np.save(calibration_dir / "intrinsics.npy", intrinsic_matrix)
    
    return dataset_name

# Update the run_sfm_pipeline function to handle custom datasets
def run_sfm_pipeline(dataset, feature_type, matcher_type, custom_intrinsics=False):
    try:
        # Check if dataset is a custom one (uses temporary directory)
        is_custom = dataset.startswith("custom_")
        
        # Create dataset-specific directories
        if is_custom:
            dataset_dir = TEMP_DIR / "data" / dataset
        else:
            dataset_dir = DATA_DIR / dataset
            
        dataset_results_dir = RESULTS_DIR / dataset
        
        if IS_DEPLOYMENT or is_custom:
            # In deployment or custom uploads: use temp directories
            dataset_results_dir = RESULTS_DIR / dataset
            features_dir = dataset_results_dir / "features" / feature_type
            matches_dir = dataset_results_dir / "matches" / matcher_type
        else:
            # Local: use standard directories in the repository
            features_dir = dataset_dir / "features" / feature_type
            matches_dir = dataset_dir / "matches" / matcher_type
        
        point_clouds_dir = dataset_results_dir / "point-clouds"
        errors_dir = dataset_results_dir / "errors"
        
        os.makedirs(features_dir, exist_ok=True)
        os.makedirs(matches_dir, exist_ok=True)
        os.makedirs(point_clouds_dir, exist_ok=True)
        os.makedirs(errors_dir, exist_ok=True)
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Check if dataset exists
        image_dir = dataset_dir / "images"
        if not image_dir.exists():
            st.error(f"Dataset '{dataset}' not found. The app requires datasets to be available at {image_dir}")
            return
        
        # Get current environment
        env = os.environ.copy()
        
        # Add the site-packages directory to PYTHONPATH for the subprocess
        env['PYTHONPATH'] = os.path.dirname(os.__file__) + '/site-packages:' + env.get('PYTHONPATH', '')

        # Step 1: Run feature matching
        status_text.text("Step 1/2: Extracting and matching features...")
        featmatch_cmd = [
            sys.executable, 
            str(SFM_DIR / "script" / "featmatch.py"),
            f"--data_dir={str(image_dir)}",
            f"--out_dir={str(dataset_dir)}",
            f"--features={feature_type}",
            f"--matcher={matcher_type}",
            f"--feat_out_dir={str(features_dir)}",
            f"--matches_out_dir={str(matches_dir)}"
        ]
        
        logger.info(f"Running feature matching command: {' '.join(featmatch_cmd)}")
        result = subprocess.run(featmatch_cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            st.error(f"Feature matching failed: {result.stderr}")
            logger.error(f"Feature matching stderr: {result.stderr}")
            logger.error(f"Feature matching stdout: {result.stdout}")
            return
        
        progress_bar.progress(50)
        
        # Step 2: Run SFM
        status_text.text("Step 2/2: Performing 3D reconstruction...")
        
        # Determine the data_dir to use for SFM based on whether it's a custom dataset
        if is_custom:
            sfm_data_dir = TEMP_DIR / "data"
        else:
            sfm_data_dir = DATA_DIR
        
        sfm_cmd = [
            sys.executable, 
            str(SFM_DIR / "script" / "sfm.py"),
            f"--data_dir={str(sfm_data_dir)}",
            f"--dataset={dataset}",
            f"--features={feature_type}",
            f"--matcher={matcher_type}",
            f"--feat_in_dir={str(features_dir)}",
            f"--matches_in_dir={str(matches_dir)}", 
            f"--out_cloud_dir={str(point_clouds_dir)}",
            f"--out_err_dir={str(errors_dir)}"
        ]
        
        # Add custom parameters for user-uploaded datasets
        if custom_intrinsics:
            sfm_cmd.append(f"--skip_reprojection=True")
            sfm_cmd.append(f"--custom_intrinsics=True")
        else:
            sfm_cmd.append(f"--plot_error=True")
        
        logger.info(f"Running SFM command: {' '.join(sfm_cmd)}")
        result = subprocess.run(sfm_cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            st.error(f"SFM reconstruction failed: {result.stderr}")
            logger.error(f"SFM stderr: {result.stderr}")
            logger.error(f"SFM stdout: {result.stdout}")
            return
        
        progress_bar.progress(100)
        status_text.text("Reconstruction completed!")
        
        # Display results
        display_results(dataset)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.exception("Exception during SFM pipeline")

def display_results(dataset):
    # Display the generated point cloud
    st.subheader("Reconstruction Results")
    
    point_cloud_dir = RESULTS_DIR / dataset / "point-clouds"
    if not point_cloud_dir.exists():
        st.error(f"Results directory not found: {point_cloud_dir}")
        return
    
    # Find the final point cloud file
    cloud_files = sorted(list(point_cloud_dir.glob("cloud_*_view.ply")))
    if not cloud_files:
        st.error("No point cloud files were generated")
        return
    
    final_cloud = cloud_files[-1]
    st.success(f"Successfully generated point cloud with {len(cloud_files)} views")
    
    # NEW CODE: Add 3D visualization of the point cloud
    st.subheader("3D Visualization")
    try:
        # Add controls for visualization
        with st.expander("Visualization Settings", expanded=True):
            subsample_ratio = st.slider(
                "Point Density", 
                min_value=0.01, 
                max_value=1.0, 
                value=0.5 if os.path.getsize(final_cloud) > 10e6 else 1.0,  # Subsample large clouds
                step=0.01,
                help="Reduce this value to improve performance for large point clouds"
            )
            
            point_size = st.slider("Point Size", 1, 5, 2)
        
        # Load and display the point cloud
        with st.spinner("Loading 3D point cloud..."):
            x, y, z, r, g, b, has_colors = load_ply_file(final_cloud)
            
            if not has_colors:
                st.info("This point cloud doesn't contain color information. Displaying with default colors.")
            
            fig = create_3d_point_cloud_plot(x, y, z, r, g, b, subsample=subsample_ratio)
            
            # Update point size based on user selection
            fig.update_traces(marker=dict(size=point_size))
            
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 Tip: You can rotate, zoom, and pan the 3D view using your mouse")
    except Exception as e:
        st.error(f"Error creating 3D visualization: {str(e)}")
        st.info("You can still download the PLY file and view it in an external viewer.")
        
    # Display reprojection errors only if they exist
    error_dir = RESULTS_DIR / dataset / "errors"
    if error_dir.exists():
        error_images = sorted(list(error_dir.glob("*.png")))
        if error_images:
            st.subheader("Reprojection Errors")
            cols = st.columns(min(3, len(error_images)))
            for i, img_path in enumerate(error_images[:3]):
                cols[i % 3].image(str(img_path), caption=f"Error map {i+1}")
    
    # Provide download buttons for results
    st.subheader("Download Results")
    
    # Download final point cloud
    try:
        with open(final_cloud, "rb") as file:
            btn = st.download_button(
                label=f"Download Final Point Cloud ({final_cloud.name})",
                data=file,
                file_name=final_cloud.name,
                mime="application/octet-stream"
            )
    except Exception as e:
        st.error(f"Error preparing download for point cloud: {str(e)}")
    
    # Option to download all point clouds as zip
    if len(cloud_files) > 1:
        try:
            # Create a temp file for the zip
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
                zip_path = Path(tmp.name)
            
            # Create zip file
            shutil.make_archive(
                str(zip_path.with_suffix('')),
                'zip', 
                point_cloud_dir
            )
            
            # Offer download
            with open(zip_path, "rb") as file:
                btn = st.download_button(
                    label=f"Download All Point Clouds (ZIP)",
                    data=file,
                    file_name=f"{dataset}_point_clouds.zip",
                    mime="application/zip"
                )
                
            # Clean up temp file (will run after the session ends)
            try:
                os.unlink(zip_path)
            except:
                pass
        except Exception as e:
            st.error(f"Error creating zip file: {str(e)}")

def render_other_approaches():
    st.header("Other 3D Reconstruction Approaches")
    st.info("This section will contain other approaches to 3D reconstruction.")
    # Placeholder for other approaches

# Cleanup function for temp files
def cleanup():
    try:
        # Remove the entire temporary directory including all uploaded data
        if os.path.exists(TEMP_DIR):
            logger.info(f"Cleaning up temporary directory: {TEMP_DIR}")
            shutil.rmtree(TEMP_DIR)
            
        # Clean up any custom datasets that might have been created in DATA_DIR
        for item in os.listdir(DATA_DIR):
            if item.startswith("custom_"):
                custom_dir = DATA_DIR / item
                if os.path.exists(custom_dir):
                    logger.info(f"Cleaning up custom dataset: {custom_dir}")
                    shutil.rmtree(custom_dir)
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.exception("Unhandled exception in main app")
    finally:
        # Register cleanup for when the app shuts down
        # Note: In Streamlit Cloud, this may not always run as expected
        import atexit
        atexit.register(cleanup)