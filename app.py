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
        feature_type = st.selectbox("Feature Detector", ["ORB", "SIFT", "SURF"], index=0)
        matcher_type = st.selectbox("Feature Matcher", ["BFMatcher", "FlannBasedMatcher"], index=0)
        
    # Process button
    if st.button("Run Reconstruction", type="primary"):
        run_sfm_pipeline(dataset, feature_type, matcher_type)

def run_sfm_pipeline(dataset, feature_type, matcher_type):
    try:
        # Create dataset-specific directories
        dataset_dir = DATA_DIR / dataset
        # Always define dataset_results_dir (this was missing)
        dataset_results_dir = RESULTS_DIR / dataset
        
        # features_dir = dataset_dir / "features"
        # matches_dir = dataset_dir / "matches"
        
        # Use different strategies for local vs deployment
        if IS_DEPLOYMENT:
            # In deployment: use temp directories
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
        sfm_cmd = [
            sys.executable, 
            str(SFM_DIR / "script" / "sfm.py"),
            f"--data_dir={str(DATA_DIR)}",
            f"--dataset={dataset}",
            f"--features={feature_type}",
            f"--matcher={matcher_type}",
            f"--plot_error=True",
            f"--feat_in_dir={str(features_dir)}",         # Add this line
            f"--matches_in_dir={str(matches_dir)}", 
            f"--out_cloud_dir={str(point_clouds_dir)}",  # Ensure your sfm.py accepts this parameter
            f"--out_err_dir={str(errors_dir)}"           # Ensure your sfm.py accepts this parameter
        ]
        
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
    
    # Display reprojection errors
    error_dir = RESULTS_DIR / dataset / "errors"
    if error_dir.exists():
        st.subheader("Reprojection Errors")
        error_images = sorted(list(error_dir.glob("*.png")))
        
        if error_images:
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
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
    except Exception as e:
        logger.error(f"Error cleaning up temp directory: {e}")

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