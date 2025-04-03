import streamlit as st
import os
import subprocess
import time
import base64
from pathlib import Path
import shutil

# Set page config
st.set_page_config(
    page_title="3D Reconstruction from 2D Images",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
BASE_DIR = Path(__file__).parent
SFM_DIR = BASE_DIR / "SFM"
DATA_DIR = SFM_DIR / "data"
RESULTS_DIR = SFM_DIR / "results"
ASSETS_DIR = BASE_DIR / "assets" / "sample_outputs"

def main():
    st.title("3D Reconstruction from 2D Images")
    
    st.sidebar.title("Navigation")
    approach = st.sidebar.selectbox(
        "Select Approach",
        ["Structure from Motion (SFM)", "Other Approaches"]
    )
    
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
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Run feature matching
        status_text.text("Step 1/2: Extracting and matching features...")
        featmatch_cmd = [
            "python", 
            str(SFM_DIR / "script" / "featmatch.py"),
            f"--data_dir={str(DATA_DIR / dataset / 'images')}",
            f"--out_dir={str(DATA_DIR / dataset)}",
            f"--features={feature_type}",
            f"--matcher={matcher_type}"
        ]
        
        result = subprocess.run(featmatch_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Feature matching failed: {result.stderr}")
            return
        
        progress_bar.progress(50)
        
        # Step 2: Run SFM
        status_text.text("Step 2/2: Performing 3D reconstruction...")
        sfm_cmd = [
            "python", 
            str(SFM_DIR / "script" / "sfm.py"),
            f"--data_dir={str(DATA_DIR)}",
            f"--dataset={dataset}",
            f"--features={feature_type}",
            f"--matcher={matcher_type}",
            f"--plot_error=True"
        ]
        
        result = subprocess.run(sfm_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"SFM reconstruction failed: {result.stderr}")
            return
        
        progress_bar.progress(100)
        status_text.text("Reconstruction completed!")
        
        # Display results
        display_results(dataset)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

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
    with open(final_cloud, "rb") as file:
        btn = st.download_button(
            label=f"Download Final Point Cloud ({final_cloud.name})",
            data=file,
            file_name=final_cloud.name,
            mime="application/octet-stream"
        )
    
    # Option to download all point clouds as zip
    if len(cloud_files) > 1:
        # Create a zip file of all point clouds
        zip_path = RESULTS_DIR / f"{dataset}_point_clouds.zip"
        shutil.make_archive(
            str(zip_path.with_suffix("")), 
            'zip', 
            point_cloud_dir
        )
        
        with open(zip_path, "rb") as file:
            btn = st.download_button(
                label=f"Download All Point Clouds (ZIP)",
                data=file,
                file_name=zip_path.name,
                mime="application/zip"
            )

def render_other_approaches():
    st.header("Other 3D Reconstruction Approaches")
    st.info("This section will contain other approaches to 3D reconstruction.")
    # Placeholder for other approaches

if __name__ == "__main__":
    main()