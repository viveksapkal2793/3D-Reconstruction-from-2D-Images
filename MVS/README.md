# Multi-View Stereo (MVS) 3D Reconstruction

This repository implements a Multi-View Stereo approach for dense 3D reconstruction from calibrated image sequences. Unlike Structure-from-Motion which produces sparse point clouds, MVS generates dense, detailed 3D reconstructions by computing depth maps between image pairs.

## Introduction

Multi-View Stereo (MVS) is a computer vision technique that creates dense 3D models from multiple overlapping images with known camera parameters. Our implementation uses the Temple Ring dataset, which consists of 12 images of a temple figurine captured in a circular arrangement with calibrated cameras.

The pipeline consists of several key steps:
1. Image rectification to align epipolar lines
2. Stereo matching to compute disparity maps
3. Triangulation to generate 3D points
4. Point cloud alignment and merging using ICP

## Required Packages

```
numpy
opencv-contrib-python>=4.6.0
matplotlib
pillow
open3d>=0.15.0
scipy
pathlib
```

To install all requirements:

```bash
pip install numpy opencv-contrib-python matplotlib pillow open3d scipy pathlib
```

## Dataset

The Temple Ring dataset can be downloaded from:
- http://vision.middlebury.edu/mview/data/temple/

After downloading, extract the dataset to the `Data/temple/undistorted` directory relative to the code. The dataset includes:
- 12 calibrated images in PNG format
- Camera calibration files (`templeSR_par.txt` and `templeSR_ang.txt`)

## Usage Instructions

### 1. Running the Basic MVS Pipeline

The main script processes the temple images, computes disparity maps, and generates point clouds:

```bash
python Temple.py
```

Key parameters you can modify in the script:
- `index`: The starting image pair index
- `topologies`: Different camera pairing configurations ('360', 'overlapping', 'adjacent', 'skipping_1', 'skipping_2')
- `StereoMatcherOptions`: Parameters for the stereo matcher

### 2. Merging Point Clouds

After generating individual point clouds, use the ICP merger to align and combine them:

```bash
python temple_icp_merger.py --input_dir Data/Output --output_dir Data/Output/Merged
```

Parameters:
- `--input_dir`: Directory containing the input PLY files
- `--output_dir`: Directory to save the merged results
- `--voxel_size`: Point cloud downsampling size (default: 0.005)
- `--threshold`: ICP distance threshold (default: 0.02)
- `--create_mesh`: Enable mesh creation from point cloud
- `--mesh_depth`: Depth for Poisson reconstruction (default: 9)

## Output

The pipeline generates:
1. Rectified stereo images
2. Disparity maps
3. Individual point clouds (PLY files)
4. Merged and aligned point cloud
5. Optional mesh reconstruction

All output files are saved to the specified output directory (`Data/Output` by default).

## Camera Configurations

The code supports different camera pairing topologies:
- `360`: All cameras in sequence with cyclic connection (0-1, 1-2, ..., 11-0)
- `overlapping`: Sequential cameras without completing the circle
- `adjacent`: Pairing alternate cameras (0-2, 2-4, 4-6, etc.)
- `skipping_1`: Cameras with one-camera gaps
- `skipping_2`: Cameras with two-camera gaps (default)

Different configurations may provide better results depending on the scene.