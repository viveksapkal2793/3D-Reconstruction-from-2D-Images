# Structure from Motion (SfM)

This repository implements a complete Structure from Motion (SfM) pipeline for reconstructing 3D structures from a sequence of 2D images. The process automatically estimates camera poses and creates a sparse 3D point cloud of the scene.

## Introduction

Structure from Motion is a photogrammetric technique that estimates three-dimensional structures from two-dimensional image sequences. The pipeline consists of two main stages:

1. **Feature Extraction and Matching**: Extract distinctive features from images and find correspondences between them
2. **Camera Pose Estimation and Triangulation**: Estimate camera positions and orientations, then triangulate 3D points

## Required Packages

```
opencv-contrib-python>=4.8.0
numpy
matplotlib
```

For the full OpenCV features (including SIFT and SURF), install:

```bash
pip install opencv-contrib-python
```

## Usage Instructions

### Step 1: Feature Extraction and Matching

First, run featmatch.py to extract features from images and compute matches between them:

```bash
python featmatch.py --data_dir <path_to_images_directory> --out_dir <output_directory> --features <SIFT|SURF|ORB> --matcher BFMatcher
```

#### Example:
```bash
python featmatch.py --data_dir ../data/castle-P19/images/ --out_dir ../data/castle-P19/ --features SIFT
```

#### Key Parameters:
- `--data_dir`: Directory containing the images
- `--out_dir`: Output directory for features and matches
- `--features`: Feature algorithm (SIFT, SURF, ORB) - SIFT generally produces the best results
- `--matcher`: Feature matching algorithm (BFMatcher, FlannBasedMatcher)
- `--cross_check`: Enable/disable cross-checking of feature matches (default: True)

### Step 2: Structure from Motion

After extracting features, run the main SfM pipeline to reconstruct the 3D point cloud:

```bash
python sfm.py --data_dir <root_data_directory> --dataset <dataset_name> --features <SIFT|SURF|ORB> --matcher BFMatcher --out_dir <results_directory>
```

#### Example:
```bash
python sfm.py --data_dir ../data/ --dataset castle-P19 --features SIFT --matcher BFMatcher --out_dir ../results/ --plot_error True
```

#### Key Parameters:
- `--data_dir`: Root directory containing dataset folders
- `--dataset`: Name of the dataset folder
- `--features`: Feature algorithm (should match what was used in featmatch.py)
- `--matcher`: Matching algorithm (should match what was used in featmatch.py)
- `--out_dir`: Directory to store results
- `--calibration_mat`: Type of intrinsic camera matrix (benchmark or lg_g3)
- `--plot_error`: Whether to create reprojection error visualizations

#### Advanced Parameters:
- `--fund_method`: Method for fundamental matrix estimation (default: FM_RANSAC)
- `--outlier_thres`: Threshold for outlier rejection (default: 0.9)
- `--pnp_method`: Method for PnP estimation (default: SOLVEPNP_DLS)
- `--reprojection_thres`: Reprojection threshold for PnP (default: 8.0)

## Dataset Structure

The code expects datasets to be organized as follows:
```
data/
├── dataset_name/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
```

## Output

The SfM pipeline generates:
- A sparse 3D point cloud (in PLY format)
- Reprojection error visualizations (if enabled)
- Multiple point cloud files showing the progression as more views are added

Results are saved to:
```
results/
├── dataset_name/
│   ├── point-clouds/
│   │   ├── cloud_2_view.ply
│   │   ├── cloud_3_view.ply
│   │   └── ...
│   ├── errors/
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
```

## Supported Datasets

The code has been tested with the following benchmark datasets:
- fountain-P11
- entry-P10
- Herz-Jesus-P8/P25 
- castle-P19

These datasets have known camera calibration parameters using the "benchmark" calibration matrix.