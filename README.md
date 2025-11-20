# **3D Scene Reconstruction & Virtual Tour — CS436 Project**

A modular, production-quality implementation of the Structure-from-Motion (SfM) pipeline, inspired by systems like **Matterport** and **Photosynth**.
This repository contains weekly deliverables for building a full 3D reconstruction + interactive virtual tour system from a sequence of photographs.

The project follows the official CS436 specification.

# 📁 **Repository Structure**

```
project-root/
│
├── src/                # Modular .py codebase (feature detection, matching, SfM, utils)
├── notebooks/          # Weekly result reports (Week 1, Week 2, …)
├── data/               # Original + processed images
├── results/            # Match visualizations, point clouds, intermediate outputs
├── docs/               # Extra documentation (optional)
│
└── README.md
```

# ⚙️ **Environment Setup**

```bash
# Create environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

**Core Libraries Used**

* OpenCV (cv2)
* NumPy
* Open3D
* Matplotlib
* tqdm (optional)

# 📸 **Dataset Requirements**

As recommended in the project specification:

* Scene must be static & textured
* Camera must translate, not rotate in place
* 60–80% overlap between images
* Use 1080p or downscaled images for performance
* Sharp, well-lit images

All datasets are stored in `data/`.

# 🚀 **Week 1 — Setup & Feature Matching**

## **Objective**

Implement the initial feature-matching pipeline that forms the backbone of the SfM reconstruction.

## **1. Image Preprocessing**

The raw dataset contained **4K images (2160×3840)**.
We rescaled all images to **1080×1920** before processing.

**Reasons for Downscaling:**

* SIFT extracts thousands of keypoints, making 4K processing slow
* BFMatcher computation is quadratic in descriptor count
* Downsampling reduces runtime and memory by 3–4×
* Downscaled images still preserve key geometric structure

## **2. Feature Detection (SIFT)**

For each consecutive image pair:

* Detect SIFT keypoints
* Compute 128-dim descriptors
* Store per-image features

SIFT is chosen for:

* Scale invariance
* Rotation invariance
* Robustness in indoor natural scenes
* Strong performance for epipolar geometry tasks

## **3. Feature Matching (BFMatcher + Lowe’s Ratio Test)**

Using:

* **Brute-Force Matcher**
* **k-NN matching** (k=2)
* **Lowe’s Ratio Test (0.75)**

This eliminates ambiguous and high-error correspondences.

## **4. Visualization of Matches**

For each consecutive pair:

* Sort by descriptor distance
* Select the **top 100** filtered matches
* Display image with match lines
* Save to `results/week1/`

These matches are used in Week 2 for Essential Matrix estimation.

## **Week 1 Output Summary**

* Downsampled dataset
* SIFT keypoints + descriptors
* Raw + filtered matches
* Top-100 match visualizations
* Pipeline implemented in `src/feature_matching.py`

# 🚀 **Week 2 — Two-View Reconstruction**

## **Objective**

Build the two-view SfM foundation:

* Compute Essential Matrix
* Recover relative pose
* Triangulate sparse 3D structure

This corresponds to **Phase 1** of the specification.

## **1. Camera Intrinsics (K Matrix)**

Approximation used:

* Center of image = principal point
* fx = fy = image width
* Zero skew

K is constructed using the resized image (1080×1920).

## **2. Essential Matrix Estimation**

Using OpenCV:

```python
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
```

## **3. Pose Recovery & Cheirality Check**

`cv2.recoverPose()` outputs 4 possible poses.
We disambiguate by:

* Triangulating for each pose
* Counting points with positive depth in both frames
* Selecting the pose with the maximum valid 3D points

## **4. 3D Point Cloud Generation**

Once the correct pose is selected:

* Triangulate inlier correspondences
* Convert to Euclidean 3D points
* Remove invalid/outlier points
* Save output as `.ply`
* Visualize via Open3D

Stored in `results/week2/point_cloud.ply`.

## **Week 2 Output Summary**

* Filtered inlier correspondences
* Essential matrix
* Valid camera pose [R|t]
* Sparse 3D point cloud
* Pipeline implemented in `src/two_view_reconstruction.py`

# ▶️ **How to Run**

## **Feature Matching (Week 1)**

```bash
python src/run_week1.py --input_dir data/
```

## **Two-View Reconstruction (Week 2)**

```bash
python src/run_week2.py \
    --img1 data/img_000.jpg \
    --img2 data/img_001.jpg \
    --output results/week2/
```

# 🧪 **Results**

Outputs are stored in:

* `results/week1/`
* `results/week2/`


