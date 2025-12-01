# 3D Scene Reconstruction and Virtual Tour — CS436 Project

This repository implements a modular **Structure-from-Motion (SfM)** pipeline inspired by **Matterport** and **Microsoft Photosynth**.
The goal is to reconstruct a 3D scene and camera trajectory from a sequence of 2D images, culminating in an interactive web-based virtual tour.

The work follows the official CS436 project specification and milestones.

## Repository Structure

```text
project-root/
│
├── src/                # Modular Python codebase (SfM core modules)
├── notebooks/          # Weekly result notebooks (Week 1–3)
├── data/               # Source images / video frames
├── results/            # Visual outputs and point clouds
├── docs/               # Report
│
└── README.md
```

## Environment Setup

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Key Libraries**

* OpenCV
* NumPy
* SciPy
* Matplotlib
* Open3D (for point-cloud visualization)

## Dataset and Preprocessing

As per project guidelines:

* Capture static, textured scenes with **60–80% overlap** between views.
* Ensure consistent lighting and physical translation (not just rotation).
* Downscale 4K frames (3840×2160) to 1080×1920 for efficiency.
* Intrinsic parameters are estimated from EXIF focal length and sensor size.

All input data are stored in `data/` and all results in `results/`.

## Week 1 — Feature Extraction and Matching

**Goal:** Implement a feature-matching foundation for the SfM pipeline.

**Key Steps**

1. **Preprocessing**

   * Downsample 4K images to 1080×1920 to reduce computation time and memory usage.

2. **Feature Detection (SIFT)**

   * Use **SIFT** to compute scale- and rotation-invariant keypoints and 128-dimensional descriptors for each image.

3. **Feature Matching**

   * Use a Brute-Force matcher with `k=2` nearest neighbors.
   * Apply **Lowe’s Ratio Test** (threshold = 0.75) to reject ambiguous or noisy matches.

4. **Visualization**

   * Sort matches by descriptor distance.
   * Display and save the top 100 filtered matches per consecutive image pair.

**Output**

* Filtered SIFT matches between 5 consecutive image pairs.
* Top-100 match visualizations saved in `results/week1/`.

## Week 2 — Two-View Reconstruction

**Goal:** Recover 3D structure and relative camera pose from a pair of images.

**Steps**

1. **Essential Matrix Estimation**

   * Use `cv2.findEssentialMat()` with RANSAC on normalized point correspondences.

2. **Pose Recovery**

   * Decompose the Essential Matrix into rotation and translation using `cv2.recoverPose()`.

3. **Cheirality Check**

   * Select the physically valid pose where most triangulated points lie in front of both cameras (positive depth in both views).

4. **Triangulation**

   * Use `cv2.triangulatePoints()` to recover 3D points from the inlier correspondences.
   * Convert homogeneous coordinates to Euclidean coordinates.

5. **Export**

   * Save the resulting sparse 3D point cloud in `.ply` format for visualization in Open3D or other viewers.

**Output**

* Sparse two-view reconstruction stored at `results/week2/point_cloud.ply`.
* Valid [R | t] relative pose between the two cameras.

## Week 3 — Incremental Multi-View SfM and Bundle Adjustment

**Goal:** Extend the two-view reconstruction to a full image sequence using **PnP** and **Bundle Adjustment**.

### 1. Feature Extraction

* Extract up to **40,000 SIFT features per frame** for 30 frames.
* Compute camera intrinsics matrix **K** from EXIF metadata (focal length and sensor width).

### 2. Map Initialization

* Select two baseline frames (e.g., Frame 0 and Frame 2) with sufficient parallax.
* Match SIFT descriptors between the two frames.
* Estimate the Essential Matrix and recover relative pose [R | t].
* Triangulate initial 3D points and perform a cheirality check.
* Initialize the reconstruction map with:

  * First camera at the origin (R = I, t = 0).
  * Second camera at the recovered pose.
  * Approximately 1,468 initial 3D points, each with RGB color sampled from the image.

### 3. Incremental Reconstruction with PnP

For each remaining frame in the sequence:

* Match its descriptors to descriptors in the most recently added camera.
* For matches where the reference keypoint is already associated with a 3D point:

  * Build 2D–3D correspondences (image points ↔ existing 3D points).
* Use `cv2.solvePnPRansac()` to estimate the new camera pose:

  * Robustly fit the pose under outliers using RANSAC.
* Add the new camera pose (R, t) to the reconstruction map.
* Register PnP inlier matches as additional 2D–3D associations for that frame.
* Triangulate new 3D points between:

  * The newly registered camera, and
  * A previous camera (often the latest previous in the map).
* Only keep triangulated points that:

  * Are in front of both cameras (positive depth), and
  * Have valid image coordinates for color sampling.

This progressively grows both the 3D point cloud and the set of registered camera poses.

### 4. Bundle Adjustment

* After every 5 newly registered frames, run **Bundle Adjustment** using `scipy.optimize.least_squares()`.
* Optimize over:

  * Camera parameters (rotation in Rodrigues form + translation) for all but the first camera (which is fixed as reference).
  * All 3D point positions.
* Use a **sparse Jacobian** structure (`lil_matrix`) to speed up optimization:

  * Each residual depends only on one camera and one 3D point.
* The cost function minimizes total **reprojection error** between observed 2D points and the projections of the optimized 3D points.

After all frames are processed, a final global bundle adjustment is run on the entire map.

### 5. Output and Visualization

* Save the final dense point cloud to `final_model.ply`.
* Filter outliers by removing points lying more than two standard deviations from the mean in any dimension.
* Visualize the cleaned point cloud using Matplotlib’s 3D scatter plot:

  * Coordinates are rearranged for a more intuitive view (e.g., using X, Z, -Y).
* Export camera poses and point cloud for Three.js:

  * Convert from OpenCV’s coordinate conventions to Three.js conventions (Y-up, Z-backward).
  * Save:

    * A `.ply` file with aligned points.
    * A `project_data.json` containing:

      * Per-camera 4×4 transformation matrices (column-major, as used in Three.js).
      * Image filenames.
      * Reference to the point cloud file.

**Final Statistics (approximate)**

* Around 27 registered camera poses.
* Around 136,000 3D points after incremental reconstruction and refinement.
* Multiple Bundle Adjustment stages with decreasing reprojection cost.

**Main Output Files**

* `results/week3/final_model.ply` — final filtered point cloud.
* `results/week3/project_data.json` — camera trajectories and point-cloud metadata for Three.js.

## Results

All visualizations and 3D outputs are stored under:

* `results/`

## Project Practices

* Modular, object-oriented Python code (SfM components separated into clear modules).
* Notebooks are used only for visualization and reporting; core logic lives in `.py` files.
* Reproducible scripts with fixed I/O conventions.
* Outputs are compatible with both Open3D and web visualization tools like Three.js.

## Team

* Muhammad Hussain Habib (27100016)
* Ayaan Ahmed (27100155)

## Future Milestones (Planned)

**Week 4 — Interactive Three.js Viewer**
Implement a Photosynth-style web viewer with smooth interpolation between camera poses and point-cloud rendering.
