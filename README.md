Sure — here’s the **complete README in pure Markdown code** format (you can copy-paste it directly into GitHub’s `README.md` editor or VS Code).
Everything is properly formatted with code fences, bold headers, and section spacing.

---

```markdown
# 3D Scene Reconstruction and Virtual Tour — CS436 Project

This repository implements a modular **Structure-from-Motion (SfM)** pipeline inspired by **Matterport** and **Microsoft Photosynth**.  
The goal is to reconstruct a 3D scene and camera trajectory from a sequence of 2D images, culminating in an interactive web-based virtual tour.

The work follows the official CS436 project specification and milestones.

## Repository Structure
```

project-root/
│
├── src/                # Modular Python codebase (SfM core modules)
├── notebooks/          # Weekly result notebooks (Week 1–3)
├── data/               # Source images / video frames
├── results/            # Visual outputs and point clouds
├── docs/               # Reports, PDFs, diagrams
│
└── README.md

````

## Environment Setup
```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
````

**Key Libraries**

* OpenCV
* NumPy
* SciPy
* Matplotlib
* Open3D (for point-cloud visualization)

## Dataset and Preprocessing

As per project guidelines:

* Capture static, textured scenes with **60–80 % overlap** between views.
* Ensure consistent lighting and physical translation (not rotation).
* Downscale 4K frames (3840×2160) to 1080×1920 for efficiency.
* Intrinsics are estimated from EXIF focal length and sensor size.

All input data are stored in `data/` and all results in `results/`.

---

## Week 1 — Feature Extraction and Matching

**Goal:** Implement a feature-matching foundation for SfM.

**Key Steps**

1. **Preprocessing:** Downsample 4K images → 1080×1920 for faster SIFT detection.
2. **Feature Detection:** Use **SIFT** to compute scale- and rotation-invariant descriptors for each image.
3. **Feature Matching:**

   * Brute-Force matcher with `k=2`
   * **Lowe’s Ratio Test (0.75)** to reject ambiguous matches
4. **Visualization:**

   * Sort by descriptor distance
   * Display top 100 matches per image pair

**Output:**
Filtered matches between 5 consecutive image pairs, visualized and saved in `results/week1/`.

---

## Week 2 — Two-View Reconstruction

**Goal:** Recover 3D structure and relative camera pose from two views.

**Steps**

1. Compute **Essential Matrix** using `cv2.findEssentialMat()` with RANSAC.
2. Decompose E → [R | t] using `cv2.recoverPose()`.
3. Perform a **cheirality check** to keep only physically valid poses (points in front of both cameras).
4. Triangulate 3D points using `cv2.triangulatePoints()`.
5. Save the reconstructed **sparse 3D point cloud** as `.ply`.

**Output:**
Sparse reconstruction for a chosen image pair, located at `results/week2/point_cloud.ply`.

---

## Week 3 — Incremental Multi-View SfM and Bundle Adjustment

**Goal:** Extend two-view reconstruction to a full sequence using **PnP** and **Bundle Adjustment**.

### 1. Feature Extraction

* Extracted up to **40 000 SIFT features per frame** for 30 frames.
* Constructed camera intrinsics K from EXIF data.

### 2. Map Initialization

* Selected two baseline frames (Frame 0 & 2).
* Matched SIFT features, estimated E and [R|t], triangulated initial 1468 3D points.
* Initialized the first two cameras in the map.

### 3. Incremental Reconstruction

* For each new frame:

  * Matched its features to existing 3D points.
  * Solved **PnP (Perspective-n-Point)** using `cv2.solvePnPRansac()` to estimate pose.
  * Added the new camera to the map.
  * Triangulated new points between the latest and previous views.
  * Logged the number of new 3D points (≈ 4000–9000 per frame).

### 4. Bundle Adjustment

* After every 5 frames, ran **global nonlinear optimization** using `scipy.optimize.least_squares()` to minimize reprojection error.
* Implemented sparse Jacobian for efficiency.
* Adjusted both camera poses and 3D points, achieving convergence after each iteration.

### 5. Output and Visualization

* Exported the final model (`final_model.ply`) and cleaned it by removing 2σ outliers.
* Rendered a 3D scatter visualization in Matplotlib.
* Exported camera poses and aligned coordinates (Y-up, Z-backward) to a JSON + PLY format for integration with **Three.js**.

**Final Statistics**

* ≈ 27 registered cameras
* ≈ 136 000 3D points
* Multiple bundle-adjustment iterations with cost reductions from ~9.2×10⁷ to 1.8×10⁸ (cost reported per stage)

**Output Files**

* `results/week3/final_model.ply` — Final sparse point cloud
* `results/week3/project_data.json` — Three.js camera metadata

---

## How to Run

**Feature Matching (Week 1)**

```bash
python src/run_week1.py --input_dir data/
```

**Two-View Reconstruction (Week 2)**

```bash
python src/run_week2.py --img1 data/img_000.jpg --img2 data/img_001.jpg
```

**Incremental SfM (Week 3)**

```bash
python src/run_week3.py --input_dir data/extracted_frames/
```

---

## Results

All visualizations and point clouds are saved under:

* `results/week1/`
* `results/week2/`
* `results/week3/`

---

## Project Practices

* Modular, object-oriented Python code (no heavy logic inside notebooks)
* Reproducible scripts with consistent inputs/outputs
* Incremental weekly development aligned with course milestones
* Point-cloud outputs compatible with Open3D and Three.js

---

## Team

* Muhammad Hussain Habib (27100016)
* Ayaan Ahmed (27100155)

---

## Future Milestones (Planned)

**Week 4 — Interactive Three.js Viewer**
Build a Photosynth-style web viewer with smooth camera transitions.

**Week 5 — Final Integration and Report**
Bundle full pipeline, documentation, and demonstration video.

```
