import os
import cv2
import numpy as np
from src.config import BA_INTERVAL
from src.utils import load_images_from_dir, save_ply, export_to_web_viewer
from src.features import extract_features, match_features
from src.reconstruction import ReconstructionMap
from src.geometry import get_intrinsic_matrix, solve_pnp, triangulate_and_add
from src.optimization import bundle_adjustment

def main():
    FRAMES_DIR = "extracted_frames/"
    OUTPUT_DIR = "reconstruction_output"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # loading images
    images, image_names = load_images_from_dir(FRAMES_DIR)
    if not images:
        print("No images found.")
        return

    # 30 frames as per notebook
    LIMIT = 30
    images = images[:LIMIT]
    image_names = image_names[:LIMIT]
    
    h, w = images[0].shape[:2]
    K = get_intrinsic_matrix(w, h)
    print(f"Intrinsics:\n{K}")

    # Phase 1: Feature Extraction
    feature_cache = {}
    for idx, img in enumerate(images):
        kps, descs = extract_features(img)
        if kps is None or len(kps) < 100:
            print(f"Warning: Frame {idx} has insufficient features.")
            continue
        feature_cache[idx] = (kps, descs)
        print(f"Processed Frame {idx}: {len(kps)} features")

    # Phase 2: Initialization (Frame 0 and 2)
    recon = ReconstructionMap()
    idx1, idx2 = 0, 2

    kp1, des1 = feature_cache[idx1]
    kp2, des2 = feature_cache[idx2]
    
    matches = match_features(des1, des2)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Essential Matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    
    recon.add_camera(idx1, np.eye(3), np.zeros((3,1)), kp1, des1)
    recon.add_camera(idx2, R, t, kp2, des2)
    
    # Triangulate initial points
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R, t))
    
    pts1_good = pts1[pose_mask.ravel() > 0]
    pts2_good = pts2[pose_mask.ravel() > 0]
    points_4d = cv2.triangulatePoints(P1, P2, pts1_good.T, pts2_good.T)
    points_3d = (points_4d[:3] / points_4d[3]).T
    
    valid_mask_indices = np.where(pose_mask.ravel() > 0)[0]
    for i, pt3d in enumerate(points_3d):
        if pt3d[2] > 0:
            match_obj = matches[valid_mask_indices[i]]
            u, v = int(pts1_good[i,0]), int(pts1_good[i,1])
            color = images[idx1][v, u][::-1] # BGR to RGB
            track = [(idx1, match_obj.queryIdx), (idx2, match_obj.trainIdx)]
            recon.add_point(pt3d.tolist(), color, track)
            
    print(f"Initialization complete with {recon.get_point_count()} points.")

    # 4. Phase 3: Incremental Reconstruction
    registered = {idx1, idx2}
    all_indices = sorted(feature_cache.keys())
    
    for next_idx in all_indices:
        if next_idx in registered: continue
        
        print(f"Registering Frame {next_idx}")
        kps, descs = feature_cache[next_idx]
        
        R, t, pnp_matches = solve_pnp(recon, kps, descs, K)
        
        if R is None:
            print("Failed to localize. Skipping.")
            continue
            
        recon.add_camera(next_idx, R, t, kps, descs)
        registered.add(next_idx)
        
        # Link PnP inliers
        for (kp_idx, pt3d_idx) in pnp_matches:
            recon.point_correspondences[next_idx][kp_idx] = pt3d_idx
            
        # Triangulate new points against previous camera
        prev_idx = max([k for k in recon.camera_poses.keys() if k != next_idx])
        new_pts_count = triangulate_and_add(recon, prev_idx, next_idx, K, images[prev_idx])
        print(f"Pose: Added {new_pts_count} new points.")
        
        if len(registered) % BA_INTERVAL == 0:
            bundle_adjustment(recon, K)

    print("Final Global Bundle Adjustment...")
    bundle_adjustment(recon, K)
    
    save_ply(os.path.join(OUTPUT_DIR, "final_model.ply"), recon.points_3d, recon.point_colors)
    export_to_web_viewer(OUTPUT_DIR, recon)
    print("Reconstruction Finished.")

if __name__ == "__main__":
    main()