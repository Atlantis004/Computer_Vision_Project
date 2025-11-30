import argparse
import os
import cv2
import numpy as np
from src import utils, features, geometry, optimization, reconstruction, config

def main():
    parser = argparse.ArgumentParser(description="Run Full SfM Pipeline")
    parser.add_argument("--data", type=str, default="data", help="Input image directory")
    parser.add_argument("--out", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.out): os.makedirs(args.out)

    images, names = utils.load_images_from_dir(args.data)

    feat_cache = features.extract_features(images)

    h, w = images[0].shape[:2]
    K = geometry.get_intrinsic_matrix(w, h)
    recon = reconstruction.ReconstructionMap(names)
    
    idx1, idx2 = 0, min(2, len(images)-1)
    kp1, des1 = feat_cache[idx1]
    kp2, des2 = feat_cache[idx2]
    
    matches = features.match_features(des1, des2)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    
    recon.add_camera(idx1, np.eye(3), np.zeros((3,1)), kp1, des1)
    recon.add_camera(idx2, R, t, kp2, des2)
    geometry.triangulate_new_points(recon, idx1, idx2, K, images[idx1])
    
    for next_idx in range(len(images)):
        if next_idx in recon.camera_poses: continue
        if next_idx not in feat_cache: continue
        
        print(f"Registering Image {next_idx}")
        kps, descs = feat_cache[next_idx]
        
        R, t, pnp_matches = geometry.solve_pnp(recon, next_idx, kps, descs, K)
        if R is None:
            print("Failed to localize.")
            continue
            
        recon.add_camera(next_idx, R, t, kps, descs)
        for (kp_idx, pt3d_idx) in pnp_matches:
            recon.point_correspondences[next_idx][kp_idx] = pt3d_idx
            
        prev_idx = max([k for k in recon.camera_poses if k != next_idx])
        new_pts = geometry.triangulate_new_points(recon, prev_idx, next_idx, K, images[prev_idx])
        print(f"Added {new_pts} points.")
        
        if len(recon.camera_poses) % config.BA_INTERVAL == 0:
            optimization.bundle_adjustment(recon, K)

    print("--- Finalizing ---")
    optimization.bundle_adjustment(recon, K)
    
    utils.save_ply(os.path.join(args.out, "final_cloud.ply"), recon.points_3d, recon.point_colors)
    
    utils.export_to_web_viewer(args.out, recon)
    print("Done.")

if __name__ == "__main__":
    main()