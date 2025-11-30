import cv2
import numpy as np
from src.config import FOCAL_LENGTH_MM, SENSOR_WIDTH_MM, REPROJ_ERROR_THRESH, CONFIDENCE
from src.features import match_features

def get_intrinsic_matrix(w, h):
    max_dim = max(w, h)
    fx = (max_dim * FOCAL_LENGTH_MM) / SENSOR_WIDTH_MM
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

def solve_pnp(recon_map, new_img_idx, kps, descs, K):
    ref_cam_idx = max(recon_map.camera_poses.keys())
    ref_descs = recon_map.descriptors[ref_cam_idx]
    
    matches = match_features(descs, ref_descs)
    
    object_points = []
    image_points = []
    valid_matches = []
    
    for m in matches:
        if m.trainIdx in recon_map.point_correspondences[ref_cam_idx]:
            pt3d_idx = recon_map.point_correspondences[ref_cam_idx][m.trainIdx]
            object_points.append(recon_map.points_3d[pt3d_idx])
            image_points.append(kps[m.queryIdx].pt)
            valid_matches.append((m.queryIdx, pt3d_idx))
            
    if len(object_points) < 6:
        return None, None, []

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.array(object_points), np.array(image_points), K, None,
        reprojectionError=REPROJ_ERROR_THRESH, confidence=CONFIDENCE
    )
    
    if not success: return None, None, []
    
    R, _ = cv2.Rodrigues(rvec)
    pnp_matches = [valid_matches[i[0]] for i in inliers] if inliers is not None else []
    
    return R, tvec, pnp_matches

def triangulate_new_points(recon_map, img1_idx, img2_idx, K, img1_color_ref):
    R1, t1 = recon_map.camera_poses[img1_idx]
    R2, t2 = recon_map.camera_poses[img2_idx]
    kp1, kp2 = recon_map.keypoints[img1_idx], recon_map.keypoints[img2_idx]
    
    matches = match_features(recon_map.descriptors[img1_idx], recon_map.descriptors[img2_idx])
    
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    
    pts1, pts2, match_objs = [], [], []
    for m in matches:
        if (m.queryIdx not in recon_map.point_correspondences[img1_idx] and 
            m.trainIdx not in recon_map.point_correspondences[img2_idx]):
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
            match_objs.append(m)
            
    if not pts1: return 0

    points_4d = cv2.triangulatePoints(P1, P2, np.float32(pts1).T, np.float32(pts2).T)
    points_3d = (points_4d[:3] / points_4d[3]).T
    
    count = 0
    for i, pt in enumerate(points_3d):
        # Cheirality check
        pt_c1 = R1 @ pt + t1.flatten()
        if pt_c1[2] > 0:
            u, v = int(pts1[i][0]), int(pts1[i][1])
            try: color = img1_color_ref[v, u][::-1]
            except: color = [128, 128, 128]
            
            track = [(img1_idx, match_objs[i].queryIdx), (img2_idx, match_objs[i].trainIdx)]
            recon_map.add_point(pt.tolist(), color, track)
            count += 1
    return count