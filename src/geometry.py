import cv2
import numpy as np
import random
from src.config import FOCAL_LENGTH_MM, SENSOR_WIDTH_MM, REPROJ_ERROR_THRESH, CONFIDENCE, RATIO_TEST_THRESH

def get_intrinsic_matrix(w, h):
    max_dim = max(w, h)
    fx = (max_dim * FOCAL_LENGTH_MM) / SENSOR_WIDTH_MM
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

def find_2d_3d_correspondences(recon_map, new_keypoints, new_descriptors):
    points_2d = []
    points_3d = []
    new_kp_indices = [] 
    point_3d_indices = []
    
    new_kp_to_3d = {} 
    
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Strategy: Check Last 5 Cameras + 3 Random Older Cameras
    all_cams = sorted(recon_map.camera_poses.keys())
    if len(all_cams) > 8:
        recent = all_cams[-5:] # Last 5
        older = list(set(all_cams) - set(recent))
        random_old = random.sample(older, min(3, len(older))) 
        target_cameras = recent + random_old
    else:
        target_cameras = all_cams 

    for existing_idx in target_cameras:
        existing_desc = recon_map.descriptors[existing_idx]
        if existing_desc is None or len(existing_desc) == 0: continue

        matches = matcher.knnMatch(new_descriptors, existing_desc, k=2)
        
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < RATIO_TEST_THRESH * n.distance:
                    new_idx = m.queryIdx
                    existing_ref_idx = m.trainIdx
                    
                    if existing_ref_idx in recon_map.point_correspondences[existing_idx]:
                        pt3d_idx = recon_map.point_correspondences[existing_idx][existing_ref_idx]
                        
                        if new_idx not in new_kp_to_3d:
                            new_kp_to_3d[new_idx] = pt3d_idx
                            points_2d.append(new_keypoints[new_idx].pt)
                            points_3d.append(recon_map.points_3d[pt3d_idx])
                            new_kp_indices.append(new_idx)
                            point_3d_indices.append(pt3d_idx)

    return (np.array(points_2d, dtype=np.float32), 
            np.array(points_3d, dtype=np.float32), 
            new_kp_indices, 
            point_3d_indices)

def solve_pnp(recon_map, kps, descs, K):
    points_2d, points_3d, new_kp_indices, pt3d_indices = find_2d_3d_correspondences(
        recon_map, kps, descs
    )
    
    if len(points_2d) < 6:
        return None, None, []

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d, 
        points_2d, 
        K, None, 
        iterationsCount=1000,
        reprojectionError=REPROJ_ERROR_THRESH,
        confidence=CONFIDENCE,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success or inliers is None: 
        return None, None, []
    
    R, _ = cv2.Rodrigues(rvec)
    
    pnp_matches = []
    for i in inliers.ravel():
        pnp_matches.append((new_kp_indices[i], pt3d_indices[i]))
    
    return R, tvec, pnp_matches

def triangulate_and_add(recon_map, img1_idx, img2_idx, K, img1_color_ref):
    from src.features import match_features 

    R1, t1 = recon_map.camera_poses[img1_idx]
    R2, t2 = recon_map.camera_poses[img2_idx]
    
    kp1 = recon_map.keypoints[img1_idx]
    kp2 = recon_map.keypoints[img2_idx]
    des1 = recon_map.descriptors[img1_idx]
    des2 = recon_map.descriptors[img2_idx]
    
    matches = match_features(des1, des2)
    
    new_matches = []
    for m in matches:
        if (m.queryIdx not in recon_map.point_correspondences[img1_idx] and 
            m.trainIdx not in recon_map.point_correspondences[img2_idx]):
            new_matches.append(m)

    if not new_matches:
        return 0

    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in new_matches]).T
    pts2 = np.float32([kp2[m.trainIdx].pt for m in new_matches]).T
    
    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    points_3d = (points_4d[:3] / points_4d[3]).T
    
    count = 0
    for i, pt in enumerate(points_3d):
        pt_c1 = R1 @ pt + t1.flatten()
        pt_c2 = R2 @ pt + t2.flatten()
        
        if pt_c1[2] > 0 and pt_c2[2] > 0:
            u, v = int(pts1[0, i]), int(pts1[1, i])
            if 0 <= v < img1_color_ref.shape[0] and 0 <= u < img1_color_ref.shape[1]:
                color = img1_color_ref[v, u][::-1] # BGR to RGB
            else:
                color = [128, 128, 128]
                
            track = [(img1_idx, new_matches[i].queryIdx), 
                     (img2_idx, new_matches[i].trainIdx)]
            
            recon_map.add_point(pt.tolist(), color, track)
            count += 1
            
    return count