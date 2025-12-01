import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

def bundle_adjustment(recon_map, K, verbose=True):
    if verbose:
        print(f"  > Starting Bundle Adjustment (Cameras: {recon_map.get_camera_count()}, Points: {recon_map.get_point_count()})")
    
    cameras = sorted(recon_map.camera_poses.keys())
    cam_id_map = {cam_id: i for i, cam_id in enumerate(cameras)}

    n_cameras = len(cameras)
    n_points = len(recon_map.points_3d)
    
    x0 = []
    for cam_idx in cameras[1:]:
        R, t = recon_map.camera_poses[cam_idx]
        rvec, _ = cv2.Rodrigues(R)
        x0.extend(rvec.flatten())
        x0.extend(t.flatten())

    for pt in recon_map.points_3d:
        x0.extend(pt)
        
    x0 = np.array(x0)

    observations = []
    for cam_idx in cameras:
        for kp_idx, pt_idx in recon_map.point_correspondences[cam_idx].items():
            pt_2d = recon_map.keypoints[cam_idx][kp_idx].pt
            cam_opt_idx = cam_id_map[cam_idx] - 1
            observations.append((cam_opt_idx, pt_idx, pt_2d))

    def fun(params):
        n_opt_cams = n_cameras - 1
        cam_params = params[:n_opt_cams * 6].reshape((n_opt_cams, 6))
        points = params[n_opt_cams * 6:].reshape((n_points, 3))
        
        residuals = []
        
        R0, t0 = np.eye(3), np.zeros((3, 1))
        
        for (cam_opt_idx, pt_idx, pt_2d) in observations:
            if cam_opt_idx < 0:
                R, t = R0, t0
            else:
                rvec = cam_params[cam_opt_idx, :3]
                t = cam_params[cam_opt_idx, 3:].reshape(3, 1)
                R, _ = cv2.Rodrigues(rvec)
            
            pt_3d = points[pt_idx]
            pt_cam = R @ pt_3d + t.flatten()
            
            z = pt_cam[2]
            if z < 1e-5: z = 1e-5
            
            proj = (K @ pt_cam) / z
            residuals.extend([proj[0] - pt_2d[0], proj[1] - pt_2d[1]])
            
        return np.array(residuals)

    m = len(observations) * 2
    n = len(x0)
    A = lil_matrix((m, n), dtype=int)
    
    for i, (cam_opt_idx, pt_idx, _) in enumerate(observations):
        pt_offset = (n_cameras - 1) * 6 + pt_idx * 3
        A[2*i:2*i+2, pt_offset:pt_offset+3] = 1
        
        if cam_opt_idx >= 0:
            cam_offset = cam_opt_idx * 6
            A[2*i:2*i+2, cam_offset:cam_offset+6] = 1

    res = least_squares(fun, x0, jac_sparsity=A, loss='soft_l1', verbose=0, x_scale='jac', ftol=1e-3, method='trf')
    
    n_opt = n_cameras - 1
    optimized_cams = res.x[:n_opt*6].reshape((n_opt, 6))
    optimized_pts = res.x[n_opt*6:].reshape((n_points, 3))
    
    for i, cam_idx in enumerate(cameras[1:]):
        rvec = optimized_cams[i, :3]
        tvec = optimized_cams[i, 3:]
        R, _ = cv2.Rodrigues(rvec)
        recon_map.camera_poses[cam_idx] = (R, tvec.reshape(3, 1))
        
    recon_map.points_3d = optimized_pts.tolist()
    
    if verbose:
        print(f"  > BA Complete. Final Cost: {res.cost:.2f}")