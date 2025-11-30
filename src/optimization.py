import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

def bundle_adjustment(recon_map, K, verbose=True):
    cameras = sorted(recon_map.camera_poses.keys())
    
    cam_id_map = {cam_id: i for i, cam_id in enumerate(cameras)}
    n_opt_cams = len(cameras) - 1
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
        cam_params = params[:n_opt_cams * 6].reshape((n_opt_cams, 6))
        points = params[n_opt_cams * 6:].reshape((n_points, 3))
        residuals = []
        
        R0, t0 = np.eye(3), np.zeros((3,1))
        
        for (cam_idx, pt_idx, pt_2d) in observations:
            if cam_idx < 0: R, t = R0, t0
            else:
                rvec, t = cam_params[cam_idx, :3], cam_params[cam_idx, 3:].reshape(3,1)
                R, _ = cv2.Rodrigues(rvec)
            
            pt_3d = points[pt_idx]
            pt_cam = R @ pt_3d + t.flatten()
            z = max(pt_cam[2], 1e-5)
            proj = (K @ pt_cam) / z
            residuals.extend([proj[0] - pt_2d[0], proj[1] - pt_2d[1]])
        return np.array(residuals)

    m = len(observations) * 2
    n = len(x0)
    A = lil_matrix((m, n), dtype=int)
    for i, (c_idx, p_idx, _) in enumerate(observations):
        p_off = n_opt_cams * 6 + p_idx * 3
        A[2*i:2*i+2, p_off:p_off+3] = 1
        if c_idx >= 0:
            c_off = c_idx * 6
            A[2*i:2*i+2, c_off:c_off+6] = 1

    res = least_squares(fun, x0, jac_sparsity=A, verbose=0, x_scale='jac', method='trf')
    
    opt_cams = res.x[:n_opt_cams*6].reshape((n_opt_cams, 6))
    opt_pts = res.x[n_opt_cams*6:].reshape((n_points, 3))
    
    for i, cam_idx in enumerate(cameras[1:]):
        r, t = opt_cams[i, :3], opt_cams[i, 3:]
        R, _ = cv2.Rodrigues(r)
        recon_map.camera_poses[cam_idx] = (R, t.reshape(3,1))
    recon_map.points_3d = opt_pts.tolist()