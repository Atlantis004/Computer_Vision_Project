import os
import cv2
import numpy as np
import json

def load_images_from_dir(directory):
    files = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    images = []
    names = []
    
    for f in files:
        path = os.path.join(directory, f)
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
            names.append(f)
            
    return images, names

def save_ply(filename, points, colors):
    points = np.array(points)
    colors = np.array(colors)
    
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header"
    ]
    
    with open(filename, 'w') as f:
        f.write('\n'.join(header) + '\n')
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
            
    print(f"Saved point cloud to: {filename}")

def export_to_web_viewer(output_dir, recon_map, ply_name="model.ply"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    M = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    cameras_json = []
    
    for img_idx, (R, t) in recon_map.camera_poses.items():
        R_wc = R.T
        C = -R_wc @ t
        
        T_cv = np.eye(4)
        T_cv[:3, :3] = R_wc
        T_cv[:3, 3] = C.flatten()

        T_three = M @ T_cv @ M
        
        cameras_json.append({
            "id": img_idx,
            "filename": f"frame_{img_idx:03d}.jpg", # Placeholder name logic
            "matrix": T_three.T.flatten().tolist()
        })

    raw_points = np.array(recon_map.points_3d)
    raw_colors = np.array(recon_map.point_colors)
    
    aligned_points = raw_points.copy()
    aligned_points[:, 1] *= -1
    aligned_points[:, 2] *= -1
    
    save_ply(os.path.join(output_dir, ply_name), aligned_points, raw_colors)

    data = {"cameras": cameras_json, "point_cloud_file": ply_name}
    with open(os.path.join(output_dir, "project_data.json"), "w") as f:
        json.dump(data, f)