import numpy as np

class ReconstructionMap:
    def __init__(self, image_names):
        self.image_names = image_names 
        self.points_3d = []           
        self.point_colors = []        
        self.camera_poses = {}        
        self.keypoints = {}           
        self.descriptors = {}         
        self.point_correspondences = {}

    def add_camera(self, img_idx, R, t, kps, descs):
        self.camera_poses[img_idx] = (R, t)
        self.keypoints[img_idx] = kps
        self.descriptors[img_idx] = descs
        self.point_correspondences[img_idx] = {}

    def add_point(self, point_3d, color, track_list):
        pt_idx = len(self.points_3d)
        self.points_3d.append(point_3d)
        self.point_colors.append(color)
        for (img_idx, kp_idx) in track_list:
            self.point_correspondences[img_idx][kp_idx] = pt_idx