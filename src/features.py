import cv2
from src.config import SIFT_NFEATURES, RATIO_TEST_THRESH

def extract_features(img):
    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps, descs = sift.detectAndCompute(gray, None)
    return kps, descs

def match_features(desc1, desc2, ratio_thresh=RATIO_TEST_THRESH):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []

    raw_matches = bf.knnMatch(desc1, desc2, k=2)
    
    good_matches = []
    for match_pair in raw_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
            
    return good_matches