import cv2

def extract_features(images):
    sift = cv2.SIFT_create(nfeatures=50000)
    feature_cache = {}
    
    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, descs = sift.detectAndCompute(gray, None)
        if kps:
            feature_cache[idx] = (kps, descs)
            
    return feature_cache

def match_features(desc1, desc2, ratio=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(desc1, desc2, k=2)
    
    good_matches = []
    for m, n in raw_matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches