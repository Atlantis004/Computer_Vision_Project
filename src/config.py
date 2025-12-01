# hardware specs for the camera
FOCAL_LENGTH_MM = 24.0
SENSOR_WIDTH_MM = 36.0

# feature extraction
SIFT_NFEATURES = 70000

# reconstruction settings
RATIO_TEST_THRESH = 0.75

# used 20.0 for PnP RANSAC to handle 4K resolution better
REPROJ_ERROR_THRESH = 20.0 
CONFIDENCE = 0.99
BA_INTERVAL = 5