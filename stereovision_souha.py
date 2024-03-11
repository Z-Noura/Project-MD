import numpy as np
import cv2 as cv

# Load the data
centers = np.load('binary_image1.npy')
centers_other_image = np.load('binary_image2.npy')

# Resize the larger image to match the size of the smaller one
if centers_other_image.shape[1] > centers.shape[1] or centers_other_image.shape[0] > centers.shape[0]:
    centers_other_image = cv.resize(centers_other_image, (centers.shape[1], centers.shape[0]))
else:
    centers_other_image = cv.resize(centers_other_image, (centers.shape[1], centers.shape[0]))

focal_length_x = 100
focal_length_y = 100
center_x = 0
center_y = 0
initial_mtx = np.array([
    [focal_length_x, 0, center_x],
    [0, focal_length_y, center_y],
    [0, 0, 1]
], dtype=np.float32)

# Object points in 3D space
objp = np.array([
    [centers[0][0], centers[0][1], 424.49305556],
    [centers[1][0], centers[1][1], 424.49305556],
    [centers[2][0], centers[2][1], 424.49305556],
    [centers[3][0], centers[3][1], 424.49305556],
    [centers[4][0], centers[4][1], 424.49305556],
    [centers[5][0], centers[5][1], 424.49305556 - 82.18181818]
], dtype=np.float32)

# Image points in 2D space for image 1
imgpoints1 = np.array([
    [centers[0][0], centers[0][1]],
    [centers[1][0], centers[1][1]],
    [centers[2][0], centers[2][1]],
    [centers[3][0], centers[3][1]],
    [centers[4][0], centers[4][1]],
    [centers[5][0], centers[5][1]]
], dtype=np.float32)

# Image points in 2D space for image 2
imgpoints2 = np.array([
    [centers_other_image[0][0], centers_other_image[0][1]],
    [centers_other_image[1][0], centers_other_image[1][1]],
    [centers_other_image[2][0], centers_other_image[2][1]],
    [centers_other_image[3][0], centers_other_image[3][1]],
    [centers_other_image[4][0], centers_other_image[4][1]],
    [centers_other_image[5][0], centers_other_image[5][1]]
], dtype=np.float32)

# Convert the lists into a format usable by the calibration function
objpoints = [objp] * 2  # List containing object points for both images

# Set the correct image sizes based on the resized images
image_size = (centers.shape[1], centers.shape[0])

# Perform camera calibration with the initial guess for the camera matrix
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, [imgpoints1, imgpoints2], image_size, initial_mtx, None, None)

# Now you can use the obtained camera matrix and distortion coefficients
# to get the initial rotation and translation vectors using solvePnP
_, rvecs_initial, tvecs_initial = cv.solvePnP(objp, imgpoints1, mtx, dist)
