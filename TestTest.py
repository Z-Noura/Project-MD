import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mcubes

# Load data
binary_image1 = np.load('CerclesI1.npy')
binary_image2 = np.load('CerclesI2.npy')

center_image1 = np.load('CerclesC1.npy')
center_image2 = np.load('CerclesC2.npy')

CerclesSegmented1 = np.load('CerclesSegmented1.npy')
CerclesSegmented2 = np.load('CerclesSegmented2.npy')

# Plot images for verification
def plot_images(images):
    for image in images:
        plt.imshow(image, cmap='gray')
        plt.show()

# plot_images([binary_image1, binary_image2])

# Object points in 3D space
objp = np.array([
    [0, 0, 0], [0.7, 0.7, 0], [0, 1, 0], [1.31, 1.31, 0],
    [0, 2, 0], [2, 2, 0], [0, 3, 0], [0, 4, 0]
], dtype=np.float32)

# Image points in 2D space for image 1
imgpoints1 = np.array([
    center_image1[0], center_image1[5], center_image1[3], center_image1[6],
    center_image1[1], center_image1[8], center_image1[4], center_image1[2]
], dtype=np.float32).reshape(-1, 1, 2)

# Image points in 2D space for image 2
imgpoints2 = np.array([
    center_image2[0], center_image2[5], center_image2[3], center_image2[7],
    center_image2[1], center_image2[8], center_image2[4], center_image2[2]
], dtype=np.float32).reshape(-1, 1, 2)

# Prepare points for calibration
objpoints = [objp, objp]  # Object points
imgpoints = [imgpoints1, imgpoints2]  # Image points

# Load an image to get its size
image_size = binary_image1.shape

# Initial intrinsic matrix (to be adjusted if necessary)
initial_intrinsic_matrix = np.array([[7.5, 0, image_size[0] / 2],
                                     [0, 7.5, image_size[1] / 2],
                                     [0, 0, 1]])

# Camera calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, initial_intrinsic_matrix, None, None, flags=cv.CALIB_USE_INTRINSIC_GUESS)

# Build projection matrices
M1 = np.dot(mtx, np.hstack((cv.Rodrigues(rvecs[0])[0], tvecs[0])))
M2 = np.dot(mtx, np.hstack((cv.Rodrigues(rvecs[1])[0], tvecs[1])))

# Initialize voxel grid
voxel_size = [1, 1, 1]
xlim, ylim, zlim = [0, image_size[0]], [0, image_size[1]], [0, image_size[1]]

def initialize_voxels(xlim, ylim, zlim, voxel_size):
    voxels_number = [int(np.abs(xlim[1] - xlim[0]) / voxel_size[0]) + 1,
                     int(np.abs(ylim[1] - ylim[0]) / voxel_size[1]) + 1,
                     int(np.abs(zlim[1] - zlim[0]) / voxel_size[2]) + 1]
    total_number = np.prod(voxels_number)
    voxel_grid = np.ones((total_number, 4))
    # Create voxel coordinates
    l = 0
    for z in np.linspace(zlim[0], zlim[1], voxels_number[2]):
        for y in np.linspace(ylim[0], ylim[1], voxels_number[1]):
            for x in np.linspace(xlim[0], xlim[1], voxels_number[0]):
                try:
                    voxel_grid[l] = [x, y, z, 1]
                    l += 1
                except IndexError:
                    print("Voxelgrid too big")
                    print("Voxelgrid shape : ",voxel_grid.shape)
                    print("voxel : ", (x,y,z))
        print("Init : ",int((z-zlim[0])*100/(zlim[1]-zlim[0]))," %")
    return voxel_grid, np.array(voxels_number)

voxel_grid, voxels_number = initialize_voxels(xlim, ylim, zlim, voxel_size)
object_points3D = np.copy(voxel_grid).T
voxel_grid[:, 3] = 0  # Reset votes

# Accumulate votes for each voxel
def accumulate_votes(M, silhouettes, voxels):
    for i, M_ in enumerate(M):
        # Projection onto image plane
        points2D = np.dot(M_, object_points3D)
        points2D /= points2D[2, :]
        points2D = np.floor(points2D[:2, :]).astype(np.int32)
        
        # Ensure projected points are within image boundaries
        height, width = silhouettes.shape[:2]
        points2D[0, :] = np.clip(points2D[0, :], 0, width - 1)
        points2D[1, :] = np.clip(points2D[1, :], 0, height - 1)

        # Accumulate votes for each voxel
        valid_points = silhouettes[points2D[1], points2D[0], i]
        voxels[:, 3] += valid_points

# Prepare silhouettes
silhouettes = np.stack((CerclesSegmented1, CerclesSegmented2), axis=2)

# Matrices of projection
M = [M1, M2]
accumulate_votes(M, silhouettes, voxel_grid)

# Convert voxel list to 3D voxel grid
def convert_voxel_list_to_3D(voxels_number, voxel):
    voxel3D = np.zeros(voxels_number)
    l = 0
    for z in range(voxels_number[2]):
        for y in range(voxels_number[1]):
            for x in range(voxels_number[0]):
                voxel3D[x, y, z] = voxel[l, 3]
                l += 1
        print("Convert : ", int((z/voxels_number[2])*100), " %")
    return voxel3D

voxel3D = convert_voxel_list_to_3D(voxels_number, voxel_grid)

# Apply Marching Cubes and export 3D mesh
maxv = np.max(voxel_grid[:, 3])
iso_value = maxv - np.round(((maxv) / 100) * 5) - 0.5
vertices, triangles = mcubes.marching_cubes(voxel3D, iso_value)
mcubes.export_mesh(vertices, triangles, "Cercles3D.dae", "Cercles3D")

# Display the reconstructed 3D mesh
def plot_3d(vertices, triangles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], triangles, vertices[:, 2], cmap='Spectral')
    plt.show()

plot_3d(vertices, triangles)

print("3D reconstruction complete and displayed.")