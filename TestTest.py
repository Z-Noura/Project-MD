import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mcubes

Image_Silhouette_1 = np.load('CerclesSegmented1.npy')
Image_Silhouette_2 = np.load('CerclesSegmented2.npy')

Centres_1 = np.load('CerclesC1.npy')
Centres_2 = np.load('CerclesC2.npy')

objp = np.array([
    [0, 0, 0], [0.7, 0.7, 0], [0, 1, 0], [1.31, 1.31, 0],
    [0, 2, 0], [2, 2, 0], [0, 3, 0], [0, 4, 0]
], dtype=np.float32)

imgpoints1 = np.array([
    Centres_1[0], Centres_1[5], Centres_1[3], Centres_1[6],
    Centres_1[1], Centres_1[8], Centres_1[4], Centres_1[2]
], dtype=np.float32).reshape(-1, 1, 2)

imgpoints2 = np.array([
    Centres_2[0], Centres_2[5], Centres_2[3], Centres_2[7],
    Centres_2[1], Centres_2[8], Centres_2[4], Centres_2[2]
], dtype=np.float32).reshape(-1, 1, 2)

objpoints = [objp, objp]  
imgpoints = [imgpoints1, imgpoints2]

image_size = Image_Silhouette_1.shape

initial_intrinsic_matrix = np.array([[7.5, 0, image_size[0] / 2],
                                     [0, 7.5, image_size[1] / 2],
                                     [0, 0, 1]])

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, initial_intrinsic_matrix, None, None, flags=cv.CALIB_USE_INTRINSIC_GUESS)
#ret, mtx, dist2, rvecs2, tvecs2 = cv.calibrateCamera(objpoints, imgpoints, image_size, initial_intrinsic_matrix, None, None, flags=cv.CALIB_USE_INTRINSIC_GUESS)

M1 = np.hstack((cv.Rodrigues(rvecs[0])[0], tvecs[0]))
M1 = mtx @ M1

M2 = np.hstack((cv.Rodrigues(rvecs[1])[0], tvecs[1]))
M2 = mtx @ M2

image_width, image_height = Image_Silhouette_1.shape[1], Image_Silhouette_1.shape[0]

voxel_size = [1, 1, 1]  # Taille de voxel de 1 mm dans chaque dimension

xlim = [0, image_width]
ylim = [0, image_height]
zlim = [0,10] 
def InitializeVoxels(xlim, ylim, zlim, voxel_size):
    # Calculer le nombre de voxels le long de chaque dimension
    voxels_number = [
        int(np.abs(xlim[1] - xlim[0]) / voxel_size[0]) + 1,
        int(np.abs(ylim[1] - ylim[0]) / voxel_size[1]) + 1,
        int(np.abs(zlim[1] - zlim[0]) / voxel_size[2]) + 1
    ]
    total_number = np.abs(np.prod(voxels_number))
   
    voxel_grid = np.ones((total_number, 4))
    
    # Créer les coordonnées des voxels
    l = 0
    for z in np.linspace(zlim[0], zlim[1], voxels_number[2]):
        for y in np.linspace(ylim[0], ylim[1], voxels_number[1]):
            for x in np.linspace(xlim[0], xlim[1], voxels_number[0]):
                voxel_grid[l] = [x, y, z, 1]
                l += 1
        print("Init : ",int((z-zlim[0])*100/(zlim[1]-zlim[0]))," %")

    return voxel_grid, voxels_number


def accumulate_votes(M, silhouettes, voxels):
    for i, M_ in enumerate(M):
        # Projection sur le plan d'image
        points2D = M_ @ object_points3D
        points2D /= points2D[2, :]  # Normaliser
        points2D = np.floor(points2D[:2, :]).astype(np.int32)  # Ne prendre que les coordonnées x et y

        # Assurez-vous que les points projetés sont dans les limites de l'image
        height, width = silhouettes.shape[:2]
        points2D[0, :] = np.clip(points2D[0, :], 0, width - 1)
        points2D[1, :] = np.clip(points2D[1, :], 0, height - 1)
        
        # Accumuler les votes pour chaque voxel
        valid_points = silhouettes[points2D[1], points2D[0], i]
        voxels[:, 3] += valid_points
    return voxels

def ConvertVoxelList2Voxel3D(voxels_number, voxel):
    voxel3D = np.zeros(voxels_number)
    l = 0
    for z in range(voxels_number[2]):
        for y in range(voxels_number[1]):
            for x in range(voxels_number[0]):
                voxel3D[x, y, z] = voxel[l, 3]
                l += 1
        print("Convert : ", int((z/voxels_number[2])*100), " %")
    return voxel3D

print(1)
voxel_grid, voxels_number = InitializeVoxels(xlim, ylim, zlim, voxel_size)
object_points3D = np.copy(voxel_grid).T
print(object_points3D)

voxel_grid[:, 3] = 0

silhouettes = np.stack((Image_Silhouette_1, Image_Silhouette_2), axis=2)
M = [M1, M2]
print(11)
voxels = accumulate_votes(M, silhouettes, voxel_grid)

voxel3D = ConvertVoxelList2Voxel3D(voxels_number, voxels)

maxv = np.max(voxel_grid[:, 3])

iso_value = maxv - np.round(((maxv) / 100) * 5) - 0.5
vertices, triangles = mcubes.marching_cubes(voxel3D, iso_value)
mcubes.export_mesh(vertices, triangles, "Cercles3D.dae", "Cercles3D")

# Afficher la reconstruction 3D
def plot_3d(vertices, triangles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], triangles, vertices[:, 2], cmap='Spectral')
    plt.show()

plot_3d(vertices, triangles)