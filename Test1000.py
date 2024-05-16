import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mcubes

Image_Silhouette_1 = np.load('CerclesSegmented1.npy')
Image_Silhouette_2 = np.load('CerclesSegmented2.npy')

Centres_1 = np.load('CerclesC1.npy')
Centres_2 = np.load('CerclesC2.npy')
image_size = Image_Silhouette_1.shape
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

