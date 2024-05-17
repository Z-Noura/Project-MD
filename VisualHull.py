import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Charger les données
binary_image1 = np.load('Images/CerclesI1.npy')
binary_image2 = np.load('Images/CerclesI2.npy')
center_image1 = np.load('Images/CerclesC1.npy')
center_image2 = np.load('Images/CerclesC2.npy')
CerclesSegmented1 = np.load('Images/CerclesSegmented1.npy')
CerclesSegmented2 = np.load('Images/CerclesSegmented2.npy')

"""
# Afficher les images pour vérification
plt.imshow(binary_image1, cmap='gray')
plt.show()
plt.imshow(binary_image2, cmap='gray')
plt.show()
"""

# Points objets en espace 3D
objp = np.array([
    [0, 0, 0], [0.7, 0.7, 0], [0, 1, 0], [1.31, 1.31, 0],
    [0, 2, 0], [2, 2, 0], [0, 3, 0], [0, 4, 0]
], dtype=np.float32)


# Points images en espace 2D pour l'image 1
imgpoints1 = np.array([
    center_image1[0], center_image1[5], center_image1[3], center_image1[6],
    center_image1[1], center_image1[8], center_image1[4], center_image1[2]
], dtype=np.float32).reshape(-1, 1, 2)


# Points images en espace 2D pour l'image 2
imgpoints2 = np.array([
    center_image2[0], center_image2[5], center_image2[3], center_image2[7],
    center_image2[1], center_image2[8], center_image2[4], center_image2[2]
], dtype=np.float32).reshape(-1, 1, 2)


# Préparation des points pour la calibration
objpoints = [objp, objp]  # Points objets
imgpoints = [imgpoints1, imgpoints2]  # Points images


image_size = binary_image1.shape
print(image_size)


# Matrice intrinsèque initiale (à ajuster selon les besoins)
initial_intrinsic_matrix = np.array([[7.5, 0, image_size[1] / 2],
                                     [0, 7.5, image_size[0] / 2],
                                     [0, 0, 1]])


# Calibration de la caméra
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, initial_intrinsic_matrix, None, None, flags=cv.CALIB_USE_INTRINSIC_GUESS)


# Construire les matrices de projection
M1 = np.hstack((cv.Rodrigues(rvecs[0])[0], tvecs[0]))
M1 = np.dot(mtx, M1)


M2 = np.hstack((cv.Rodrigues(rvecs[1])[0], tvecs[1]))
M2 = np.dot(mtx, M2)


# Dimensions de l'image
image_width, image_height = binary_image1.shape[1], binary_image1.shape[0]


# Définir la taille des voxels
voxel_size = [1, 1, 1]  # Taille de voxel de 1 mm dans chaque dimension


# Définir les limites du volume 3D en fonction des dimensions de l'image
xlim = [0, image_width]
ylim = [0, image_height]
zlim = [0, 10]  # Ajuster cette valeur en fonction des besoins

# Afficher les dimensions de l'image et les limites du volume 3D pour vérification
print("Dimensions de l'image:", (image_width, image_height))
print("Limites du volume 3D (xlim, ylim, zlim):", (xlim, ylim, zlim))

# Initialiser la grille de voxels
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


    return voxel_grid, voxels_number

# Projeter les voxels sur les images 2D
def ProjectVoxels(voxel_grid, M):
    projected_points = M.dot(voxel_grid.T).T
    projected_points = projected_points[:, :2] / projected_points[:, 2:3]
    return projected_points

# Vérifier si les points projetés sont dans la région segmentée
def CheckInsideSegmented(image, points,plan):
    inside = []
    for pt in points:
        if plan == 'xy':
            x, y = int(pt[1]), int(pt[0])
        elif plan == 'yz':
            x,y = int(pt[1]),int(pt[0])
        else: 
            KeyError
        
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            if image[y, x] > 0:
                inside.append(True)
            else:
                inside.append(False)
        else:
            inside.append(False)
    return np.array(inside)


# Initialiser la grille de voxels
voxel_grid, voxels_number = InitializeVoxels(xlim, ylim, zlim, voxel_size)


# Projeter les voxels sur les deux images
projected_points1 = ProjectVoxels(voxel_grid, M1)
projected_points2 = ProjectVoxels(voxel_grid, M2)


# Vérifier si les points projetés sont dans la région segmentée pour chaque image
inside1 = CheckInsideSegmented(CerclesSegmented1, projected_points1,'xy')
inside2 = CheckInsideSegmented(CerclesSegmented2, projected_points2, 'yz')


# Créer les visual hulls pour chaque silhouette
visual_hull1 = voxel_grid[inside1, :3]
visual_hull2 = voxel_grid[inside2, :3]


# Rotate the second visual hull by 90 degrees around the y-axis
theta = np.radians(-90)
rotation_matrix = np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]
])


visual_hull2_rotated = (visual_hull2.dot(rotation_matrix.T))


# Plot the visual hulls
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(visual_hull1[:, 0], visual_hull1[:, 1], visual_hull1[:, 2], c='r', s=1, label='Image Droite')
ax.scatter(visual_hull2_rotated[:, 0], visual_hull2_rotated[:, 1], visual_hull2_rotated[:, 2], c='b', s=1, label='Image Gauche')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()