import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# Load the data
centers = np.load('binary_image1.npy')
centers_other_image = np.load('binary_image2.npy')
center_image1= np.load('centers_image1.npy')
center_image2= np.load('centers_image_2.npy')
print(center_image1)

# Resize the larger image to match the size of the smaller one
if centers_other_image.shape[1] > centers.shape[1] or centers_other_image.shape[0] > centers.shape[0]:
    centers_other_image = cv.resize(centers_other_image, (centers.shape[1], centers.shape[0]))
else:
    centers_other_image = cv.resize(centers_other_image, (centers.shape[1], centers.shape[0]))

""" focal_length_x = 100
focal_length_y = 100
center_x = 0
center_y = 0
initial_mtx1 = np.array([
    [focal_length_x, 0, center_x],
    [0, focal_length_y, center_y],
    [0, 0, 1]
], dtype=np.float32)

initial_mtx2 = np.array([
    [focal_length_x, 0, center_x],
    [0, focal_length_y, center_y],
    [0, 0, 1]
], dtype=np.float32) """


# Object points in 3D space
objp = np.array([
    [center_image1[0][0], center_image1[0][1], 0],
    [center_image1[1][0], center_image1[1][1], 0],
    [center_image1[2][0], center_image1[2][1], 0],
    [center_image1[3][0], center_image1[3][1], 0],
    [center_image1[4][0], center_image1[4][1], 0],
    [center_image1[5][0], center_image1[5][1], 0]
], dtype=np.float32)
print("objp",objp)

# Image points in 2D space for image 1
imgpoints1 = np.array([
    [center_image1[0][0], center_image1[0][1]],
    [center_image1[1][0], center_image1[1][1]],
    [center_image1[2][0], center_image1[2][1]],
    [center_image1[3][0], center_image1[3][1]],
    [center_image1[4][0], center_image1[4][1]],
    [center_image1[5][0], center_image1[5][1]]
], dtype=np.float32)

# Image points in 2D space for image 2
imgpoints2 = np.array([
    [center_image2[0][0], center_image2[0][1]],
    [center_image2[1][0], center_image2[1][1]],
    [center_image2[2][0], center_image2[2][1]],
    [center_image2[3][0], center_image2[3][1]],
    [center_image2[4][0], center_image2[4][1]],
    [center_image2[5][0], center_image2[5][1]]
], dtype=np.float32)

# Convert the lists into a format usable by the calibration function
objpoints = [objp] * 2  # List containing object points for both images
print(objpoints)

# Set the correct image sizes based on the resized images
image_size = (centers.shape[0], centers.shape[1])

# Perform camera calibration with the initial guess for the camera matrix
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, [imgpoints1, imgpoints2], image_size,  None, None)
print(ret)
print(tvecs)
 
 
print("rvecs",rvecs[0])
##matrice de rotation => convertir le vecteur en matrice
rmatRight = cv.Rodrigues(rvecs[0])[0] 
print("rmatRight",rmatRight)
rmatLeft = cv.Rodrigues(rvecs[1])[0]  


#matrice complète [R|t] => ajouter t dans R
rotMatRight = np.concatenate((rmatRight,tvecs[0]), axis=1)
rotMatLeft = np.concatenate((rmatLeft,tvecs[1]), axis=1)
#print(rotMatLeft)


#matrice de la camera @ RT
camLeft = mtx @ rotMatLeft
camRight = mtx @ rotMatRight

# trouver cx et cy (coo centre optique dans limage) pour les 2 cameras
camWorldCenterLeft = np.linalg.inv(np.concatenate((rotMatLeft,[[0,0,0,1]]), axis=0)) @ np.transpose([[0,0,0,1]])
camWorldCenterRight = np.linalg.inv(np.concatenate((rotMatRight,[[0,0,0,1]]), axis=0)) @ np.transpose([[0,0,0,1]])
print('Centre Gauche\n',camWorldCenterLeft) #1colonne à 4lignes
print('Centre Gauche\n', camWorldCenterRight) #1colonne à 4lignes


def crossMat(v):
    v = v[:,0]   #matrice ligne afin d'acceder plus facilement aux valeurs v[1]...
    return np.array([ [ 0,-v[2],v[1] ],
                      [ v[2],0,-v[0] ],
                      [-v[1],v[0],0 ] ]) #l’image du centre optique = 𝑃′𝐶⃗ = [𝑥]×



def matFondamental(camLeft,centerRight,camRight):
        
        return np.array((crossMat(camLeft @ centerRight)) @ camLeft @ np.linalg.pinv(camRight))

""" def triangulate_points(camLeft, camRight, pts_left, pts_right):
    # Convert points to homogeneous coordinates
    pts_left = np.hstack((pts_left, np.ones((pts_left.shape[0], 1))))
    pts_right = np.hstack((pts_right, np.ones((pts_right.shape[0], 1))))

    # Calculate the epipolar lines
    lines_right = np.dot(camRight.T, pts_left.T).T
    lines_left = np.dot(camLeft.T, pts_right.T).T

    # Triangulate the points
    points_3d = []
    for i in range(pts_left.shape[0]):
        A = np.array([
            [pts_left[i][0] * camLeft[2][0] - camLeft[0][0], pts_left[i][0] * camLeft[2][1] - camLeft[1][0],
             pts_left[i][0] * camLeft[2][2] - camLeft[0][2]],
            [pts_left[i][1] * camLeft[2][0] - camLeft[0][1], pts_left[i][1] * camLeft[2][1] - camLeft[1][1],
             pts_left[i][1] * camLeft[2][2] - camLeft[1][2]],
            [pts_right[i][0] * camRight[2][0] - camRight[0][0], pts_right[i][0] * camRight[2][1] - camRight[1][0],
             pts_right[i][0] * camRight[2][2] - camRight[0][2]],
            [pts_right[i][1] * camRight[2][0] - camRight[0][1], pts_right[i][1] * camRight[2][1] - camRight[1][1],
             pts_right[i][1] * camRight[2][2] - camRight[1][2]]
        ])

        B = np.array([
            camLeft[0][2] - pts_left[i][0] * camLeft[2][2],
            camLeft[1][2] - pts_left[i][1] * camLeft[2][2],
            camRight[0][2] - pts_right[i][0] * camRight[2][2],
            camRight[1][2] - pts_right[i][1] * camRight[2][2]
        ])

        point_3d_homogeneous = np.linalg.lstsq(A, B, rcond=None)[0]
        point_3d_homogeneous = np.append(point_3d_homogeneous, 1)  # Convert back to homogeneous coordinates
        if len(point_3d_homogeneous) == 4 and point_3d_homogeneous[3] != 0:
            points_3d.append(point_3d_homogeneous / point_3d_homogeneous[3])
        else:
            print("Unable to triangulate point:", i)

    return np.array(points_3d)
points_3d = triangulate_points(camLeft, camRight, imgpoints1, imgpoints2)
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Extract x, y, z coordinates from 3D points
x = points_3d[:, 0]
y = points_3d[:, 1]
z = points_3d[:, 2]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o')

# Set labels and title
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('3D Reconstruction')

# Show plot
plt.show()
 """
# Utiliser la matrice de la caméra (mtx) et les coordonnées 2D pour calculer les coordonnées 3D
def calculate_3d_coordinates(mtx, imgpoints):
    # Matrice inverse de la caméra
    mtx_inv = np.linalg.inv(mtx)
    
    # Convertir les coordonnées 2D en coordonnées homogènes
    imgpoints_homogeneous = np.hstack((imgpoints, np.ones((imgpoints.shape[0], 1))))
    
    # Calculer les coordonnées 3D
    points_3d = np.dot(mtx_inv, imgpoints_homogeneous.T).T
    points_3d /= points_3d[:, 2][:, np.newaxis]  # Normaliser les coordonnées homogènes
    
    return points_3d[:, :3]  # Retourner uniquement les trois premières colonnes pour les coordonnées 3D

# Calculer les coordonnées 3D pour les deux images
points_3d_left = calculate_3d_coordinates(mtx, imgpoints1)
points_3d_right = calculate_3d_coordinates(mtx, imgpoints2)

# Afficher les résultats
print("Points 3D pour l'image 1:", points_3d_left)
print("Points 3D pour l'image 2:", points_3d_right)

#Extract x, y, z coordinates from 3D points
x = points_3d_right[:, 0]
y = points_3d_right[:, 1]
z = points_3d_right[:, 2]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o')

# Set labels and title
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('3D Reconstruction')

# Show plot
plt.show()
