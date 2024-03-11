import numpy as np
import cv2 as cv

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
 
 

##matrice de rotation => convertir le vecteur en matrice
rmatRight = cv.Rodrigues(rvecs[0])[0] #5=6eme image car meilleur detection des coins
rmatLeft = cv.Rodrigues(rvecs[1])[0]  #4=5eme image car meilleur detection des coins


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






