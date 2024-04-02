import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
# Load the data
Image1 = np.load('CerclesC1.npy')
print(Image1.shape[0])
Image2 = np.load('CerclesC2.npy')
print(Image2.shape)

Centre1 = np.load('CerclesI1.npy')
Centre2 = np.load('CerclesI2.npy')
print(Centre1)

# Resize the larger image to match the size of the smaller one
if Image2.shape[1] > Image1.shape[1] or Image2.shape[0] > Image1.shape[0]:
    Image2 = cv.resize(Image2, (Image1.shape[1], Image1.shape[0]))
else:
    Image2 = cv.resize(Image2, (Image1.shape[1], Image1.shape[0]))

# Object points in 3D space
objp = np.array([
    [Centre1 [0][0], Centre1 [0][1], 0],
    [Centre1 [1][0], Centre1 [1][1], 0],
    [Centre1 [2][0], Centre1 [2][1], 0],
    [Centre1 [3][0], Centre1 [3][1], 0],
    [Centre1 [4][0], Centre1 [4][1], 0],
    [Centre1 [5][0], Centre1 [5][1], 0]], 
    dtype=np.float32)
print("objp",objp)

# Image points in 2D space for image 2
objp2 = np.array([
    [Centre2 [0][0], Centre2 [0][1],0],
    [Centre2 [1][0], Centre2 [1][1],0],
    [Centre2 [2][0], Centre2 [2][1],0],
    [Centre2 [3][0], Centre2 [3][1],0],
    [Centre2 [4][0], Centre2 [4][1],0],
    [Centre2 [6][0], Centre2 [6][1],0]
], dtype=np.float32)

# Image points in 2D space for image 1
imgpoints1 = np.array([
    [Centre1 [0][0], Centre1 [0][1]],
    [Centre1 [1][0], Centre1 [1][1]],
    [Centre1 [2][0], Centre1 [2][1]],
    [Centre1 [3][0], Centre1 [3][1]],
    [Centre1 [4][0], Centre1 [4][1]],
    [Centre1 [5][0], Centre1 [5][1]]
], dtype=np.float32)

# Image points in 2D space for image 2
imgpoints2 = np.array([
    [Centre2 [0][0], Centre2 [0][1]],
    [Centre2 [1][0], Centre2 [1][1]],
    [Centre2 [2][0], Centre2 [2][1]],
    [Centre2 [3][0], Centre2 [3][1]],
    [Centre2 [4][0], Centre2 [4][1]],
    [Centre2 [6][0], Centre2 [6][1]]
], dtype=np.float32)

# Convert the lists into a format usable by the calibration function
objpoints = [objp] * 2  # List containing object points for both images
print(objpoints)

# Set the correct image sizes based on the resized images
image_size = (Image1.shape[0], Image1.shape[1])

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
    v = v[:, 0]   # Matrice ligne afin d'accéder plus facilement aux valeurs v[1]...
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def matFondamental(camLeft, centerRight, camRight):
    print("a",np.array((crossMat(camLeft @ centerRight)) @ camLeft @ np.linalg.pinv(camRight)))
    return np.array((crossMat(camLeft @ centerRight)) @ camLeft @ np.linalg.pinv(camRight))
    

F = matFondamental(camRight,camWorldCenterLeft,camLeft)
print(F)
# Liste des valeurs de x'^T F x pour chaque paire de points correspondants
values = [
    np.dot(np.dot(objp[i], F), objp2[i]) for i in range(len(objp))
]

# Imprimer les valeurs
for i, value in enumerate(values):
    print("Valeur pour la paire de points", i, ":", value)


def getEpiLines(F, points):
    return F @ points
print(F)
print(objp)
listI2 = []
for x1 in objp:
    I2=getEpiLines(F,x1)
    listI2.append(I2)
print(listI2)

print("obj2",objp2)
def findEpipiline(path, objp2, F):
    epipilines = []
    for center in objp2:
        epipilinesRight = getEpiLines(F, center)
        print("epipilinesRight",epipilinesRight)
        tempEpipiline = [center, epipilinesRight]  # Create a new list containing center and epipilinesRight
        epipilines.append(tempEpipiline)
    return epipilines

epl = findEpipiline('Image2.png',objp2,F)
epl=np.array(epl)
print(epl)
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Charger l'image binary_image_1
image_path = 'Image1.png'
image = plt.imread(image_path)

# Afficher l'image
plt.imshow(image, cmap='gray')

# Parcourir chaque élément de epl
for entry in epl:
    # Extraire les coordonnées des deux points de la ligne épipolaire
    point1 = entry[0]  # Premier point
    print("point1",point1)
    point2 = entry[1]  # Deuxième point
    print("point2",point2)
    
    # Extraire les coordonnées x et y des deux points
    x = [point1[0], point2[0]]  # Coordonnées x des deux points
    y = [point1[1], point2[1]]  # Coordonnées y des deux points
    
    # Tracer la ligne épipolaire
    plt.plot(x, y, color='red')  # Couleur rouge pour la ligne épipolaire

# Afficher le graphique
plt.xlabel('X')  # Libellé de l'axe des abscisses
plt.ylabel('Y')  # Libellé de l'axe des ordonnées
plt.title('Lignes épipolaires sur binary_image_1')  # Titre du graphique
plt.grid(True)  # Afficher une grille sur le graphique
plt.gca().set_aspect('equal', adjustable='box')  # Assurer l'aspect ratio correct
plt.show()
