import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

# Load the data
binary_image1 = np.load('CerclesI1.npy')
print(binary_image1.shape[0])
binary_image2 = np.load('CerclesI2.npy')
print(binary_image2.shape)
center_image1= np.load('CerclesC2.npy')
print("center_image1",center_image1)
center_image2= np.load('CerclesC2.npy')
print("center_image2",center_image2)


# Resize the larger image to match the size of the smaller one
if binary_image2.shape[1] > binary_image1.shape[1] or binary_image2.shape[0] > binary_image1.shape[0]:
    binary_image2 = cv.resize(binary_image2, (binary_image1.shape[1], binary_image1.shape[0]))
else:
    binary_image2 = cv.resize(binary_image2, (binary_image1.shape[1], binary_image1.shape[0]))




# Object points in 3D space
objp = np.array([
    [center_image1[0][0], center_image1[0][1], 0],
    [center_image1[1][0], center_image1[1][1], 0],
    [center_image1[2][0], center_image1[2][1], 0],
    [center_image1[3][0], center_image1[3][1], 0],
    [center_image1[4][0], center_image1[4][1], 0],
    [center_image1[6][0], center_image1[6][1], 0]
], dtype=np.float32)
print("objp",objp)
print("objp[0]",objp[0])

# Image points in 2D space for image 2
objp2 = np.array([
    [center_image2[0][0], center_image2[0][1],0],
    [center_image2[1][0], center_image2[1][1],0],
    [center_image2[2][0], center_image2[2][1],0],
    [center_image2[3][0], center_image2[3][1],0],
    [center_image2[4][0], center_image2[4][1],0],
    [center_image2[6][0], center_image2[6][1],0]
], dtype=np.float32)
print("objp2",objp2)
# Image points in 2D space for image 1
imgpoints1 = np.array([
    [center_image1[0][0], center_image1[0][1]],
    [center_image1[1][0], center_image1[1][1]],
    [center_image1[2][0], center_image1[2][1]],
    [center_image1[3][0], center_image1[3][1]],
    [center_image1[4][0], center_image1[4][1]],
    [center_image1[6][0], center_image1[6][1]]
], dtype=np.float32)

# Image points in 2D space for image 2
imgpoints2 = np.array([
    [center_image2[0][0], center_image2[0][1]],
    [center_image2[1][0], center_image2[1][1]],
    [center_image2[2][0], center_image2[2][1]],
    [center_image2[3][0], center_image2[3][1]],
    [center_image2[4][0], center_image2[4][1]],
    [center_image2[6][0], center_image2[6][1]]
], dtype=np.float32)

# Convert the lists into a format usable by the calibration function
objpoints = [objp] * 2  # List containing object points for both images
print(objpoints)

# Set the correct image sizes based on the resized images
image_size = (binary_image1.shape[0], binary_image1.shape[1])

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
    

F = matFondamental(camLeft,camWorldCenterLeft,camRight)

def evaluer_contrainte_epipolaire(F, x_left, x_right):
    # Transposer x_left pour obtenir x_left'
    x_left_transpose = np.transpose(x_left)
    # Calculer x'^T * F * x
    contrainte_epipolaire = np.dot(np.dot(x_left_transpose, F), x_right)
    return contrainte_epipolaire
verif = evaluer_contrainte_epipolaire(F, objp2[0], objp[0])
print("verif",verif)



print(F)
# Liste des valeurs de x'^T F x pour chaque paire de points correspondants
values = [
    np.dot(np.dot(objp[i], F), objp2[i]) for i in range(len(objp))
]

# Imprimer les valeurs
for i, value in enumerate(values):
    print("Valeur pour la paire de points", i, ":", value)


def getEpipolarLines(F, points):
    return F @ points
print(F)
print(objp)
listI2 = []
for x1 in objp:
    I2=getEpipolarLines(F,x1)
    listI2.append(I2)
print(listI2)
""" I2=getEpipolarLines(F,imgpoints1())
print(I2) """
""" for l in range(1,3):
        strp = os.path.join(path,'binary_image'+str(l) + '.png')
        print(strp)
        image = plt.imread(strp)
        plt.imshow(image)
        plt.show() """
print("obj2",objp2)
def findEpipiline(path, objp2, F):
    epipilines = []
    for center in objp2:
        epipilinesRight = getEpipolarLines(F, center)
        print("epipilinesRight",epipilinesRight)
        tempEpipiline = [center, epipilinesRight]  # Create a new list containing center and epipilinesRight
        epipilines.append(tempEpipiline)
    return epipilines

epl = findEpipiline('CerclesC2.png',objp2,F)
for entry in epl:
    # Extraire les coordonnées des deux points de la ligne épipolaire
    point1 = entry[0]  # Premier point
    print("point1",point1)
    point2 = entry[1]  # Deuxième point
    print("point2",point2)
epl=np.array(epl)
print("epl[1][0]",epl[1][0])
# Assurez-vous que binary_image2 est une image en niveaux de gris avec une profondeur de couleur prise en charge
binary_image2 = binary_image2.astype(np.uint8)

# Tracé des épilignes
def drawEpilines(img, lines, pts):
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for line, pt in zip(lines, pts):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        # Extraire les coordonnées x et y des points
        x0, y0 = int(pt[0]), int(pt[1])
        # Calculer le deuxième point en utilisant la pente de l'épiligne
        x1 = int(x0 + 1000 * (-line[1])) # 1000 est une valeur arbitraire pour la longueur de la ligne
        y1 = int(y0 + 1000 * line[0])
        # Tracer la ligne
        img = cv.line(img, (x0, y0), (x1, y1), color, 1)
        # Tracer un petit cercle à l'extrémité du premier point pour indication
        img = cv.circle(img, (x0, y0), 5, color, -1)
    return img

# Tracé des épilignes sur l'image de gauche
img_with_epilines = drawEpilines(binary_image2, epl[:, 1], objp2)

# Affichage de l'image avec les épilignes tracées
cv.imshow('Image avec épilignes', img_with_epilines)
cv.waitKey(0)
cv.destroyAllWindows()
