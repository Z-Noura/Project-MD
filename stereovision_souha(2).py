import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

# Load the data
binary_image1 = np.load('CerclesI1.npy')
print(binary_image1.shape[0])
binary_image2 = np.load('CerclesI2.npy')
print(binary_image2.shape)
center_image1= np.load('CerclesC1.npy')
print("center_image1",center_image1)
center_image2= np.load('CerclesC2.npy')
print("center_image2",center_image2)


# Resize the larger image to match the size of the smaller one
#if binary_image2.shape[1] > binary_image1.shape[1] or binary_image2.shape[0] > binary_image1.shape[0]:
#    binary_image2 = cv.resize(binary_image2, (binary_image1.shape[1], binary_image1.shape[0]))
#else:
#    binary_image2 = cv.resize(binary_image2, (binary_image1.shape[1], binary_image1.shape[0]))

# Object points in 3D space
objp = np.array([
    [center_image1[0][0], center_image1[0][1], 0],
    [center_image1[3][0], center_image1[3][1], 0],
    [center_image1[1][0], center_image1[1][1], 0],
    [center_image1[4][0], center_image1[4][1], 0],
    [center_image1[2][0], center_image1[2][1], 0],
    [center_image1[5][0], center_image1[5][1], 0]
], dtype=np.float32)
print("objp",objp)
print("objp[0]",objp[0])

# Image points in 2D space for image 2
objp2 = np.array([
    [center_image2[0][0], center_image2[0][1],0],
    [center_image2[3][0], center_image2[3][1],0],
    [center_image2[1][0], center_image2[1][1],0],
    [center_image2[4][0], center_image2[4][1],0],
    [center_image2[2][0], center_image2[2][1],0],
    [center_image2[6][0], center_image2[6][1],0]
], dtype=np.float32)
print("objp2",objp2)
# Image points in 2D space for image 1
imgpoints1 = np.array([
    [center_image1[0][0], center_image1[0][1]],
    [center_image1[3][0], center_image1[3][1]],
    [center_image1[1][0], center_image1[1][1]],
    [center_image1[4][0], center_image1[4][1]],
    [center_image1[2][0], center_image1[2][1]],
    [center_image1[5][0], center_image1[5][1]]
], dtype=np.float32)

# Image points in 2D space for image 2
imgpoints2 = np.array([
    [center_image2[0][0], center_image2[0][1]],
    [center_image2[3][0], center_image2[3][1]],
    [center_image2[1][0], center_image2[1][1]],
    [center_image2[4][0], center_image2[4][1]],
    [center_image2[2][0], center_image2[2][1]],
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
 
print(rvecs)
print("rvecs1",rvecs[0])
print("rvecs2",rvecs[1])
##matrice de rotation => convertir le vecteur en matrice
print("(rvecs[0])[0] ",(rvecs[0])[0] )
rmatRight = cv.Rodrigues(rvecs[0])[0] 
print("rmatRight",rmatRight)
rmatLeft = cv.Rodrigues(rvecs[1])[0]  

print("tvecs[0]",tvecs[0])
print("tvecs[1]",tvecs[1])
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

def matFondamental(camLeft,centerRight,camRight):
        
        return np.array((crossMat(camLeft @ centerRight)) @ camLeft @ np.linalg.pinv(camRight))
    





def mark_circle_centers(binary_image, center_image):
    # Create a new image with the same dimensions as binary_image
    center_marked_image = np.zeros_like(binary_image)
    
    # Iterate through each center and mark it as 1
    for center in center_image:
        # Retrieve center coordinates
        center_x, center_y = int(center[1]), int(center[0])  # Convert to integers
        
        # Ensure the center coordinates are within the bounds of the image
        if 0 <= center_x < binary_image.shape[1] and 0 <= center_y < binary_image.shape[0]:
            # Mark the center as 1 in the new image
            center_marked_image[center_y, center_x] = 1
    
    return center_marked_image



# Mark circle centers in both binary images
center_marked_image1 = mark_circle_centers(binary_image1, center_image1)
center_marked_image2 = mark_circle_centers(binary_image2, center_image2)

# Display the shape of the new images
print("Shape of center_marked_image1:", center_marked_image1)
print("Shape of center_marked_image2:", center_marked_image2.shape)

plt.imshow(center_marked_image1)
plt.show()


def getEpiLines(F, points):
    return F @ points


def findEpilines(path,center_image):
    epilines = [] # liste de chaque point gauche associe à sa ligne epipolaire
    for l in range(1):  #parcourt dossier images (26)
        
            strp = path 
    
    strp = path
    
        
        
    red = mark_circle_centers(strp, center_image)
    print("red",red)
    tempEpilines = []  #epiline associées a leurs points gauches temporairement 
    pointsLeft = [[],[],[]]
#Trouver Les points left'''        
    #i = lindice de chaque ligne epipolaire
    for i, line in enumerate(red):  #line = une liste de pixels rouges
        for pixel in line:
            if pixel != 0:
                pixel = 1
        try:
            #weighted average => (0*0 + 1*0 + 2*0 + ... + 1248 * 1 + 1249 * 0) / n° of red pixels
            #for instance => (1261+1262+1267)/3 = 1263.33
            #give position of the red line in x axis
            pointsLeft[0].append(np.average(range(264), weights = line)) #calcul des points de chaque ligne epipolair
                                    # moyenne pondere : on parcourt limage horizantalement de position 0 a position 1920
                                    #ces positions sont multipliés par la valeur du pixel(0 ou 1 cad rouge ou non)
            pointsLeft[1].append(i) # y axis
            pointsLeft[2].append(1)  
        except:
            pass
        #A partir de la ligne rouge de l'image de gauche, trouver l'épiline correspondante sur l'image de droite.
        #Trouver les épilines sur l'image de droite de tous les points rouges de l'image de gauche.

            #Calculer les epilines grace a la fonction getEpilines''' 
        
    print("pointsLeft",pointsLeft)    
    epilinesRight = getEpiLines(Fondamental, pointsLeft)
    
    tempEpilines.append(pointsLeft)
    tempEpilines.append(epilinesRight)
    epilines.append(tempEpilines)  # # liste de chaque point gauche associe à sa aligne epipolaire
    print("epilines",epilines)
    return epilines  


Fondamental = matFondamental(camRight,camWorldCenterLeft,camLeft)


epl = findEpilines(binary_image2, center_image2)  #liste de chaque point gauche associe à sa lignes epipolaires de la camera droite stocke sous forme de matrice a 2 colonnes
#scan gauche vers droite
epl= np.array(epl)
coef , length = epl[0][1].shape
print("coef , length ",coef , length )
print(epl.shape)
print(epl[0][1])

def lineY(coefs,x):
    a,b,c = coefs
    return-(c+a*x)/b
print()

def drawEpl(fname,EplRight):
    print(EplRight)
    #img = cv.imread(fname)
    img=fname
    
    coef , length = EplRight.shape #shape=number of elements in each dimension.
                                    #coef= nbre de lignes , length = nbre colonnes
    print(coef)
    print(length)
    print(coef)
    print(length)
    
    for i in range(0,length): #40 = le pas, pn dessine les epilines avec un intervalle de 40 verticalement donc jsq 1080
        plt.plot([0,264],[lineY(EplRight[:,i],0),lineY(EplRight[:,i],264)],'y')
        
        
    plt.imshow(img)
    plt.show()

drawEpl(binary_image1,epl[0][1])



'''Obtenir la position x du point rouge de chaque ligne de pixels (un seul point par ligne, d'où la position moyenne)'''
def getRedAvg(fname, center_image):
    red = mark_circle_centers(fname, center_image)
    redPoints = [[],[],[]]  

    for i, line in enumerate(red):
        for pixel in line:
            if pixel != 0:
                pixel = 1
        try:
            
            redPoints[0].append(np.average(range(264), weights = line)) #Calcule la moyenne pondérée le long de l'axe x.
            redPoints[1].append(i)
            redPoints[2].append(1)
        except:
            pass
    return redPoints


def eplRedPoints(path,center_image,EplRight):
    points = []
    
    strp = path
    redPoints = getRedAvg(strp,center_image)
    pointsRight = [[],[],[]]
    eplImg = EplRight[0][1] #garder uniquement les epilines et non les points rouges (EplRight[l][0])
    print("EplRight",EplRight)  
    print("eplImg",eplImg)                     
    print(strp)
    for i in range(len(eplImg[0])):
        try : 
            #x,y  les coordonnees du point rouge moyen de la ligne rouge (calculé pou)
            x = int(redPoints[0][i]) #resultat de getRedAvg
            y = int(lineY(eplImg[:,i],x)) #use of y = (-ax+c)/b to find y position
            pointsRight[0].append(x)
            pointsRight[1].append(y)
            pointsRight[2].append(1)
            
        except:
            pass
    points.append(pointsRight)
    #plt.imshow(scan)
    #plt.show()
    return points
pointsRight = eplRedPoints(binary_image1,center_image1,epl)
print("epl",epl)
print("pointsRight",pointsRight)

pointsRight = eplRedPoints(binary_image1,center_image1,epl)
print("pointsRight",pointsRight)


from mathutils import geometry as pygeo
from mathutils import Vector



def arrayToVector(p):
    return Vector((p[0],p[1],p[2]))


def getIntersection(pointsLeft,pointsRight):

    pL = np.array(pointsLeft)
    pR = np.array(pointsRight)
    
    camCenterRight = np.transpose(camWorldCenterRight)[0]
    camCenterLeft = np.transpose(camWorldCenterLeft)[0]

    # obtenir les coordonnées 3D du monde de tous les points 
                    #camLeft = mtx @ rotMatLeft
    leftObject = (np.linalg.pinv(camLeft) @ pL)
    
    rightObject = (np.linalg.pinv(camRight) @ pR) 

    # Points caractéristiques des lignes rétro-projetées debut et fin
    
    leftEndVec = arrayToVector(leftObject)
    rightEndVec = arrayToVector(rightObject)

    leftStartVec = arrayToVector(camCenterLeft)
    rightStartVec = arrayToVector(camCenterRight)
    print("leftEndVec",leftEndVec)
    # intersection entre deux lignes rétroprojetées = point du monde réel
    return pygeo.intersect_line_line(leftStartVec,leftEndVec,rightStartVec,rightEndVec)



def getObjectPoint():
    point = [[],[],[]]
    
        
    pointsLeft = np.array(epl[0][0])
    ("pointsLeft",pointsLeft)
    pointRight = np.array(pointsRight[0])
    
    for i in range(len(pointsLeft[0])):
        try:
            # calcul du point d'intersection sur l'objet -> on obtient une liste de vector
            intersection = getIntersection(pointsLeft[:,i],pointRight[:,i])
            print("intersection",intersection)
            for inter in intersection:
                inter *= 200
                x,y,z = inter
                point[0].append(x)
                point[1].append(y)
                point[2].append(z)
        except:
            #print('erroroooooor')
            pass
    return np.array(point)
        

def drawPointObject(points):
    
    ax = Axes3D(plt.figure())
    
    ax.scatter3D(points[0,:],points[1,:],points[2,:],c='purple',marker='^')     
        
    ax.view_init(-105,-75)
    plt.axis()
    plt.show()
point = getObjectPoint()
print(point)
drawPointObject(point) 