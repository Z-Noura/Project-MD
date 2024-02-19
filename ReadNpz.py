import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as scio

data = scio.loadmat('matrix1.mat')
data_array = data['img']
print(data_array)
#(452,652,253)

I0 = 20000
l = 0.1
x_source = -400
y_source = 326
z_source = 127
x_final = 451
height = 252

def bresenham(x1,y1,z1,x2,y2,z2):
    """
    Renvoie dans une liste les coordonnées des points par lequel passe un rayon envoyés
    d'une source (x1,y1,z1) à un point de destination (x2,y2,z2)
    """
    V = []
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    
    amax = max(dx,dy,dz)

    deltax = dx/amax
    deltay = dy/amax
    deltaz = dz/amax

    for i in range(amax):
        Vx = x1 + i*deltax
        Vy = y1 + i*deltay
        Vz = z1 + i*deltaz
        Vfinal = [int(Vx),int(Vy),int(Vz)]
        V.append(Vfinal)

    return V


I_array = []
for i in range(height): #parcours les Z
    SubI =  []
    for j in range(452): #parcours les y
        Itot = 0
        coords = bresenham(x_source,y_source,z_source,x_final,j,i)
       
        #print(coords)

        
        for k in range(len(coords)): #parcours les points par lequel le rayon traverse la matiere
            
            x = coords[k][0]
            y = coords[k][1]
            z = coords[k][2]
            mu = data_array[y,x,z] 
            I = 20000*np.exp(-mu*l)
            Itot += I
            
            
        SubI.append(Itot)
       
    I_array.append(SubI)
    if( i%(int(height/200))) == 0:
        print(int(100*i/height), '%')


print(I_array)

plt.imshow(I_array)
plt.show() 