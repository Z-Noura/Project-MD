import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_array = mesh.Mesh.from_file('Test1.stl')
print(data_array)

I0 = 20000
l = 0.1
x_source = 0
y_source = 256
z_source = 508
x_final = 511
height = 1017

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
    for j in range(512): #parcours les y
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