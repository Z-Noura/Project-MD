import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

data = scio.loadmat('Cercles.mat')
data_array = data['img']
print(data_array)
#(452,652,253)

I0 = 20000
l = 1
x_source = -150000
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

def corrected_intensity_computation():
    I_array = np.zeros((height, 452))  # Pre-allocate the array for performance
    for i in range(height):  # traverse the Z
        for j in range(452):  # traverse the Y
            coords = bresenham(x_source, y_source, z_source, x_final, j, i)
            mu_total = 0
            for x, y, z in coords:  # traverse the points the ray crosses
                if (x<0) or (y<0) or (z<0):
                    mu = 0
                else:
                    mu = data_array[y, x, z]
                mu_total += mu  # Summing up the attenuation values
            I = I0 * np.exp(-mu_total * l)  # Applying the exponential decay once

            I_array[i, j] = I
        print(f'{100 * i / height:.1f} %')  # Improved progress print

    plt.imshow(I_array)
    plt.savefig('CircleB1.png')
    np.save('CircleB1.npy',I_array)
    plt.show()

# Call the function to perform the computation and plotting
corrected_intensity_computation()