import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

data = scio.loadmat('Cercles.mat')
data_array = data['img']
(size_x,size_y,size_z) = data_array.shape
print((size_x,size_y,size_z))
#(452,652,253)
#plan yz (652,253)

I0 = 20000
l = 1
x_source = -15000
y_source = int(size_y/2)
z_source = int(size_z/2)
x_final = size_x
height = size_x

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
    I_array = np.zeros((2*size_y,2*size_y))  # Pre-allocate the array for performance
    ind = 0
    for i in range(-size_y,size_y):  # traverse the X
        for j in range(-size_y,size_y):  # traverse the Y
            coords = bresenham(x_source, y_source, z_source, x_final, i, j)
            mu_total = 0
            for x, y, z in coords:  # traverse the points the ray crosses
                if (x<0) or (y<0) or (z<0) or (x>=size_x) or (y>=size_y) or (z>=size_z):
                    mu = 0
                else:
                    try:
                        mu = data_array[x, y, z]
                    except:
                        print((x,y,z))
                        break
                    
                mu_total += mu  # Summing up the attenuation values
            I = I0 * np.exp(-mu_total * l)  # Applying the exponential decay once
            #print(i," ", j)
            I_array[i, j] = I
        ind+=1
        print(f'{100 * ind / (2*size_y):.1f} %')  # Improved progress print

    plt.imshow(I_array)
    plt.savefig('CircleC2.png')
    np.save('CircleC2.npy',I_array)
    plt.show()

# Call the function to perform the computation and plotting
corrected_intensity_computation()