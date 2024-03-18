import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

data = scio.loadmat('Cercles.mat')
data_array = data['img']
(size_x, size_y, size_z) = data_array.shape
# Assuming the given shape is (452, 652, 253)
# Plan xz (452, 253)

y_flag = 0
I0 = 20000
l = 1
if y_flag==0:
    x_source = int(size_x/2)
    y_source = -15000
    z_source = int(size_z / 2)
    y_final = size_y
else:
    x_source = -15000
    y_source = int(size_y/2)
    z_source = int(size_z / 2)
    x_final = size_x

def bresenham(x1, y1, z1, x2, y2, z2):
    """
    Returns a list of coordinates through which a ray sent
    from a source (x1, y1, z1) to a destination point (x2, y2, z2) passes.
    """
    V = []
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    
    amax = max(abs(dx), abs(dy), abs(dz))
    
    deltax = dx / amax
    deltay = dy / amax
    deltaz = dz / amax

    for i in range(amax):
        Vx = x1 + i * deltax
        Vy = y1 + i * deltay
        Vz = z1 + i * deltaz
        Vfinal = [int(round(Vx)), int(round(Vy)), int(round(Vz))]
        V.append(Vfinal)

    return V

def corrected_intensity_computation():
    I_array = np.zeros((2 * size_y, 2 * size_y))  # Pre-allocate the array for performance
    offset = size_y  # Offset to shift indices from [-size_y, size_y) to [0, 2*size_y)
    ind = 0
    for i in range(-size_y, size_y):  # Traverse the X
        for j in range(-size_y, size_y):  # Traverse the Y
            # Adjust indices to fit into the array
            adj_i = i + offset
            adj_j = j + offset
            if y_flag == 1:
                coords = bresenham(x_source, y_source, z_source, i, y_final, j)
            else:
                coords = bresenham(x_source, y_source, z_source, x_final, i, j)
            mu_total = 0
            for x, y, z in coords:  # Traverse the points the ray crosses
                if 0 <= x < size_x and 0 <= y < size_y and 0 <= z < size_z:
                    mu = data_array[x, y, z]
                else:
                    mu = 0
                    
                mu_total += mu  # Summing up the attenuation values
            
            if mu_total == 0:
                I = I0
            else:
                I = I0 * np.exp(-mu_total * l)  # Applying the exponential decay once
           
            I_array[adj_i, adj_j] = I
        ind += 1
        print(f'{100 * ind / (2*size_y):.1f} %')  # Improved progress print
    
    plt.imshow(I_array, extent=(-size_y, size_y, -size_y, size_y))
    plt.colorbar()
    plt.title('Corrected Intensity Image')
    plt.savefig('CircleC1bis.png')
    np.save('CircleC1bis.npy', I_array)
    plt.show()

# Call the function to perform the computation and plotting
corrected_intensity_computation()
