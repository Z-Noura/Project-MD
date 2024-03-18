import numpy as np
import matplotlib.pyplot as plt

def bresenham_optimized(x1, y1, z1, x2, y2, z2):
    """
    An optimized version of the Bresenham algorithm for 3D.
    """
    V = []
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    amax = max(abs(dx), abs(dy), abs(dz))
    deltax, deltay, deltaz = dx / amax, dy / amax, dz / amax
    Vx, Vy, Vz = x1, y1, z1

    for _ in range(amax + 1):
        V.append([int(round(Vx)), int(round(Vy)), int(round(Vz))])
        Vx += deltax
        Vy += deltay
        Vz += deltaz

    return V

def corrected_intensity_computation_optimized(data_array, I0, l, x_source, y_source, z_source, x_final):
    height, width, depth = data_array.shape
    I_array = np.zeros((width, depth))
    
    for j in range(width):  # Note: Adjusted to match your data orientation
        for k in range(depth):
            coords = bresenham_optimized(x_source, y_source, z_source, x_final, j, k)
            mu_total = sum(data_array[y, x, z] if 0 <= x < height and 0 <= y < width and 0 <= z < depth else 0 for x, y, z in coords)
            I_array[j, k] = I0 * np.exp(-mu_total * l)

    plt.imshow(I_array.T)  # Transpose to match the orientation
    plt.savefig('CircleB1bis_optimized.png')
    np.save('CircleB1bis_optimized.npy', I_array)
    plt.show()

# Example call (ensure you have defined all the necessary variables and have the data loaded)
# corrected_intensity_computation_optimized(data_array, I0, l, x_source, y_source, z_source, x_final)
