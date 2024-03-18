import numpy as np
import matplotlib.pyplot as plt

# Charger l'image 2D
Circle = np.load('CircleB.npy')

erosed = np.load("Traitement image/erosed1.npy")

dilated = np.zeros_like(erosed)

for i in range(0, len(erosed)):
    for j in range(len(erosed[i])):
        if i != 0 and i != len(erosed) - 1 and j != 0 and j != len(erosed[i]) - 1:
            #neighbors.append(a[i+1],a[i-1],a[i][j+1],a[i][j-1],)
            neighbor = [erosed[i - 1][j], erosed[i + 1][j], erosed[i][j - 1], erosed[i][j + 1]]
            if erosed[i][j] == 1:
                dilated[i - 1][j] = 1
                dilated[i + 1][j] = 1
                dilated[i][j - 1] = 1
                dilated[i][j + 1] = 1

plt.imshow(dilated, cmap='viridis')
plt.colorbar()

plt.savefig('Traitement image/dilated1.png')
np.save('Traitement image/dilated1.npy', dilated)
plt.show()
