import numpy as np
import matplotlib.pyplot as plt

# Charger l'image 2D
Circle = np.load('CircleB.npy')

import numpy as np
import matplotlib.pyplot as plt
segmented = np.load("Traitement image/segmented1.npy")
erosed = np.load("Traitement image/erosed1.npy")
dilated = np.load("Traitement image/dilated1.npy")

difference = dilated-erosed
plt.imshow(difference, cmap='viridis')
plt.colorbar()

plt.savefig('Traitement image/difference1.png')
np.save('Traitement image/difference1.npy', difference)
plt.show()