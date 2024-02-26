import numpy as np
import matplotlib.pyplot as plt

# Charger l'image 2D
Circle = np.load('Circle.npy')

import numpy as np
import matplotlib.pyplot as plt
segmented = np.load("Traitement image/segmented.npy")
erosed = np.load("Traitement image/erosed.npy")
dilated = np.load("Traitement image/dilated.npy")

difference = dilated-erosed
plt.imshow(difference, cmap='viridis')
plt.colorbar()

plt.savefig('Traitement image/difference.png')
np.save('Traitement image/difference.npy', difference)
plt.show()