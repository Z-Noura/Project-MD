import numpy as np
import matplotlib.pyplot as plt

# Charger l'image 2D
listIf_2d = np.load('listIf_2d.npy')

import numpy as np
import matplotlib.pyplot as plt
segmented = np.load("segmented.npy")
erosed = np.load("erosed.npy")
dilated = np.load("dilated.npy")

difference = dilated-erosed
plt.imshow(difference, cmap='viridis')
plt.colorbar()

plt.savefig('difference.png')
np.save('difference.npy', difference)
plt.show()