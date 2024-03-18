import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

# Charger l'image 2D
Circles = np.load('Traitement image/difference.npy')
s = [[1,1,1],[1,1,1],[1,1,1]]
labeled_array,num_features = label(Circles, structure = s)
print(num_features)

plt.imshow(labeled_array)
np.save("Traitement image/labeled.npy",labeled_array)
plt.show()

