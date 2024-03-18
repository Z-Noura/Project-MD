import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, feature
from scipy.optimize import minimize

# Load the difference image
difference = np.load("difference.npy")

# Apply edge detection using Canny filter
edges = feature.canny(difference)

# Find contours in the edge-detected image
contours = measure.find_contours(edges, 0.1)
import numpy as np
import matplotlib.pyplot as plt

distance_connu_mm = 6.5  # Remplacez par la distance connue en millimètres
distance_connu_pixels = 650  # Remplacez par la distance correspondante en pixels

# Calcul de la résolution spatiale en pixels par millimètre
resolution_spatiale_ppmm = distance_connu_pixels / distance_connu_mm
print(resolution_spatiale_ppmm)
import numpy as np
import matplotlib.pyplot as plt

list1 = []

# Définir la résolution spatiale en pixels par millimètre
resolution_spatiale_ppmm = 100
import numpy as np
import matplotlib.pyplot as plt

list1 = []

# Définir la résolution spatiale en pixels par millimètre
resolution_spatiale_ppmm = 100

# Définir le rayon en millimètres
rayon_mm = 5

# Calculer le rayon en pixels
rayon_pixels = int(rayon_mm )

for i in range(len(difference)):
    for j in range(len(difference[i])):
        if difference[i][j] > 0:
            list1.append([i, j, rayon_pixels])

# Créer une image avec des cercles et des centres colorés sur un fond blanc
result_image = np.ones((len(difference), len(difference[0]), 3), dtype=np.uint8) * 255  # Fond blanc

for center in list1:
    # Dessiner le centre en bleu (couleur modifiée)
    result_image[center[0], center[1]] = [0, 0, 255]  # Bleu

    # Dessiner le cercle en violet
    for theta in range(0, 360):
        a = int(center[1] + center[2] * np.cos(np.radians(theta)))
        b = int(center[0] + center[2] * np.sin(np.radians(theta)))

        # Vérifier si les coordonnées sont valides
        if 0 <= a < result_image.shape[1] and 0 <= b < result_image.shape[0]:
            result_image[b, a] = [128, 0, 128]  # Violet

# Afficher l'image avec les cercles et les centres colorés sur un fond blanc
plt.imshow(result_image)
plt.show()
