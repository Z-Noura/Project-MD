import numpy as np
import cv2
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

# Load the binary image from the .npz file
binary_image = np.load('CerclesI2.npy')
label_image = label(binary_image)
centers = []
for region in regionprops(label_image):
    centers.append(region.centroid)

# Convert the center positions to a more convenient format (optional)
centers = np.array(centers)

print("Centers of the spheres:", centers)
# Convert numpy array to uint8
binary_image_uint8 = (binary_image * 255).astype(np.uint8)

# Create a kernel for morphological operations
kernel = np.ones((3,3), np.uint8)

# Apply opening to remove small artifacts or separate connected objects
processed_image = cv2.morphologyEx(binary_image_uint8, cv2.MORPH_OPEN, kernel)

# Follow the same steps to label and find centers on 'processed_image' instead of 'binary_image'

# Assuming 'binary_image' is the image you're working with
plt.imshow(binary_image, cmap='gray')
for center in centers:
    plt.plot(center[1], center[0], '*')  # Note: matplotlib's plot function expects (x, y), hence the reversal
plt.show()

np.save('CerclesF2.npy',centers)