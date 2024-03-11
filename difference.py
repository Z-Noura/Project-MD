import numpy as np
import matplotlib.pyplot as plt

# Charger l'image 2D
listIf_2d = np.load('listIf_2d.npy')
other_image = np.load('other_image.npy')


segmented = np.load("segmented.npy")
segmented_other_image = np.load("segmented_other_image.npy")
erosed = np.load("erosed.npy")
erosed_other_image= np.load("erosed_other_image.npy")
dilated = np.load("dilated.npy")
dilated_other_image= np.load("dilated_other_image.npy")

difference = dilated-erosed
difference_other_image = dilated_other_image-erosed_other_image

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Display the first image
axs[0].imshow(difference, cmap='viridis')
axs[0].set_title('difference Image 1')
axs[0].axis('off')

# Display the second image
axs[1].imshow(difference_other_image, cmap='viridis')
axs[1].set_title('difference Image 2')
axs[1].axis('off')



# Save the numpy arrays
np.save('difference.npy', difference)
np.save('difference_other_image.npy', difference_other_image)

plt.show()
print(difference.shape)
print(difference_other_image.shape)
