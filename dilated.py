import numpy as np
import matplotlib.pyplot as plt

# Charger l'image 2D
listIf_2d = np.load('listIf_2d.npy')
other_image = np.load('other_image.npy')

erosed = np.load("erosed.npy")
erosed_other_image= np.load("erosed_other_image.npy")

dilated = np.zeros_like(erosed)
dilated_other_image = np.zeros_like(erosed_other_image)

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



for i in range(0, len(erosed_other_image)):
    for j in range(len(erosed_other_image[i])):
        if i != 0 and i != len(erosed_other_image) - 1 and j != 0 and j != len(erosed_other_image[i]) - 1:
            #neighbors.append(a[i+1],a[i-1],a[i][j+1],a[i][j-1],)
            neighbor = [erosed_other_image[i - 1][j], erosed_other_image[i + 1][j], erosed_other_image[i][j - 1], erosed_other_image[i][j + 1]]
            if erosed_other_image[i][j] == 1:
                dilated_other_image[i - 1][j] = 1
                dilated_other_image[i + 1][j] = 1
                dilated_other_image[i][j - 1] = 1
                dilated_other_image[i][j + 1] = 1


# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Display the first image
axs[0].imshow(dilated, cmap='viridis')
axs[0].set_title('dilated Image 1')
axs[0].axis('off')

# Display the second image
axs[1].imshow(dilated_other_image, cmap='viridis')
axs[1].set_title('dilated Image 2')
axs[1].axis('off')



# Save the numpy arrays
np.save('dilated.npy', dilated)
np.save('dilated_other_image.npy', dilated_other_image)

plt.show()
