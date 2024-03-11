import numpy as np
import matplotlib.pyplot as plt

# Charger l'image 2D
listIf_2d = np.load('listIf_2d.npy')
other_image = np.load('other_image.npy')

import numpy as np
import matplotlib.pyplot as plt
segmented = np.load("segmented.npy")
segmented_other_image = np.load("segmented_other_image.npy")
erosed = np.zeros_like(listIf_2d)
erosed_other_image = np.zeros_like(other_image)

for i in range(0,len(segmented)):
    for j in range(len(segmented[i])):
        if i!=0 and i!= len(segmented)-1 and j!=0 and j!= len(segmented[i])-1 :
            neighbor=[segmented[i-1][j],segmented[i+1][j],segmented[i][j-1],segmented[i][j+1]]
            if segmented[i][j]==1 and not(np.any(neighbor==0)):
                erosed[i][j] = 1

for i in range(0,len(segmented_other_image)):
    for j in range(len(segmented_other_image[i])):
        if i!=0 and i!= len(segmented_other_image)-1 and j!=0 and j!= len(segmented_other_image[i])-1 :
            neighbor=[segmented_other_image[i-1][j],segmented_other_image[i+1][j],segmented_other_image[i][j-1],segmented_other_image[i][j+1]]
            if segmented_other_image[i][j]==1 and not(np.any(neighbor==0)):
                erosed_other_image[i][j] = 1

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Display the first image
axs[0].imshow(erosed, cmap='viridis')
axs[0].set_title('Erosed Image 1')
axs[0].axis('off')

# Display the second image
axs[1].imshow(erosed_other_image, cmap='viridis')
axs[1].set_title('Erosed Image 2')
axs[1].axis('off')



# Save the numpy arrays
np.save('erosed.npy', erosed)
np.save('erosed_other_image.npy', erosed_other_image)

plt.show()
