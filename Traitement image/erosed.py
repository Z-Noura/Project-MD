import numpy as np
import matplotlib.pyplot as plt

# Charger l'image 2D
Circle = np.load('CircleTest.npy')

import numpy as np
import matplotlib.pyplot as plt
segmented = np.load("Traitement image/segmented.npy")
erosed = np.zeros_like(Circle)


for i in range(0,len(segmented)):
    for j in range(len(segmented[i])):
        
            if i!=0 and i!= len(segmented)-1 and j!=0 and j!= len(segmented[i])-1 :
               
                neighbor=[segmented[i-1][j],segmented[i+1][j],segmented[i][j-1],segmented[i][j+1]]
                if segmented[i][j]==1 and not(np.any(neighbor==0)):
                    erosed[i][j] = 1
                



plt.imshow(erosed, cmap='viridis')  
plt.colorbar() 
 

plt.savefig('Traitement image/erosed.png')
np.save('Traitement image/erosed.npy',erosed)
plt.show()
