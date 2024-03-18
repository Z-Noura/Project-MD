import numpy as np
import matplotlib.pyplot as plt

# Charger l'image 2D
Circle = np.load('Circle.npy')

import numpy as np
import matplotlib.pyplot as plt

binaryCircle = np.zeros_like(Circle) 


unique_values, counts = np.unique(Circle, return_counts=True)

listofvalue = []
listofcount= []

# Print the results
for value, count in zip(unique_values, counts):
    #print(f"{value} occurs {count} times")
    listofvalue.append(value)
    listofcount.append(count)

plt.bar(listofvalue,listofcount)
plt.title("histogram Circle")
plt.show()

print(listofcount)
print(listofvalue)


for i in range(len(Circle)):
    for j in range(len(Circle[i])):
        
            
                if Circle[i][j]>9020000:
                    binaryCircle[i][j] = 1
                
print(binaryCircle)
np.save('binary.npy',binaryCircle)
segmented = np.zeros_like(Circle) 

for i in range(0,len(Circle)):
    for j in range(len(binaryCircle[i])):
        
            
            
            if i!=0 and i!= len(binaryCircle)-1 and j!=0 and j!= len(binaryCircle[i])-1 :
                
                neighbor=[binaryCircle[i+1][j],binaryCircle[i-1][j],binaryCircle[i][j-1],binaryCircle[i][j+1]]
                if binaryCircle[i][j]>0 and np.all(neighbor):
                    segmented[i][j] = 1

                
        



plt.imshow(Circle, cmap='viridis')  
plt.colorbar() 
 

plt.savefig('Traitement image/segmented.png')
np.save('Traitement image/segmented.npy', segmented)
plt.show()

unique_values, counts = np.unique(segmented, return_counts=True)

listofvaluesegmented = []
listofcountsegmented = []

# Print the results
for value, count in zip(unique_values, counts):
    listofvaluesegmented.append(value)
    listofcountsegmented.append(count)

plt.bar(listofvaluesegmented, listofcountsegmented)
plt.title("histogram segmented")
plt.show()

print(listofcountsegmented)
print(listofvaluesegmented)