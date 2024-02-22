import numpy as np
import matplotlib.pyplot as plt

# Charger l'image 2D
listIf_2d = np.load('listIf_2d.npy')

import numpy as np
import matplotlib.pyplot as plt

binarylistIf_2d = np.zeros_like(listIf_2d) 


unique_values, counts = np.unique(listIf_2d, return_counts=True)

listofvalue = []
listofcount= []

# Print the results
for value, count in zip(unique_values, counts):
    #print(f"{value} occurs {count} times")
    listofvalue.append(value)
    listofcount.append(count)

plt.bar(listofvalue,listofcount)
plt.title("histogram listIf_2d")
plt.show()

print(listofcount)
print(listofvalue)


for i in range(len(listIf_2d)):
    for j in range(len(listIf_2d[i])):
        
            
                if listIf_2d[i][j]>9020000:
                    binarylistIf_2d[i][j] = 1
                
print(binarylistIf_2d)
np.save('binary.npy',binarylistIf_2d)
segmented = np.zeros_like(listIf_2d) 

for i in range(0,len(listIf_2d)):
    for j in range(len(binarylistIf_2d[i])):
        
            
            
            if i!=0 and i!= len(binarylistIf_2d)-1 and j!=0 and j!= len(binarylistIf_2d[i])-1 :
                
                neighbor=[binarylistIf_2d[i+1][j],binarylistIf_2d[i-1][j],binarylistIf_2d[i][j-1],binarylistIf_2d[i][j+1]]
                if binarylistIf_2d[i][j]>0 and np.all(neighbor):
                    segmented[i][j] = 1

                
        



plt.imshow(listIf_2d, cmap='viridis')  
plt.colorbar() 
 

plt.savefig('segmented.png')
np.save('segmented.npy', segmented)
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