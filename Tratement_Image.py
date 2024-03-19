import numpy as np
import matplotlib.pyplot as plt

# Charger l'image 2D
Circle = np.load('femur.npy')

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
#plt.show()
print(listofcount)
print(listofvalue)

for i in range(len(Circle)):
    for j in range(len(Circle[i])):
        if Circle[i][j]<15000:
            binaryCircle[i][j] = 1
                
print(binaryCircle)
#np.save('binary.npy',binaryCircle)
plt.imshow(binaryCircle, cmap='viridis')  
#plt.show()

segmented = np.zeros_like(Circle) 

for i in range(0,len(Circle)):
    for j in range(len(binaryCircle[i])):
            if i!=0 and i!= len(binaryCircle)-1 and j!=0 and j!= len(binaryCircle[i])-1 :
                neighbor=[binaryCircle[i+1][j],binaryCircle[i-1][j],binaryCircle[i][j-1],binaryCircle[i][j+1]]
                if binaryCircle[i][j]>0 and np.all(neighbor):
                    segmented[i][j] = 1

plt.imshow(Circle, cmap='viridis')  
plt.colorbar() 

#plt.savefig('Traitement image/segmented.png')
#np.save('Traitement image/segmented.npy', segmented)
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

#Erosion
erosed = np.zeros_like(Circle)
for i in range(0,len(segmented)):
    for j in range(len(segmented[i])):
            if i!=0 and i!= len(segmented)-1 and j!=0 and j!= len(segmented[i])-1 :
                neighbor=[segmented[i-1][j],segmented[i+1][j],segmented[i][j-1],segmented[i][j+1]]
                if segmented[i][j]==1 and not(np.any(neighbor==0)):
                    erosed[i][j] = 1
                
plt.imshow(erosed, cmap='viridis')  
plt.colorbar() 

#plt.savefig('Traitement image/erosed.png')
#np.save('Traitement image/erosed.npy',erosed)
#plt.show()

#dilatation
dilated = np.zeros_like(erosed)

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

plt.imshow(dilated, cmap='viridis')
plt.colorbar()
#plt.savefig('Traitement image/dilated.png')
#np.save('Traitement image/dilated.npy', dilated)
#plt.show()

#difference

difference = dilated-erosed
plt.imshow(difference, cmap='viridis')
plt.colorbar()

#plt.savefig('Traitement image/difference.png')
#np.save('Traitement image/difference.npy', difference)
plt.savefig('FemurFinal1.png')
np.save('FemurFinal1.npy', difference)
plt.show()
