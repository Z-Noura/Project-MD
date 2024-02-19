import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as scio


data = scio.loadmat('matrix1.mat')
data_array = data['img']
print(data_array.shape)
I0=5
#(452,652,253)

listIf = []
Ifinalforeachrayon = 0
for zplanfinal in range(252):
    for yplanfinal in range(651):


        Ifinalforeachrayon = 0
        for xpourchaquerayon in range(451):
            
            nu = data_array[xpourchaquerayon,yplanfinal, zplanfinal]
            
            #print(nu)
            IX= I0*(np.exp((-nu)*0.1))
            
            Ifinalforeachrayon += IX
            
        listIf.append(Ifinalforeachrayon)
        # À intervalles réguliers, le pourcentage d'avancement est affiché pour surveiller le progrès.
        if( zplanfinal%(int(252/200))) == 0:

            print(int((100*zplanfinal)/252), '%')

        
listIf = np.array(listIf)

plane_shape = (252, 651)  # Define the shape of 2D plane
#les données sont remodelées sous forme d'une image 2D
listIf_2d = np.reshape(listIf, plane_shape)

sh=listIf.shape


plt.imshow(listIf_2d, cmap='viridis')  
plt.colorbar() 
 

plt.savefig('listIf_2d.png')
np.save('listIf_2d.npy', listIf_2d)
plt.show()

