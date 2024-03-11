import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as scio


data = scio.loadmat('Cercles.mat')
data_array = data['img']
print(data_array.shape)
list=[]

                    
I0=5
#(452,652,253)

listIf = []
Ifinalforeachrayon = 0
for xpourchaquerayon in range(451):#451

    for yplanfinal in range(651):#651
        Ifinalforeachrayon = 0
        
        for zplanfinal in range(251):#252
            
            nu = data_array[xpourchaquerayon,yplanfinal, zplanfinal]
            
            #print(nu)
            IX= I0*(np.exp((-nu)*0.1))
            
            Ifinalforeachrayon += IX
            
        listIf.append(Ifinalforeachrayon)
        # À intervalles réguliers, le pourcentage d'avancement est affiché pour surveiller le progrès.
        if( zplanfinal%(int(251/200))) == 0:#if( zplanfinal%(int(252/200)))

            print(int((100*zplanfinal)/252), '%')

        
listIf = np.array(listIf)

plane_shape = ( 451,651)  # Define the shape of 2D plane
#les données sont remodelées sous forme d'une image 2D
listIf_2d_other_image = np.reshape(listIf, plane_shape)

sh=listIf.shape


plt.imshow(listIf_2d_other_image, cmap='viridis')  
plt.colorbar() 
 

plt.savefig('other_image.png')
np.save('other_image.npy', listIf_2d_other_image)
plt.show()

