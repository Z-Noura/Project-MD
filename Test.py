import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as scio

I0 = 20000
l = 0.1

#dI = I_array - I0
I_array = []
for i in range(1017): #parcours les Z
    SubI =  []
    for j in range(512): #parcours les Y
        Itot = 0
        for k in range(512): #parcours les X
            mu = data[j,k,i] 
            I = 20000*np.exp(-mu*l)
            Itot += I
            
            
        SubI.append(Itot)
       
    I_array.append(SubI) 
    if( i%(int(1017/200))) == 0:
        print(int(100*i/1017), '%')

I_final = I_array - I
print(I_array)

plt.imshow(I_final)
plt.show()