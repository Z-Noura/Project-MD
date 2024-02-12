import matplotlib.pyplot as plt
import scipy.io as scio

test = scio.loadmat('matrix1.mat')
plt.imshow(test['img'][10])
plt.show()