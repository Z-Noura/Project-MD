import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np

test = scio.loadmat('femur.mat')
np.save('femur',test['img'])
plt.imshow(test['img'][10])
plt.show()