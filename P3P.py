import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


listIf_2d = np.load('listIf_2d.npy')

w = listIf_2d.shape[1]
h = listIf_2d.shape[0]
print(w)
print(h)

centers = np.load('centers.npy')
u1, v1 = centers[0][1], centers[0][0]
u2, v2 = centers[3][1], centers[3][0]
u3, v3 = centers[6][1], centers[6][0]
u4, v4 = centers[7][1], centers[7][0]
print(u3)
print(v3)

f = 200
d23 = 400
fd = f+d23
print(fd)

x1 = np.array([u1 - (w / 2), v1 - (h / 2), f])
x2 = np.array([u2 - (w / 2), v2 - (h / 2), f])
x3 = np.array([u3 - (w / 2), v3 - (h / 2), f])
x4 = np.array([u4 - (w / 2), v4 - (h / 2), f])
alpha = np.arccos(np.dot(x2, x3) / (LA.norm(x2) * LA.norm(x3)))
alpha2 = np.arccos(np.dot(x2, x4) / (LA.norm(x2) * LA.norm(x4)))
beta = np.arccos(np.dot(x3, x1) / (LA.norm(x3) * LA.norm(x1)))
beta2 = np.arccos(np.dot(x4, x1) / (LA.norm(x4) * LA.norm(x1)))
gamma = np.arccos(np.dot(x1, x2) / (LA.norm(x1) * LA.norm(x2)))

X1 = np.array([u1 - (w / 2), v1 - (h / 2), fd])
X2 = np.array([u2 - (w / 2), v2 - (h / 2), fd])
X3 = np.array([u3 - (w / 2), v3 - (h / 2), fd])
X4 = np.array([u4 - (w / 2), v4 - (h / 2), fd])

a=LA.norm(X3-X2)
print("a",a)
a2=LA.norm(X4-X2)
b = LA.norm(X1-X3)
b2 = LA.norm(X1-X4)
c = LA.norm(X2-X1)

k = ((np.power(a,2))-(np.power(c,2)))/(np.power(b,2))
kt = ((np.power(a2,2))-(np.power(c,2)))/(np.power(b2,2))
k2 = ((np.power(a,2))+(np.power(c,2)))/(np.power(b,2))

k2t = ((np.power(a2,2))+(np.power(c,2)))/(np.power(b2,2))

k3 = ((np.power(b,2))-(np.power(c,2)))/(np.power(b,2))
k3t = ((np.power(b2,2))-(np.power(c,2)))/(np.power(b2,2))

k4 = ((np.power(b,2))-(np.power(a,2)))/(np.power(b,2))
k4t = ((np.power(b2,2))-(np.power(a2,2)))/(np.power(b2,2))

A4 = (np.power((k-1),2))-(((4*(np.power(c,2)))/(np.power(b,2)))*np.power((np.cos(alpha)),2))
A4t = (np.power((kt-1),2))-(((4*(np.power(c,2)))/(np.power(b2,2)))*np.power((np.cos(alpha2)),2))

A3 = 4*( (k*(1-k)*np.cos(beta)) - ((1-k2) *np.cos(alpha)*np.cos(gamma)) +(2*((np.power(c,2))/(np.power(b,2)))*np.power((np.cos(alpha)),2)*(np.cos(beta))))
A3t = 4*( (kt*(1-kt)*np.cos(beta2)) - ((1-k2t) *np.cos(alpha2)*np.cos(gamma)) +(2*((np.power(c,2))/(np.power(b2,2)))*np.power((np.cos(alpha2)),2)*(np.cos(beta2))))

A2 = 2 * (
    np.power(k, 2) - 1 +
    2 * np.power(k, 2) * np.power(np.cos(beta), 2) +
    2 * np.power(k3, 2) * np.power(np.cos(alpha), 2) -
    4 * k2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma) +
    2 * k4 * np.power(np.cos(gamma), 2)
)
A2t = 2 * (
    np.power(kt, 2) - 1 +
    2 * np.power(kt, 2) * np.power(np.cos(beta2), 2) +
    2 * np.power(k3t, 2) * np.power(np.cos(alpha2), 2) -
    4 * k2t * np.cos(alpha2) * np.cos(beta2) * np.cos(gamma) +
    2 * k4t * np.power(np.cos(gamma), 2)
)

A1 = 4*(  ((-k)*(1+k)*np.cos(beta)) + (((2 * np.power(a,2))/np.power(b,2))*np.power((np.cos(gamma)),2)*np.cos(beta)) - ((1-k2)*np.cos(alpha)*(np.cos(gamma)))  )
A1t = 4*(  ((-kt)*(1+kt)*np.cos(beta2)) + (((2 * np.power(a2,2))/np.power(b2,2))*np.power((np.cos(gamma)),2)*np.cos(beta2)) - ((1-k2t)*np.cos(alpha2)*(np.cos(gamma)))  )


A0 = ((np.power((1 + k),2)) - (4*(((np.power(a,2))/np.power(b ,2))* (np.power(np.cos(gamma),2))) ))
A0t = ((np.power((1 + kt),2)) - (4*(((np.power(a2,2))/np.power(b2 ,2))* (np.power(np.cos(gamma),2))) ))
coefficients = [A4, A3, A2, A1, A0]
coefficientst = [A4t, A3t, A2t, A1t, A0t]

# Résoudre l'équation quadratique
v = np.roots(coefficients)
v = np.real(v)
vt = np.roots(coefficientst)
vt = np.real(vt)

firstpart=(-1+k)*np.power(v,2)
firstpartt=(-1+kt)*np.power(vt,2)
secondpart = 2*k*v*np.cos(beta)
secondpartt = 2*kt*vt*np.cos(beta2)
thirdpart = 1+k
thirdpartt = 1+kt
fourthpart=2 *(np.cos(gamma) -(v*np.cos(alpha)))
fourthpartt=2 *(np.cos(gamma) -(vt*np.cos(alpha2)))

u = (firstpart - secondpart + thirdpart) / fourthpart
ut = (firstpartt - secondpartt + thirdpartt) / fourthpartt
s1 = np.sqrt( (np.power(c,2))/(1+np.power(u,2)-(2*u*np.cos(gamma)))  )
s1t = np.sqrt( (np.power(c,2))/(1+np.power(ut,2)-(2*ut*np.cos(gamma)))  )

s2 = np.sqrt((np.power(c, 2)) / (1 + np.power(u, 2) - 2 * u * np.cos(gamma))) * u
s2t = np.sqrt((np.power(c, 2)) / (1 + np.power(ut, 2) - 2 * ut * np.cos(gamma))) * ut

s3= s1 * v
s3t= s1t * vt

s1 = np.array(s1)
s2 = np.array(s2)
s3 = np.array(s3)

s1t = np.array(s1t)
s2t= np.array(s2t)
s3t = np.array(s3t)

print("s1", s1)
print("s1t", s1t)
print("s2", s2)
print("s2t", s2t)
print("s3", s3)
print("s3t", s3t)





# Créer une figure
fig, ax = plt.subplots()

# Tracer les valeurs de s1, s2, s3
ax.plot(range(1, 5), s1, label='s1', marker='o')
ax.plot(range(1, 5), s2, label='s2', marker='o')
ax.plot(range(1, 5), s3, label='s3', marker='o')


# Ajouter des étiquettes et une légende
ax.set_xlabel('Pose de la caméra')
ax.set_ylabel('Longueur du rayon de projection')
ax.legend()

# Afficher le graphique
plt.show()


X_cam = np.array([X1, X2, X3])  # Remplacez par les coordonnées actuelles
print("X_cam",X_cam)
X_world = np.array([s1[2], s2[2], s3[2]]) 
print(s1[2])
print("X_world",X_world)

def arun(A, B):
    N = A.shape[1]
    assert B.shape[1] == N

    A_centroid = np.reshape(1 / N * (np.sum(A, axis=1)), (3, 1))
    B_centroid = np.reshape(1 / N * (np.sum(B, axis=1)), (3, 1))

    A_prime = A - A_centroid
    B_prime = B - B_centroid

    H = np.zeros([3, 3])
    for i in range(N):
        ai = A_prime[:, i]
        bi = B_prime[:, i]
        H = H + np.outer(ai, bi)

    U, S, V_transpose = np.linalg.svd(H)
    V = np.transpose(V_transpose)
    U_transpose = np.transpose(U)

    R = V @ np.diag([1, 1, np.linalg.det(V) * np.linalg.det(U_transpose)]) @ U_transpose
    t = B_centroid - R @ A_centroid

    return R, t


Rc, tc = arun(X_cam, X_world) 
 