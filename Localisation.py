import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


listIf_2d = np.load('Traitement image/listIf_2d.npy')

w = listIf_2d.shape[1]
h = listIf_2d.shape[0]

centers = np.load('Centers.npy')

print(centers)
u1, v1 = centers[0][1], centers[0][0]
u2, v2 = centers[3][1], centers[3][0]
u3, v3 = centers[6][1], centers[6][0]
u4,v4 = centers[5][1], centers[5][0]

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

print(alpha)
print(beta)
print(gamma)

X1 = np.array([u1 - (w / 2), v1 - (h / 2), fd])
X2 = np.array([u2 - (w / 2), v2 - (h / 2), fd])
X3 = np.array([u3 - (w / 2), v3 - (h / 2), fd])
X4 = np.array([u4 - (w / 2), v4 - (h / 2), fd])

a=LA.norm(X3-X2)
a2=LA.norm(X4-X2)

b = LA.norm(X1-X3)
b2 = LA.norm(X1-X4)

c = LA.norm(X2-X1)




k = ((np.power(a,2))-(np.power(c,2)))/(np.power(b,2))
kte = ((np.power(a2,2))-(np.power(c,2)))/(np.power(b2,2))

k2 = ((np.power(a,2))+(np.power(c,2)))/(np.power(b,2))
k2te = ((np.power(a2,2))+(np.power(c,2)))/(np.power(b2,2))

k3 = ((np.power(b,2))+(np.power(c,2)))/(np.power(b,2))
k3te = ((np.power(b2,2))+(np.power(c,2)))/(np.power(b2,2))


k4 = ((np.power(b,2))-(np.power(a,2)))/(np.power(b,2))
k4te = ((np.power(b2,2))-(np.power(a2,2)))/(np.power(b2,2))


A4 = (np.power((k-1),2))-((4*(np.power(c,2)))/(np.power(b,2)))*np.power((np.cos(alpha)),2)
A4te = (np.power((kte-1),2))-((4*(np.power(c,2)))/(np.power(b2,2)))*np.power((np.cos(alpha2)),2)

A3 = 4*( (k*(1-k)*np.cos(beta)) - ((1-k2) *np.cos(alpha)*np.cos(gamma)) +(2*((np.power(c,2))/(np.power(b,2)))*np.power((np.cos(alpha)),2)*(np.cos(beta))))
A3te = 4*( (kte*(1-kte)*np.cos(beta2)) - ((1-k2te) *np.cos(alpha2)*np.cos(gamma)) +(2*((np.power(c,2))/(np.power(b2,2)))*np.power((np.cos(alpha2)),2)*(np.cos(beta2))))

A2 = 2 * (
    np.power(k, 2) - 1 +
    2 * np.power(k, 2) * np.power(np.cos(beta), 2) +
    2 * np.power(k3, 2) * np.power(np.cos(alpha), 2) -
    4 * k2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma) +
    2 * k4 * np.power(np.cos(gamma), 2)
)

A2te = 2 * (
    np.power(kte, 2) - 1 +
    2 * np.power(kte, 2) * np.power(np.cos(beta2), 2) +
    2 * np.power(k3te, 2) * np.power(np.cos(alpha2), 2) -
    4 * k2te * np.cos(alpha2) * np.cos(beta2) * np.cos(gamma) +
    2 * k4te * np.power(np.cos(gamma), 2)
)

A1 = 4*(  ((-k)*(1+k)*np.cos(beta)) + (((2 * np.power(a,2))/np.power(b,2))*np.power((np.cos(gamma)),2)*np.power(beta,2)) - ((1-k3)*np.cos(alpha)*(np.cos(gamma)))  )

A1te = 4*(  ((-kte)*(1+kte)*np.cos(beta2)) + (((2 * np.power(a2,2))/np.power(b2,2))*np.power((np.cos(gamma)),2)*np.power(beta2,2)) - ((1-k3te)*np.cos(alpha2)*(np.cos(gamma)))  )

A0 = ((np.power((1 + k),2)) - (4*(((np.power(a,2))/np.power(b ,2))* (np.power(np.cos(gamma),2))) ))

A0te = ((np.power((1 + kte),2)) - (4*(((np.power(a2,2))/np.power(b2 ,2))* (np.power(np.cos(gamma),2))) ))

print(A0)


print("A4",A4)
print("A3",A3)
# Définir l'équation quadratique
coefficients = [A4, A3, A2, A1, A0]
coefficients2 = [A4te, A3te, A2te, A1te, A0te]

# Résoudre l'équation quadratique
v = np.roots(coefficients)
v = np.real(v)

vte = np.roots(coefficients2)
vte = np.real(vte)

# Afficher les solutions
print("Solutions de l'équation quadratique :", v)
print("second Solutions de l'équation quadratique :", vte)


firstpart=(-1+k)*np.power(v,2)
secondpart = 2*k*v*np.cos(beta)
thirdpart = 1+k
fourthpart=2 *(np.cos(gamma) -(v*np.cos(alpha)))


firstpartte=(-1+kte)*np.power(vte,2)
secondpartte = 2*kte*v*np.cos(beta2)
thirdpartte = 1+kte
fourthpartte=2 *(np.cos(gamma) -(v*np.cos(alpha2)))


u = (firstpart-secondpart+thirdpart)/fourthpart
ute = (firstpartte-secondpartte+thirdpartte)/fourthpartte



print(u)

s1 = np.sqrt( (np.power(c,2))/(1+np.power(u,2)-(2*u*np.cos(gamma)))  )


s1te = np.sqrt((np.power(c, 2)) / (1 + np.power(ute, 2) - (2 * ute * np.cos(gamma))))
print("s1",s1)
print("s1te",s1te)



s2 = s1 * u
s2te=s1te*ute
s3= s1 * v
s3te= s1te * vte


print("s2",s2)
print("s2te",s2te)

print("s3",s3)
print("s3te",s3te)
s1 = np.array(s1)
s2 = np.array(s2)
s3 = np.array(s3)


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