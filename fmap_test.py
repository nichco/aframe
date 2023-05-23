import numpy as np
import matplotlib.pyplot as plt

# create a beam mesh:
n = 10
x = np.zeros((n,3))
x[:,1] = np.linspace(0,10,n)

# create an OML mesh:
m = 20
y = np.zeros((m,3))
#y[:,0] = (np.random.rand(m) - 0.5)*0.25
y[:,0] = 0.25
y[:,1] = np.linspace(0,12,m)

# create OML forces:
f = np.ones((m,3))


# get the closest two beam node indices:
d = np.zeros((m,2))
for i in range(m):
    p = y[i,:]
    dist = np.sum((x - p)**2, axis=1)
    a = np.argsort(dist)[:2]
    d[i,:] = a


# create the weighting matrix:
weights = np.zeros((n,m))


for i in range(m):
    ia = int(d[i,0])
    ib = int(d[i,1])
    a = x[ia,:]
    b = x[ib,:]
    p = y[i,:]

    length = np.linalg.norm(b - a)
    norm = (b - a)/length
    t = np.dot(p - a, norm)
    # c is the closest point on the line segment (a,b) to point p:
    c =  a + t*norm

    ac = np.linalg.norm(c - a)
    bc = np.linalg.norm(c - b)
    l = max(length, bc)

    wa = (l - ac)/length
    wb = (l - bc)/length

    weights[ia, i] = wa
    weights[ib, i] = wb