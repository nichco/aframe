import numpy as np
import matplotlib.pyplot as plt

# create a beam mesh:
n = 10
x = np.zeros((n,3))
x[:,1] = np.linspace(0,10,n)

# create an OML mesh:
m = 20
y = np.zeros((m,3))
y[:,0] = (np.random.rand(m) - 0.5)*0.25
#y[:,0] = 0.25
y[:,1] = np.linspace(0,10,m)



# def closest_node(node, nodes):
#     nodes = np.asarray(nodes)
#     dist_a = np.sum((nodes - node)**2, axis=1)
#     #a = np.argmin(dist_a)
#     a = np.argsort(dist_a)[:2]
#     return a



d = np.zeros((m,2))
for i in range(m):
    p = y[i,:]
    dist = np.sum((x - p)**2, axis=1)
    a = np.argsort(dist)[:2]
    d[i,:] = a




fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2])
ax.scatter(x[:,0],x[:,1],x[:,2],color='yellow',edgecolors='black',linewidth=1,zorder=10)
ax.scatter(y[:,0],y[:,1],y[:,2],color='red',edgecolors='black',linewidth=1,zorder=10)


# create the weighting matrix:
weights = np.zeros((m,n))
for i in range(m):
    ia = int(d[i,0])
    ib = int(d[i,1])

    a = x[ia,:]
    b = x[ib,:]
    p = y[i,:]

    length = np.linalg.norm(b - a)
    norm = (b - a)/length
    ap = p - a
    t = np.dot(ap,norm)
    # c is the closest point on the line segment (a,b) to point p:
    c =  a + t*norm

    ax.scatter(c[0],c[1],c[2],color='k')

    vec_x = [c[0], p[0]]
    vec_y = [c[1], p[1]]
    vec_z = [c[2], p[2]]

    ax.plot(vec_x, vec_y, vec_z, color='grey')

    ac = np.linalg.norm(c - a)
    wa = 1 - (ac/length)
    wb = 1 - wa
    

    weights[i, ia] = wa
    weights[i, ib] = wb



# test the weights method:
u = np.zeros((n,3))
u[:,0] = np.linspace(0,0.1,n)

# calculate the new OML points as a function of the weights:
for i in range(m):
    ia = int(d[i,0])
    ib = int(d[i,1])

    a = x[ia,:] + u[ia,:]
    b = x[ib,:] + u[ib,:]
    ax.scatter(a[0],a[1],a[2],color='blue')
    ax.scatter(b[0],b[1],b[2],color='teal')


    wa = weights[i,ia]
    wb = weights[i,ib]

    p = y[i,:]


    pn = p + wa*u[ia,:] + wb*u[ib,:]

    ax.scatter(pn[0],pn[1],pn[2],color='g')




# for i in range(m):
#     index_a = int(d[i,0])
#     index_b = int(d[i,1])

#     xa = [y[i,0], x[index_a,0]]
#     ya = [y[i,1], x[index_a,1]]
#     za = [y[i,2], x[index_a,2]]
#     ax.plot(xa, ya, za)

#     xb = [y[i,0], x[index_b,0]]
#     yb = [y[i,1], x[index_b,1]]
#     zb = [y[i,2], x[index_b,2]]
#     ax.plot(xb, yb, zb, color='grey')



plt.show()