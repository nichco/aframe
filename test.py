import numpy as np

dim=12
k = np.random.rand(dim,dim)

mask = np.eye(dim)
mask[0,:] = 0


new_k = np.matmul(mask,k)
new_k = np.matmul(mask,new_k.T)
new_k = new_k.T


print(np.round(new_k,2))