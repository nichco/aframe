import numpy as np

dim=12
k = np.random.rand(dim,dim)

mask = np.eye(dim)

new_k = np.matmul(mask,k)


print(np.round(new_k,2))