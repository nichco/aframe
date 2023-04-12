import numpy as np

n=4
k = np.random.rand(n,n)


row_mask = np.eye(n)
row_mask[0,:] = 0

col_mask = np.ones((n,1))



k = np.matmul(np.matmul(row_mask,k),row_mask)

print(k)