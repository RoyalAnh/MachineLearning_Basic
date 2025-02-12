import numpy as np
from numpy import linalg as LA # toán đại số tuyến tính như SVD, giá trị riêng (eigenvalue), v.v

m, n = 2, 3
A = np.random.rand(m, n)

U, S, V = LA.svd(A) #  thực hiện phân tích SVD của ma trận A=U(m*m) * Sigma(m*n) * V.T(n*n)

# checking if U, V are orthogonal and S is a diagonal matrix with
# nonnegative decreasing elements
print ('Frobenius norm of (UU^T - I) =', LA.norm(U.dot(U.T) - np.eye(m)))
print ('\n S = ', S, '\n')
print ('Frobenius norm of (VV^T - I) =', LA.norm(V.dot(V.T) - np.eye(n)))