from cvxopt import matrix, solvers

# 1. Linear Programming : LP
c = matrix([-5., -3.]) # c^T * x = -5x1-3x2
G = matrix([[1., 2., 1., -1., 0.], [1., 1., 4., 0., -1.]])
h = matrix([10., 16., 32., 0., 0.]) # Gx <= h

solvers.options['show_progress'] = False # tắt việc in thông tin tiến trình.
sol = solvers.lp(c, G, h) # giải bài toán lập trình tuyến tính (LP).

print('Solution"')
print(sol['x'])

# 2. Quadratic Programming : QP  (bậc 2) có hàm mục tiêu: 1/2 * (x1^2 + x2^2) - 10x1 - 10x2
P = matrix([[1., 0.], [0., 1.]]) # P là ma trận đối xứng dương
q = matrix([-10., -10.]) # vector cho thành phần tuyến tính của hàm mục tiêu q^T * X
G = matrix([[1., 2., 1., -1., 0.], [1., 1., 4., 0., -1.]])
h = matrix([10., 16., 32., 0., 0])

solvers.options['show_progress'] = False
sol = solvers.qp(P, q, G, h)

print('Solution:')
print(sol['x'])

# 3. Geometric Programming : Hình học 
from math import log, exp# gp
from numpy import array
import numpy as np
K = [4]
F = matrix([[-1., 1., 1., 0.],
            [-1., 1., 0., 1.],
            [-1., 0., 1., 1.]])
g = matrix([log(40.), log(2.), log(2.), log(2.)])
solvers.options['show_progress'] = False
sol = solvers.gp(K, F, g)

print('Solution:')
print(np.exp(np.array(sol['x'])))

print('\nchecking sol^5')
print(np.exp(np.array(sol['x']))**5)