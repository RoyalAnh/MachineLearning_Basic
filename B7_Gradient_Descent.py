# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import math
import numpy as np 
import matplotlib.pyplot as plt

def grad(x):
    return 2*x+ 5*np.cos(x)

def cost(x):
    return x**2 + 5*np.sin(x)

def myGD1(eta, x0):
    x = [x0] # danh sách với giá trị x0 ban đầu
    for it in range(100): # lặp lại tối đa 100 lần 
        x_new = x[-1] - eta*grad(x[-1]) # x1 = x0 - eta*f'(x0)
        if abs(grad(x_new)) < 1e-3: # 0.001 DK hội tụ khi x_last xấp xỉ = x*
            break
        x.append(x_new)
    return (x, it) # trả về ds và số lần lặp

(x1, it1) = myGD1(.1, -5)
(x2, it2) = myGD1(.1, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))  # [-1] ;à GT cuối cùng trong DS
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))

np.random.seed(2) # seed(a) luôn cho duy nhất 1 giá trị

X = np.random.rand(1000, 1) # 1000 dòng và 1 cột chứa các giá trị ngẫu nhiên trong khoảng [0, 1].
y = 4 + 3 * X + .2*np.random.randn(1000, 1) # noise added độ lệch chuẩn là 0.2

# Building Xbar 
one = np.ones((X.shape[0],1)) # Tạo một cột ma trận toàn các giá trị 1 với cùng số dòng như X
Xbar = np.concatenate((one, X), axis = 1) # Ghép cột one vào trước ma trận X  để tạo ra ma trận Xbar

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_lr = np.dot(np.linalg.pinv(A), b) # w = (Xbar^T * Xbar)^(-1) * Xbar^T * y. np.linalg.pinv(A) là pseudo-inverse (nghịch đảo giả) của ma trận A
print('Solution found by formula: w = ',w_lr.T)

# Display result
w = w_lr
w_0 = w[0][0] # Trích xuất giá trị của hệ số tự do w_0.
w_1 = w[1][0]
x0 = np.linspace(0, 1, 2, endpoint=True)
y0 = w_0 + w_1*x0

# Draw the fitting line 
plt.plot(X.T, y.T, 'b.')     # data 
plt.plot(x0, y0, 'y', linewidth = 2)   # the fitting line
plt.axis([0, 1, 0, 10])
plt.show()

def grad(w):
    N = Xbar.shape[0] # trả về kích thước
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y) # dot ở đây là nhân 2 MT

def cost(w):
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2 # chuẩn norm của vector hoặc MT

def numerical_grad(w, cost):
    eps = 1e-4
    g = np.zeros_like(w) # tạo MT cùng số dòng vột với w, các GT = 0
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps 
        w_n[i] -= eps
        g[i] = (cost(w_p) - cost(w_n))/(2*eps)
    return g 

def check_grad(w, cost, grad):
    w = np.random.rand(w.shape[0], w.shape[1])
    grad1 = grad(w)
    grad2 = numerical_grad(w, cost)
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False 

print( 'Checking gradient...', check_grad(np.random.rand(2, 1), cost, grad))

def myGD(w_init, grad, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta*grad(w[-1])
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break 
        w.append(w_new)
    return (w, it) 

w_init = np.array([[2], [1]])
(w1, it1) = myGD(w_init, grad, 1)
print('Solution found by GD: w = ', w1[-1].T, ',\nafter %d iterations.' %(it1+1))