# Hồi quy tuyến tính
import matplotlib.pyplot as plt # vẽ hình
import numpy as np # đại số tuyến tính
from sklearn import datasets, linear_model # thư viện scikit-learn của Python để tìm nghiệm.

# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# Building Xbar 
one = np.ones((X.shape[0], 1)) # mảng cột full số 1, cùng số hàng với X
Xbar = np.concatenate((one, X), axis = 1) # gồm 1 cột full 1 và côtj dữ liệu chiều cao

# Calculating weights of the fitting line 
A = np.dot(Xbar.T, Xbar) # ma trận hiệp phương sai
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b) # hồi quy tuyéne tính ( nghịch đảo)
print('w = ', w)

# Preparing the fitting line 
w_0 = w[0][0] # hệ số chệch
w_1 = w[1][0] # hệ số góc
x0 = np.linspace(145, 185, 2) # mảng 2 GT từ 145 đến 185
y0 = w_0 + w_1*x0

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)

# Compare two results
print( 'Solution found by scikit-learn  : ', regr.coef_ )
print( 'Solution found by (5): ', w.T)

# Drawing the fitting line 
plt.plot(X.T, y.T, 'ro')     # data 
plt.plot(x0, y0)               # the fitting line
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
