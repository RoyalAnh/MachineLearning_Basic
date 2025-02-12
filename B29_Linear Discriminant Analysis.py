# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
# list of points
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
np.random.seed(22)

means = [[0, 5], [5, 0]]
cov0 = [[4, 3], [3, 4]]
cov1 = [[3, 1], [1, 1]]
N0 = 50
N1 = 40
N = N0 + N1
X0 = np.random.multivariate_normal(means[0], cov0, N0) # each row is a data point
X1 = np.random.multivariate_normal(means[1], cov1, N1)

# tính các within-class và between-class covariance matrices
# Build S_B
m0 = np.mean(X0.T, axis = 1, keepdims = True)
m1 = np.mean(X1.T, axis = 1, keepdims = True)

a = (m0 - m1)
S_B = a.dot(a.T)

# Build S_W
A = X0.T - np.tile(m0, (1, N0)) # Ma trận chứa sự chênh lệch của các điểm trong X0 so với vector trung bình m0
B = X1.T - np.tile(m1, (1, N1))

S_W = A.dot(A.T) + B.dot(B.T)

_, W = np.linalg.eig(np.linalg.inv(S_W).dot(S_B)) # Tính các trị riêng (eigenvalues) và vector riêng (eigenvectors) của ma trận
w = W[:,0] # Chọn vector riêng tương ứng với trị riêng lớn nhất (vector phân biệt chính)
print(w)

# kiểm chứng độ chính xác của nghiệm tìm được bằng LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
X = np.concatenate((X0, X1))
y = np.array([0]*N0 + [1]*N1) # Tạo nhãn cho hai tập dữ liệu, 0 cho các điểm từ X0 và 1 cho các điểm từ X1
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)

print(clf.coef_/np.linalg.norm(clf.coef_)) # normalize : Trả về vector hệ số của mô hình LDA