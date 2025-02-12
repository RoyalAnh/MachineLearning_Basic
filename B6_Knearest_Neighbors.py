import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

iris = datasets.load_iris()
iris_X = iris.data # Các đặc trưng của từng mẫu trong tập dữ liệu.
iris_y = iris.target # Nhãn lớp (loại hoa) của từng mẫu.

X0 = iris_X[iris_y == 0,:] #  Lấy tất cả các đặc trưng của các mẫu thuộc lớp 0

print ('Number of classes: %d' %len(np.unique(iris_y))) # Đếm số lượng lớp (nên là 3).
print ('\nSamples from class 0:\n', X0[:5,:])

X1 = iris_X[iris_y == 1,:]
print ('\nSamples from class 1:\n', X1[:5,:])

X2 = iris_X[iris_y == 2,:]
print ('\nSamples from class 2:\n', X2[:5,:])

# dùng 50 điểm dữ liệu cho test set, 100 điểm còn lại cho training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)

print ("Training size: %d" %len(y_train))
print ("Test size    : %d" %len(y_test))

# K = 1, tức là với mỗi điểm test data, ta chỉ xét 1 điểm training data gần nhất lấy label của điểm đó để dự đoán cho điểm test này.
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2, weights = 'distance') # K = n_neighbor
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print ("Print results for 20 test data points:")
print ("Predicted labels: ", y_pred[20:40])
print ("Ground truth    : ", y_test[20:40])

from sklearn.metrics import accuracy_score
print ("Accuracy of KNN(1/distance weights): %.2f %%" %(100*accuracy_score(y_test, y_pred)))

def myweight(distances):
    sigma2 = .5 # we can change this number
    return np.exp(-distances**2/sigma2)

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = myweight)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print ("Accuracy of 10NN (customized weights): %.2f %%" %(100*accuracy_score(y_test, y_pred)))

# %reset
import numpy as np 
from mnist.loader import MNIST # require `pip install python-mnist`
# https://pypi.python.org/pypi/python-mnist/

import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import time

# you need to download the MNIST dataset first
# at: http://yann.lecun.com/exdb/mnist/
mndata = MNIST('library/MNIST\\raw') 
mndata.load_testing()
mndata.load_training()
X_test = mndata.test_images
X_train = mndata.train_images
y_test = np.asarray(mndata.test_labels)
y_train = np.asarray(mndata.train_labels)


start_time = time.time()
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
end_time = time.time()
print ("Accuracy of 1NN for MNIST: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
print ("Running time: %.2f (s)" % (end_time - start_time))