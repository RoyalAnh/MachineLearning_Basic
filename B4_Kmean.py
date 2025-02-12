from __future__ import print_function 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist #tính k/c các điểm trong 2 tập hợp  
#scipy: mở rộng của numpy, spatial.distance: tính k/c giữa các điểm
np.random.seed(11) # thiết lập GT bộ số sinh ngẫu nhiên

means = [[2, 2], [8, 3], [3, 6]] #các tâm của các cụm.
cov = [[1, 0], [0, 1]] # ma trận hiệp phương sai, định nghĩa độ phân tán của các cụm.
N = 500 #1 cluster có 500 điểm 
X0 = np.random.multivariate_normal(means[0], cov, N) #tạo ra dữ liệu từ phân phối chuẩn đa biến
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0) #dữ liệu tổng hợp từ ba cụm khác nhau.
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T #nhãn gốc của các điểm dữ liệu (0, 1, hoặc 2).

def kmeans_display(X, label): #hiển thị dữ liệu
    K = np.amax(label) + 1 #tìm giá trị lớn nhất trong mảng label, tương ứng với số cụm tối đa được sử dụng.
    X0 = X[label == 0, :] #chọn tất cả các điểm dữ liệu có nhãn 0 từ mảng X
    X1 = X[label == 1, :] 
    X2 = X[label == 2, :]
    
    # X0[:, 0] và X0[:, 1] là các giá trị của các điểm dữ liệu trên trục x và trục y.
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8) # alpha = .8 điều chỉnh độ trong suốt của điểm
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()
    
kmeans_display(X, original_label)

def kmeans_init_centers(X, k): #để khởi tạo các centers ban đầu.
    # randomly pick k rows of X as initial centers
    return X[np.random.choice(X.shape[0], k, replace=False)]

def kmeans_assign_labels(X, centers): #để gán nhán mới cho các điểm khi biết các centers.
    # calculate pairwise distances btw data and centers
    D = cdist(X, centers)
    # return index of the closest center
    return np.argmin(D, axis = 1)

def kmeans_update_centers(X, labels, K): #để cập nhật các centers mới dữa trên dữ liệu vừa được gán nhãn.
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        # collect all points assigned to the k-th cluster 
        Xk = X[labels == k, :]
        # take average
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers

def has_converged(centers, new_centers): #để kiểm tra điều kiện dừng của thuật toán.
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))

def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0 
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)

(centers, labels, it) = kmeans(X, K)
print('Centers found by our algorithm:')
print(centers[-1])

kmeans_display(X, labels[-1])

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(X)
kmeans_display(X, pred_label)