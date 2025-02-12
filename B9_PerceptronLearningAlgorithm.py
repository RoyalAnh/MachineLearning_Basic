# generate data
# list of points 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(2)

means = [[2, 2], [4, 2]] # Tọa độ trung tâm (mean) của hai cụm dữ liệu (class 1 và class -1)
cov = [[.3, .2], [.2, .3]] # Ma trận hiệp phương sai (covariance matrix) của mỗi cụm.
N = 10

#  Sinh các điểm dữ liệu ngẫu nhiên từ phân phối Gaussian đa biến 
# (Multivariate Normal Distribution) dựa trên trung tâm (means) và độ phân tán (cov).
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis = 1) # Kết hợp hai cụm dữ liệu lại với nhau dọc theo trục cột, tạo thành một ma trận 2x20.
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)

# Xbar 
X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)


def h(w, x):    
    return np.sign(np.dot(w.T, x)) # trả về 1 nếu kết quả nội tích dương, -1 nếu kết quả âm. Đây là nhãn dự đoán cho điểm dữ liệu x

def has_converged(X, y, w):
    
    return np.array_equal(h(w, X), y) #True if h(w, X) == y else False

def perceptron(X, y, w_init):
    w = [w_init]
    N = X.shape[1]
    mis_points = []
    while True:
        # mix data 
        mix_id = np.random.permutation(N) # Trộn ngẫu nhiên thứ tự của các điểm dữ liệu để giảm thiểu hiện tượng lặp 

        for i in range(N):
            xi = X[:, mix_id[i]].reshape(3, 1) # Lấy điểm dữ liệu xi tại vị trí ngẫu nhiên mix_id[i] từ tập dữ liệu X và định hình lại thành vector cột 3x1.
            yi = y[0, mix_id[i]] # Lấy nhãn thực tế yi của điểm dữ liệu này từ y.
            if h(w[-1], xi)[0] != yi:
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi*xi 

                w.append(w_new)
                
        if has_converged(X, y, w[-1]):
            break
    return (w, mis_points)

d = X.shape[0]
w_init = np.random.randn(d, 1)
(w, m) = perceptron(X, y, w_init)
print(m)
# print(w)
# print(len(w))

def draw_line(w):
    w0, w1, w2 = w[0], w[1], w[2]
    if w2 != 0: # giải phương trình để tính y theo x, vẽ đường thẳng bằng hai điểm x11, x12.
        x11, x12 = -100, 100
        return plt.plot([x11, x12], [-(w1*x11 + w0)/w2, -(w1*x12 + w0)/w2], 'k')
    else: # w2 = 0, đường phân loại là đường thẳng đứng tại x = -w0/w1
        x10 = -w0/w1
        return plt.plot([x10, x10], [-100, 100], 'k')


## Visualization : Sử dụng hình động
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation 
def viz_alg_1d_2(w):
    it = len(w)    
    fig, ax = plt.subplots(figsize=(5, 5))  
    
    def update(i):
        ani = plt.cla() # Xóa nội dung hiện tại trên biểu đồ để chuẩn bị vẽ lại.
        #points
        ani = plt.plot(X0[0, :], X0[1, :], 'b^', markersize = 8, alpha = .8)
        ani = plt.plot(X1[0, :], X1[1, :], 'ro', markersize = 8, alpha = .8)
        ani = plt.axis([0 , 6, -2, 4])
        i2 =  i if i < it else it-1
        ani = draw_line(w[i2])
        if i < it-1:
            # draw one  misclassified point
            circle = plt.Circle((X[1, m[i]], X[2, m[i]]), 0.15, color='k', fill = False)
            ax.add_artist(circle)

        # hide axis Nếu i chưa phải là bước cuối cùng, vẽ một vòng tròn quanh điểm bị phân loại sai để đánh dấu.
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_ticks([])
        cur_axes.axes.get_yaxis().set_ticks([])

        # hide axis
        label = 'PLA: iter %d/%d' %(i2, it-1)
        ax.set_xlabel(label)
        return ani, ax 
        
    #Hiển thị nhãn cho biết số lần lặp hiện tại.
    anim = FuncAnimation(fig, update, frames=np.arange(0, it + 2), interval=1000)

    anim.save('pla_vis.gif', dpi = 100, writer = 'imagemagick')
    plt.show()
    
viz_alg_1d_2(w)