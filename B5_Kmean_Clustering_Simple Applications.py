import numpy as np 
from mnist.loader import MNIST
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans #Thuật toán phân cụm K-means từ thư viện scikit-learn.
from sklearn.preprocessing import normalize # Hàm để chuẩn hóa dữ liệu 
import imageio # Thư viện để đọc và ghi hình ảnh 

img = mpimg.imread('ima.png') # imread trả về một mảng numpy chứa các giá trị pixel của hình ảnh
plt.imshow(img)
imgplot = plt.imshow(img) 
plt.axis('off') # Tắt các trục của đồ thị để chỉ hiển thị hình ảnh mà không có thông tin trục.
plt.show() 

# Biến đổi bức ảnh thành 1 ma trận mà mỗi hàng là 1 pixel với 3 giá trị màu ( trắng , xanh, da)
X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))

for K in [5, 10, 15, 20]: # mỗi giá trị K tương ứng với số lượng cụm mà thuật toán K-means sẽ tìm kiếm
    kmeans = KMeans(n_clusters=K).fit(X) # Khởi tạo và huấn luyện mô hình K-means với K cụm.
    label = kmeans.predict(X)

    img4 = np.zeros_like(X) # lưu trữ các giá trị màu của các cụm.

    # replace each pixel by its center
    for k in range(K):
        img4[label == k] = kmeans.cluster_centers_[k] 
        # Thay thế giá trị màu của các pixel thuộc cụm k bằng giá trị màu trung tâm của cụm đó.
    # reshape and display output image
    img5 = img4.reshape((img.shape[0],img.shape[1], img.shape[2])) # Chuyển đổi ma trận img4 về kích thước của hình ảnh gốc.
    plt.imshow(img5, interpolation='nearest') # Hiển thị hình ảnh sau khi phân cụm với K màu.
    plt.axis('off')
    plt.show()