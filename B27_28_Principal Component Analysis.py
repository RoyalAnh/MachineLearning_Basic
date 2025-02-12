import numpy as np
import imageio.v2 as imageio  # sử dụng imageio.v2 để tránh cảnh báo DeprecationWarning
np.random.seed(1)

# filename structure
path = 'YALE/unpadded/'  # đường dẫn tới cơ sở dữ liệu
ids = range(1, 16)  # 15 người
states = ['centerlight', 'glasses', 'happy', 'leftlight',
          'noglasses', 'normal', 'rightlight', 'sad',
          'sleepy', 'surprised', 'wink']
prefix = 'subject'
surfix = '.pgm'

# kích thước dữ liệu
h = 116  # chiều cao
w = 98   # chiều rộng
D = h * w
N = len(states) * 15
K = 100

# thu thập tất cả dữ liệu
X = np.zeros((D, N))
cnt = 0
for person_id in range(1, 16):
    for state in states:
        fn = path + prefix + str(person_id).zfill(2) + '.' + state + surfix
        X[:, cnt] = imageio.imread(fn).reshape(D)  # sử dụng imageio.v2.imread
        cnt += 1

# Thực hiện PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=K)  # K = 100
pca.fit(X.T)

# ma trận projection
U = pca.components_.T

import matplotlib.pyplot as plt
for i in range(U.shape[1]):
    plt.axis('off')
    f1 = plt.imshow(U[:, i].reshape(116, 98), interpolation='nearest')
    f1.axes.get_xaxis().set_visible(False)
    f1.axes.get_yaxis().set_visible(False)
    plt.gray()
    fn = 'eigenface' + str(i).zfill(2) + '.png'
    plt.savefig(fn, bbox_inches='tight', pad_inches=0)

# Xem kết quả tái tạo của 6 người đầu tiên
for person_id in range(1, 7):
    for state in ['centerlight']:
        fn = path + prefix + str(person_id).zfill(2) + '.' + state + surfix
        im = imageio.imread(fn)  # thay thế misc.imread bằng imageio.imread
        plt.axis('off')
        f1 = plt.imshow(im, interpolation='nearest')
        f1.axes.get_xaxis().set_visible(False)
        f1.axes.get_yaxis().set_visible(False)
        plt.gray()
        fn = 'ori' + str(person_id).zfill(2) + '.png'
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)
        plt.show()

        # reshape và trừ mean
        x = im.reshape(D, 1) - pca.mean_.reshape(D, 1)

        # mã hóa (encode)
        z = U.T.dot(x)

        # giải mã (decode)
        x_tilde = U.dot(z) + pca.mean_.reshape(D, 1)

        # reshape lại về kích thước ban đầu
        im_tilde = x_tilde.reshape(116, 98)
        plt.axis('off')
        f1 = plt.imshow(im_tilde, interpolation='nearest')
        f1.axes.get_xaxis().set_visible(False)
        f1.axes.get_yaxis().set_visible(False)
        plt.gray()
        fn = 'res' + str(person_id).zfill(2) + '.png'
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)
        plt.show()

cnt = 0 
for person_id in [10]:
    for ii, state in enumerate(states):
        fn = path + prefix + str(person_id).zfill(2) + '.' + state + surfix
        im = imageio.imread(fn)
        f1 = plt.imshow(im, interpolation='nearest')
        f1.axes.get_xaxis().set_visible(False)
        f1.axes.get_yaxis().set_visible(False)

        fn = 'ex' + str(ii).zfill(2) +  '.png'
        plt.axis('off')
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)
         
        plt.show()
#         cnt += 1 