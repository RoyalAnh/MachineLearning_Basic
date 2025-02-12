# 1. Phân biệt giới tính
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model #  Cung cấp các mô hình hồi quy.
from sklearn.metrics import accuracy_score #  Cung cấp các mô hình hồi quy.
from scipy import misc      # Chứa các hàm hỗ trợ xử lý ảnh.
from sklearn import preprocessing # Các công cụ tiền xử lý dữ liệu.
np.random.seed(1)
path = '../data/AR/' # path to the database 

train_ids = np.arange(1, 26)
test_ids = np.arange(26, 50)
view_ids = np.hstack((np.arange(1, 8), np.arange(14, 21))) # ID của các góc nhìn khác nhau của ảnh.

D = 165*120 # original dimension Kích thước gốc của ảnh (165x120).
d = 500 # new dimension  Kích thước mới sau khi chiếu.

# generate the projection matrix : Ma trận chiếu ngẫu nhiên để giảm chiều dữ liệu.
ProjectionMatrix = np.random.randn(D, d) 

def build_list_fn(pre, img_ids, view_ids): # Hàm xây dựng danh sách các tên file ảnh
    """
    pre = 'M-' or 'W-'
    img_ids: indexes of images
    view_ids: indexes of views
    """
    list_fn = []
    for im_id in img_ids:
        for v_id in view_ids:
            fn = path + pre + str(im_id).zfill(3) + '-' + \
                str(v_id).zfill(2) + '.bmp'
            list_fn.append(fn)
    return list_fn 

def rgb2gray(rgb): # Hàm chuyển đổi ảnh từ RGB sang grayscale
#     Y' = 0.299 R + 0.587 G + 0.114 B 
    return rgb[:,:,0]*.299 + rgb[:, :, 1]*.587 + rgb[:, :, 2]*.114

# feature extraction  Hàm trích xuất đặc trưng từ ảnh
def vectorize_img(filename):    
    # load image 
    rgb = misc.imread(filename)
    # convert to gray scale 
    gray = rgb2gray(rgb)
    # vectorization each row is a data point 
    im_vec = gray.reshape(1, D)
    return im_vec 

def build_data_matrix(img_ids, view_ids): # Hàm xây dựng ma trận dữ liệu từ ảnh
    total_imgs = img_ids.shape[0]*view_ids.shape[0]*2 
        
    X_full = np.zeros((total_imgs, D))
    y = np.hstack((np.zeros((total_imgs/2, )), np.ones((total_imgs/2, ))))
    
    list_fn_m = build_list_fn('M-', img_ids, view_ids)
    list_fn_w = build_list_fn('W-', img_ids, view_ids)
    list_fn = list_fn_m + list_fn_w 
    
    for i in range(len(list_fn)):
        X_full[i, :] = vectorize_img(list_fn[i])

    X = np.dot(X_full, ProjectionMatrix)
    return (X, y)

# Tiền xử lý dữ liệu huấn luyện
(X_train_full, y_train) = build_data_matrix(train_ids, view_ids)
x_mean = X_train_full.mean(axis = 0)
x_var  = X_train_full.var(axis = 0)

def feature_extraction(X):
    return (X - x_mean)/x_var     

X_train = feature_extraction(X_train_full)
X_train_full = None ## free this variable 

# Tiền xử lý dữ liệu kiểm tra
(X_test_full, y_test) = build_data_matrix(test_ids, view_ids)
X_test = feature_extraction(X_test_full)
X_test_full = None 

# Huấn luyện mô hình hồi quy logistic và đánh giá mô hình
logreg = linear_model.LogisticRegression(C=1e5) # just a big number 
logreg.fit(X_train, y_train)


y_pred = logreg.predict(X_test)
print ("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred)))

def feature_extraction_fn(fn): # Dự đoán xác suất cho các ảnh mới
    im = vectorize_img(fn)
    im1 = np.dot(im, ProjectionMatrix)
    return feature_extraction(im1)

fn1 = path + 'M-036-18.bmp'
fn2 = path + 'W-045-01.bmp'
fn3 = path + 'M-048-01.bmp'
fn4 = path + 'W-027-02.bmp'

x1 = feature_extraction_fn(fn1)
p1 = logreg.predict_proba(x1)
print(p1)

x2 = feature_extraction_fn(fn2)
p2 = logreg.predict_proba(x2)
print(p2)

x3 = feature_extraction_fn(fn3)
p3 = logreg.predict_proba(x3)
print(p3)

x4 = feature_extraction_fn(fn4)
p4 = logreg.predict_proba(x4)
print(p4)

# Dự đoán và hiển thị kết quả:
def display_result(fn):
    x1 = feature_extraction_fn(fn)
    p1 = logreg.predict_proba(x1)
    print(logreg.predict_proba(x1))
    rgb = misc.imread(fn)
    
    
    fig = plt.figure()
#     gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
#     plt.subplot(1, 2, 1)
    plt.figure(facecolor="white")
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(rgb)
#     plt.axis('off')
#     plt.show()
    plt.subplot(122)
    plt.barh([0, 1], p1[0], align='center', alpha=0.9)
    plt.yticks([0, 1], ('man', 'woman'))
    plt.xlim([0,1])
    plt.show()
    
    
   
    # load an img 
fn1 = path + 'M-036-18.bmp'
fn2 = path + 'W-045-01.bmp'
fn3 = path + 'M-048-01.bmp'
fn4 = path + 'W-027-02.bmp'
display_result(fn1)
display_result(fn2)
display_result(fn3)
display_result(fn4)

# 2. Bài toán phân biệt hai chữ số viết tay
import numpy as np
import matplotlib.pyplot as plt
from mnist.loader import MNIST
from sklearn import linear_model
from sklearn.metrics import accuracy_score

# Load data
mntrain = MNIST('library/MNIST\\raw')
mntrain.load_training()
Xtrain_all = np.asarray(mntrain.train_images)
ytrain_all = np.array(mntrain.train_labels.tolist())

mntest = MNIST('library/MNIST\\raw')
mntest.load_testing()
Xtest_all = np.asarray(mntest.test_images)
ytest_all = np.array(mntest.test_labels.tolist())

cls = [[0], [1]]

def extract_data(X, y, classes):
    y_res_id = np.concatenate([np.where(y == i)[0] for i in classes[0]])
    n0 = len(y_res_id)
    y_res_id = np.concatenate([y_res_id, np.where(np.isin(y, classes[1]))[0]])
    n1 = len(y_res_id) - n0 

    X_res = X[y_res_id] / 255.0
    y_res = np.array([0] * n0 + [1] * (len(y_res_id) - n0))
    return X_res, y_res

# Extract data for training and testing
X_train, y_train = extract_data(Xtrain_all, ytrain_all, cls)
X_test, y_test = extract_data(Xtest_all, ytest_all, cls)

# Train the logistic regression model
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train, y_train)

# Predict and evaluate
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f} %")

# Display misclassified images
mis = np.where(y_pred != y_test)[0]
if len(mis) > 0:
    Xmis = X_test[mis]
    
    # Plot misclassified images
    num_images = min(100, len(mis))  # Limit number of images to display
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(10, 10, i + 1)
        plt.imshow(Xmis[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.show()
else:
    print("No misclassified images to display.")
