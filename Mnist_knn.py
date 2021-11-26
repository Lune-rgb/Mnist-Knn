import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from Utils import *
from pic_mask import *

batch_size = 100
path ='./'
train_datasets = datasets.MNIST(root=path,          #选择数据的根目录
                                train = True,       #选择训练集
                                transform = None,   #不考虑使用任何数据预处理
                                download = True)    # 从网络上download图片
test_datasets = datasets.MNIST(root=path,
                               train=False,
                               transform = None,    #不考虑使用任何数据预处理
                               download=True)

train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)  # 加载数据
test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

x_train = train_loader.dataset.data.numpy()         # 对训练数据处理
mean_image = getXmean(x_train)                      # 归一化处理
x_train = centralized(x_train, mean_image)
y_train = train_loader.dataset.targets.numpy()

num_test = 10000        # 对测试数据处理，取前num_test个测试数据
x_test = test_loader.dataset.data[:num_test].numpy()
mean_image = getXmean(x_test)
x_test = centralized(x_test, mean_image)
y_test = test_loader.dataset.targets[:num_test].numpy()


for i in range(60000):
    x_train[i] = random_noise(whether = True, image = x_train[i], noise_num = 100)  # random
    x_train[i] = sp_noise(whether = False, image = x_train[i], prob = 0.5)           # sp
    x_train[i] = gauss_noise(whether = False, image = x_train[i])                          # gauss
    x_train[i] = rec(whether = False, image = x_train[i], size = 5, top = 0, left = 0)    # mask

for i in range(10000):
    x_test[i] = random_noise(whether = True, image = x_test[i], noise_num = 100)  # random
    x_test[i] = sp_noise(whether = False, image = x_test[i], prob = 0.5)           # sp
    x_test[i] = gauss_noise(whether = False, image = x_test[i])                          # gauss
    x_test[i] = rec(whether = False, image = x_test[i], size = 5, top = 0, left = 0)    # mask

print("train_data:",x_train.shape)
print("train_label:",len(y_train))
print("test_data:",x_test.shape)
print("test_labels:",len(y_test))

# train_show = x_train[0].reshape(28, 28)
# pil_image=Image.fromarray(train_show)
# pil_image.show()

#利用KNN计算识别    
# for k in range(1, 6, 2): #不同K值计算识别率
#     start = time.time()
#     classifier = Knn()
#     classifier.fit(x_train, y_train)
#     y_pred = classifier.predict(k, 'E', x_test)     ##欧拉距离
#     num_correct = np.sum(y_pred == y_test)
#     accuracy = float(num_correct) / num_test
#     end = time.time()
#     duration = end - start
#     print('Got %d / %d correct when k= %d => accuracy: %f, duration = %f seconds' % (num_correct, num_test, k, accuracy, duration))

k = 3    
start = time.time()
classifier = Knn()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(k, 'E', x_test)     ##欧拉距离
num_correct = np.sum(y_pred == y_test)
accuracy = float(num_correct) / num_test
end = time.time()
duration = end - start
print('Got %d / %d correct when k= %d => accuracy: %f, duration = %f seconds' % (num_correct, num_test, k, accuracy, duration))

