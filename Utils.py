import numpy as np

def getXmean(x_train):
    x_train = np.reshape(x_train, (x_train.shape[0], -1))  # 将28*28像素展开成一个一维的行向量
    mean_image = np.mean(x_train, axis=0)  # 求每一列均值。即求所有图片每一个像素上的平均值
    return mean_image

def centralized(x_test, mean_image):
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    x_test = x_test.astype(np.float)
    x_test -= mean_image  #减去平均值，实现均一化。
    return x_test

class Knn:
    def __init__(self):
        pass

    def fit(self, X_train, Y_train):
        self.Xtr = X_train
        self.Ytr = Y_train

    def predict(self, k, dis, X_test):
        assert dis == 'E' or dis == 'M','dis must E or M，E代表欧拉距离，M代表曼哈顿距离'
        num_test = X_test.shape[0]
        label_list = []
        # 使用欧拉公式作为距离测量
        if dis == 'E':
            for i in range(num_test):
                distances = np.sqrt(np.sum(((self.Xtr - np.tile(X_test[i],
                                                                (self.Xtr.shape[0], 1)))) ** 2, axis=1))
                nearest_k = np.argsort(distances)
                topK = nearest_k[:k]
                class_count = {}
                for i in topK:
                    class_count[self.Ytr[i]] = class_count.get(self.Ytr[i], 0) + 1
                sorted_class_count = sorted(class_count.items(), key=lambda elem: elem[1], reverse=True)
                label_list.append(sorted_class_count[0][0])
            return np.array(label_list)
        # 使用曼哈顿公式进行度量
        if dis == 'M':
            for i in range(num_test):
                distances = np.abs(np.sum(((self.Xtr - np.tile(X_test[i],
                                                                (self.Xtr.shape[0], 1)))), axis=1))
                nearest_k = np.argsort(distances)
                topK = nearest_k[:k]
                class_count = {}
                for i in topK:
                    class_count[self.Ytr[i]] = class_count.get(self.Ytr[i], 0) + 1
                sorted_class_count = sorted(class_count.items(), key=lambda elem: elem[1], reverse=True)
                label_list.append(sorted_class_count[0][0])
            return np.array(label_list)
