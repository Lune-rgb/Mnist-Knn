# A KNN Aproach to Classify on MNIST Dataset

使用KNN分类MNIST数据集，可选使用deskewing，PCA和LDA降维，模型晃动（多种噪声和矩形蒙版）

最高正确率为：

**Accuracy = 0.976400, duration = 237.759806 seconds** 

控制参数为：K = 3，L = L2, 使用PCA降维，并只对标签为“0”的测试集图片进行偏斜校正。
