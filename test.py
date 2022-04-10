import torch  # 仅用于导入模型
from dataset.mnist import load_mnist
from train import simpleNet
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
# 导入模型
network = torch.load('net.pth')
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 输出分类精度
print(network.accuracy(x_test,t_test))

# 可视化网络参数
W1 = network.params['W1']
W2 = network.params['W2']

# PCA降维法
# pca = PCA(n_components=3)
# pca.fit(W1)
# W1 = pca.transform(W1)
# W1 = (W1-W1.min(axis=0))/(W1.max(axis=0)-W1.min(axis=0))
# W1 = W1.reshape(28,28,3)
# plt.imshow(W1)
# plt.show()
#
# pca.fit(W2)
# W2 = pca.transform(W2)
# W2 = (W2-W2.min(axis=0))/(W2.max(axis=0)-W2.min(axis=0))
# W2 = W2.reshape(20,20,3)
# plt.imshow(W2)
# plt.show()

#
# 归一化
W1 = (W1-W1.min(axis=0))/(W1.max(axis=0)-W1.min(axis=0))
W2 = (W2-W2.min(axis=0))/(W2.max(axis=0)-W2.min(axis=0))
for i in range(W1.shape[1]):
    D = W1[:,i]
    D = D.reshape(28,28)
    plt.subplot(20,20,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(D)

plt.show()
for i in range(25):
    D = W1[:,i]
    D = D.reshape(28,28)
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(D)

plt.show()

for i in range(W2.shape[1]):
    D = W2[:,i]
    D = D.reshape(20,20)
    plt.subplot(5,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(D)

plt.show()

