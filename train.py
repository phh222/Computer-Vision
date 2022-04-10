import numpy as np
from dataset.mnist import load_mnist
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
matplotlib.use('TkAgg')
import math
import torch # 仅用于保存模型
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 激活函数
def Relu(x):
    """
    Relu 函数
    :param x:
    :return:max(0,x)
    """
    return np.maximum(0, x)

def sigmoid(x):
    return 1/(1+math.exp(-x))

class simpleNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01,reg=0.0003):
        """

        :param input_size: 图像尺寸输入维度 D
        :param hidden_size: 隐藏层维度 H
        :param output_size: 类别 C
        """
        # 初始化网络
        self.params = {}
        # weight_init_std:权重初始化标准差
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
                            # 用高斯分布随机初始化一个权重参数矩阵
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.reg = reg


    def predict(self, x):
        # 前向传播，用点乘实现
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        z = Relu(np.dot(x,W1)+b1)
        y = np.dot(z,W2)+b2

        return y


    def loss(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        y = self.predict(x)
        data_loss = 0.5*np.sum((y-t)**2)/y.shape[0]
        reg_loss = 0.5*self.reg*(np.sum(W1*W1)+np.sum(W2*W2))
        loss = data_loss + reg_loss

        return loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum( y==t ) / float(x.shape[0])
        return accuracy

    # 反向传播计算梯度
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        z = Relu(np.dot(x, W1) + b1)
        y = np.dot(z, W2) + b2

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z.T, dy)+self.reg*W2
        grads['b2'] = np.sum(dy, axis=0)


        dz = np.dot(dy, W2.T)
        # Relu 层
        dz[z <= 0] = 0
        grads['W1'] = np.dot(x.T, dz) + self.reg*W1
        grads['b1'] = np.sum(dz, axis=0)

        return grads

    def learning_rate_decay(self,learning_rate,decay=0.95):
        """
        步长衰减，每一个epoch过后，学习率乘衰减因子
        :param learning_rate:
        :param decay: 衰减因子
        """
        return(decay*learning_rate)

    def train(self, x_train, t_train, x_test, t_test, learning_rate=0.4, batch_size=100, iters_num=10000):
        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []
        train_size = x_train.shape[0]
        iter_per_epoch = max(train_size / batch_size, 1)

        for i in range(iters_num):
            # 获取mini-batch
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            # 计算梯度
            grad = self.gradient(x_batch, t_batch)

            # 更新参数
            for key in ('W1', 'b1', 'W2', 'b2'):
                self.params[key] -= learning_rate * grad[key]

            # 记录学习过程的损失变化
            train_loss = self.loss(x_batch, t_batch)
            train_loss_list.append(train_loss)


            if i % iter_per_epoch == 0:
                train_acc = self.accuracy(x_train, t_train)
                test_acc = self.accuracy(x_test, t_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                test_loss = self.loss(x_test, t_test)
                test_loss_list.append(test_loss)
                learning_rate = self.learning_rate_decay(learning_rate)
                # print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

        return{
            'train_loss_list':train_loss_list,
            'test_loss_list':test_loss_list,
            'train_acc_list':train_acc_list,
            'test_acc_list':test_acc_list
        }
#
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# x_valid = x_train[:10000]
# t_valid = t_train[:10000]
# x_train = x_train[10000:60000]
# t_train = t_train[10000:60000]
#
# # 使用参数查找后的结果
# network = simpleNet(input_size=784, hidden_size=400, output_size=10,reg=0.0003)
# # 训练
# dict = network.train(x_train= x_train, t_train= t_train, x_test= x_valid, t_test=t_valid,learning_rate=0.4)
#
# # 保存模型
# torch.save(network, 'net.pth')
#
# train_loss_list = dict['train_loss_list']
# test_loss_list = dict['test_loss_list']
# train_acc_list = dict['train_acc_list']
# test_acc_list = dict['test_acc_list']
#
#
# # 画损失函数的变化
# x1 = np.arange(len(train_loss_list))
# ax1 = plt.subplot(221)
# plt.plot(x1, train_loss_list)
# plt.xlabel("iteration")
# plt.ylabel("train_loss")
#
# x3 = np.arange(len(test_loss_list))
# ax3 = plt.subplot(223)
# plt.plot(x3, test_loss_list)
# plt.xlabel("epchos")
# plt.ylabel("test_loss")
#
#
# # 画训练精度，测试精度随着epoch的变化
# markers = {'train': 'o', 'test': 's'}
# x2 = np.arange(len(train_acc_list))
# ax2 = plt.subplot(222)
# plt.plot(x2, train_acc_list, label='train acc')
# plt.plot(x2, test_acc_list, label='test acc', linestyle='--')
# plt.xlabel("epochs")
# plt.ylabel("accuracy")
# plt.ylim(0, 1.0)
# plt.legend(loc='lower right')
# plt.show()
#
