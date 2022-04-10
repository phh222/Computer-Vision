import train
import numpy as np
from dataset.mnist import load_mnist
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# 参数查找
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
x_valid = x_train[:10000]
t_valid = t_train[:10000]
x_train = x_train[10000:60000]
t_train = t_train[10000:60000]

#查找学习率
# network = train.simpleNet(input_size=784, hidden_size=50, output_size=10)
# learning_rates = np.linspace(0.1,1.0,10) #学习率查找范围
#
# test_acc_l = []
# for i in range(len(learning_rates)):
#     dict = network.train(x_train=x_train, t_train=t_train, x_test=x_valid, t_test=t_valid, learning_rate=learning_rates[i])
#     test_acc_list = dict['test_acc_list']
#     test_acc_l.append(test_acc_list[len(test_acc_list)-1])
#     print(test_acc_list[len(test_acc_list)-1])
#
# plt.plot(learning_rates, test_acc_l)
# plt.xlabel("learning_rates")
# plt.ylabel("test accuracy")
# print(test_acc_l)
# print(learning_rates[np.argmax(test_acc_l)])
# plt.show()


#查找隐藏层大小
hiden_sizes = [200,250,300,350,400,450,500] # 查找范围
# hiden_sizes = [50,60,70,80,90,100,110,120,130,140,150]
test_acc_h = []
for i in range(len(hiden_sizes)):
    network = train.simpleNet(input_size=784, hidden_size=hiden_sizes[i], output_size=10,reg=0.001)
    dict = network.train(x_train=x_train, t_train=t_train, x_test=x_valid, t_test=t_valid,learning_rate=0.1)
    test_acc_list = dict['test_acc_list']
    print(test_acc_list[len(test_acc_list) - 1])
    test_acc_h.append(test_acc_list[len(test_acc_list) - 1])

plt.plot(hiden_sizes, test_acc_h)
plt.xlabel("hiden_sizes")
plt.ylabel("test accuracy")
plt.show()

print(hiden_sizes[np.argmax(test_acc_h)])

查找正则化
regs = np.linspace(0.0001,0.001,10) #查找范围
test_acc_r = []
for i in range(len(regs)):
    network = train.simpleNet(input_size=784, hidden_size=50, output_size=10, reg=regs[i])
    dict = network.train(x_train=x_train, t_train=t_train, x_test=x_valid, t_test=t_valid)
    test_acc_list = dict['test_acc_list']
    test_acc_r.append(test_acc_list[len(test_acc_list) - 1])
    print(test_acc_list[len(test_acc_list) - 1])

print(regs[np.argmax(test_acc_r)])
plt.plot(regs, test_acc_r)
plt.xlabel("regs")
plt.ylabel("test accuracy")
plt.show()