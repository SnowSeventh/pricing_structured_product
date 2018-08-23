# -*- coding: UTF-8 -*-
import numpy as np
import snowball as nt


input_list = [12, 21, 1, 1., 0.86, 0.14, 0.14, 0.1, 0.2, 100.]

W1 = np.loadtxt('/Users/wangpeng/tutu/htsc/product/snow/factor/weight1.txt')
W2 = np.loadtxt('/Users/wangpeng/tutu/htsc/product/snow/factor/weight2.txt')
b1 = np.loadtxt('/Users/wangpeng/tutu/htsc/product/snow/factor/bias1.txt')
b2 = np.loadtxt('/Users/wangpeng/tutu/htsc/product/snow/factor/bias2.txt')

'''
传统蒙特卡洛定价
'''
def monto(list):
    product = nt.SnowBall(list)
    price1 = product.pricing(list)
    return price1
price1 = monto(input_list)
'''
神经网络定价
'''
def nn(list):
    input_array = np.array(list).reshape(1,10)
    net1 = np.dot(input_array, W1)+b1
    out1 = np.where(net1<0, 0., net1)

    net2 = np.dot(out1, W2)+b2
    out2 = np.where(net2<0, 0., net2)
    price2 = out2[0]
    return price2
price2 = nn(input_list)

print '蒙特卡洛：'+ str(price1)
print '神经网：'+ str(price2)

