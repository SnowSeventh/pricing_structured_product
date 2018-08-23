# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import time as ti

'''
合约类参数
key_list = [duration, knock_out_freq, knock_in_freq, knock_out_barrier, knock_in_barrier, knock_out_rate, bonus，u，vol, spot]
duration: 产品合约期，单位按月，整型；
knock_out_freq: 敲出观察频率，单位按日，整型；
knock_in_freq: 敲入观察频率，单位按日，整型；
knock_out_barrier: 敲出界限，浮点型；
knock_in_barrier: 敲入界限，浮点型；
knock_out_rate: 敲出票息，浮点型；
bonus: 红利票息，浮点型；
u: 期望收益率，浮点型;
vol：波动率，浮点型;
spot: 标的物初期价格；
特别定义：当以上十个产品参数需要以不定态的变量形式输入时，相应的位置以字符串‘variant’表示，
程序通过查找该字符串的位置和数量决定随机生成的产品;
注意：十个参量输入的格式和顺序一定，不可更改。
注意交易日和自然日的区别
'''

'''
注：以上十个参数支持自定义随机生成，但都需满足一定的限制条件：
1. duration： 产品的续存期按月为单位，自定义范围必须与敲出观察频率和敲入观察频率自洽；
2. 敲出观察频率和敲入观察频率支持日级别以上的观察，单位一律按天，而且是真实交易日；
3. 两个障碍原则上应确保上轨高于下轨，切意义合理，不严重脱离实际；
4. 敲出票息和红利票息以及标的初期价格也应处在合理区间；
5. 上述十个参量的范围这里不做具体限制。
6. 这些变量中最难控制的是duration变量，一旦续存期为随机变量，那么训练集和测试集中的产品含有完全不同的期限，
就不能用统一的方式来定价，即必须以循环的方式来逐个求解；
7. 本次的事例中，只改变spot这一个参量；
8. 在新加入两个变量后，十个变量都要完全随机，生成大样本空间；
'''

'''
神经网络类参数
由于不同产品之间没有时间上的相关性，所以不适用于LSTM；同样的，由于样本数量不大，特征不多，所以用不上CNN
综上，这次只采用最简单的三层神经网络
input layer含有产品的十个参数;
hiden layer暂定含有十二个元素；
output layer则只含有一个输出元素来表示神经网络的定价结果；
所以，W1为（10*12）的矩阵，b1为（1*12）的矩阵
W2为（12*1）的矩阵，b2位（1*1）的矩阵
激活函数可以选择sigmoid或者relu
'''

'''
最新更新：
1. 添加两个波动参数，波动率vol和期望收益率u，分别在8号位和7号位，spot依然在最后一位；
2. 在做了改动之后，产品的输入结构以及神经网络的输入层都要做相应的变动，甚至神经网络的隐藏层也要做修改；
3. 具体而言，生成curve的函数，判断敲入和敲出的函数以及神经网络的设定都要变；
'''

'''
2018.7.20日更新：
1. 产品结构需要做变化了，之前的十个输入参数的设计忽略了日期和日历的概念，实际中的定价是需要考虑真实日期和节假日的，所以，这次修改产品结构，
    也就是产品的输入组成：将duration换成两个日期，start date和end date；
2. 加入pricing date的概念，之前的设计只考虑合约签署也就是合约生成日的价格和其他的性质，忽略了合同期内合约价格的变化和对其他性质的观测，这问题
    在之后计算希腊值的过程中彻底暴露了，所以必须加；
3. 还有一点非常重要的就是要加入日历的概念：按照quantlib的理念，日历作为一个地区的交易时间安排，是定价的基础，这次为了使定价的工具模型更具使用价值，
    尽量要写一个单独的日历模块，加入定价过程中；
4. 单独写一个很简单的贴现模块，这东西几乎处处通用；
5. 这次的工作包括以后为其他产品写定价模块都是一样的道理，产品部分负责定价，既包括传统的定价方式，也包括另外一部分更重要的神经网络定价，也可以说，
    传统定价更多的是为训练生成训练集；
6. 希腊值的算法按照当前目录中的图片来做，通过改变相应的变量来实现；在生成训练集的过程中，由于算力有限，所以锁定其他所有参数，只更改相应的希腊值自变量，
    以获得足够的训练样本；
7. 给雪球产品计算希腊值的部分功能最好也写在雪球产品同一模块中，以后其他的产品同理；
8. 这次需要用两种不同的神经网络来测试两种不同的希腊值计算顺序：看看是通过训练得到的价格来计算希腊值更好还是通过训练直接得到的希腊值更好，标准时蒙特卡洛的
    计算结果；
'''

class SnowBall:

    duration = None
    knock_out_freq = None
    knock_in_freq = None
    knock_out_barrier = None
    knock_in_barrier = None
    knock_out_rate = None
    bonus = None
    spot = None
    std_spot = 100.
    argument = 100.
    u = None # 注：这个参数后期应该需要波动；
    sigma = None # 同上
    curve_num = 50000
    r = 0.1 # 无风险利率也可能需要波动生成，待定；更新，在计算Vega的过程中，改变的利率既包括期望收益率μ，也包括无风险收益率r，所以都得设成随机变量；
    bussiness_day = 252
    nature_day = 360
    bussiness_month = 21
    nature_month = 30
    month_num = 12
    # 以上五个关于日期的参数理论上应该由日历模块完美的替代，只要将日历传入，那么在相应的产品起始日和合约期内，产品的以上几个相关参数都是确定的。
    stdi = np.float(np.sqrt(1./bussiness_day))# 模拟标的物价格曲线的标准差以日为波动时间间隔是没有问题的，但应该和bussinessday脱离；
    aver = 0.
    tolerance = 0.001


    session = None
    X_train_holder = None
    Y_train_holder = None
    Y_ = None
    cost = None
    optimizer = None
    w1 = None
    w2 = None
    b1 = None
    b2 = None
    ratio = 0.95

    learning_rate = 0.01
    input_num = 1000
    input_dimension = 10 #注：后期可能再加入新的波动参数输入，比如重要的波动率，所以维度一定不够；
    output_dimension = 1
    hiden_layer_dimension = 12 #随着input layer维度的增加，隐藏层的维度也可能需要增加;
    iteration = 3000 #训练次数随机调整；
    accuracy = None

    def __init__(self, key_list):
        self.duration = key_list[0]
        self.knock_out_freq = key_list[1]
        self.knock_in_freq = key_list[2]
        self.knock_out_barrier = key_list[3]
        self.knock_in_barrier = key_list[4]
        self.knock_out_rate = key_list[5]
        self.bonus = key_list[6]
        self.u = key_list[7]
        self.sigma = key_list[8]
        self.spot = key_list[9]


        # build neural network
        self.create_network()

        # tensorflow
        self.session = tf.Session()

        # self.session.run(tf.global_variables_initializer())


    def create_network(self):
        self.X_train_holder = tf.placeholder(shape=[int(self.input_num*self.ratio), self.input_dimension], dtype= tf.float32)
        self.Y_train_holder = tf.placeholder(shape=[int(self.input_num*self.ratio), self.output_dimension], dtype= tf.float32)

        self.w1 = tf.random_uniform(shape=[self.input_dimension, self.hiden_layer_dimension], minval=-0.05
        ,maxval=0.05, dtype=tf.float32, name='weight1')
        self.b1 = tf.Variable(tf.zeros([1, self.hiden_layer_dimension]), dtype= tf.float32)
        l1 = tf.nn.relu(tf.matmul(self.X_train_holder, self.w1) + self.b1)

        self.w2 = tf.random_uniform(shape=[self.hiden_layer_dimension, self.output_dimension], minval=-0.05
        ,maxval=0.05, dtype=tf.float32, name='weight2')
        self.b2 = tf.Variable(tf.zeros([1, self.output_dimension])+ 0.1, dtype= tf.float32)
        self.Y_ = tf.nn.relu(tf.matmul(l1, self.w2) + self.b2)

        self.cost = tf.reduce_mean(tf.square(self.Y_train_holder - self.Y_)) / 2.
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

    def book(self):

        '''
        产品续存期，取值范围12-24个月，随机取值取整数，可重复；
        :return:
        '''
        if self.duration is 'variant':
            c1 = np.random.random_integers(12,24,[self.input_num*1])
        else:
            c1 = np.ones((self.input_num,),dtype=np.int)*self.duration
        '''
        产品敲入和敲出观察频率，按天为单位，取值范围1-252天，随机取整数，可重复；
        '''
        if self.knock_out_freq is 'variant':
            c2 = np.random.random_integers(1,252,[self.input_num*1])
        else:
            c2 = np.ones((self.input_num,),dtype=np.int)*self.knock_out_freq

        if self.knock_in_freq is 'variant':
            c3 = np.random.random_integers(1,252,[self.input_num*1])
        else:
            c3 = np.ones((self.input_num,),dtype=np.int)*self.knock_in_freq

        '''
        两个barrier一上一下，upper barrier是0.9-1.2之间的float，随机取值；lower barrier是0.7-0.9之间的float，随机取值，并且上下界都
        可取到0和10，为了达到这种效果，利用了random函数的边缘特征，当随机变量取到区间左边界的时候，自动替换成0或10；
        '''
        if self.knock_out_barrier is 'variant':

            c4 = np.random.uniform(0.9, 1.2, [self.input_num*1])
            c4 = np.where(c4 == 0.9, np.random.choice([0., 10.]), c4)

        else:
            c4 = np.ones((self.input_num,))*self.knock_out_barrier

        if self.knock_in_barrier is 'variant':
            c5 = np.random.uniform(0.7, 0.9, [self.input_num*1])
            c5 = np.where(c5 == 0.7, np.random.choice([0., 10.]), c5)

        else:
            c5 = np.ones((self.input_num,))*self.knock_in_barrier

        '''
        敲出票息和红利都取到0-0.2；
        '''
        if self.knock_out_rate is 'variant':
            c6 = np.random.uniform(0,0.2,[self.input_num*1])
        else:
            c6 = np.ones((self.input_num,))*self.knock_out_rate

        if self.bonus is 'variant':
            c7 = np.random.uniform(0,0.2,[self.input_num*1])
        else:
            c7 = np.ones((self.input_num,))*self.bonus

        '''
        波动率和期望收益率都取到0-0.3；
        '''
        if self.u is 'variant':
            c8 = np.random.uniform(0., 0.3, [self.input_num*1])
        else:
            c8 = np.ones((self.input_num,))*self.u

        if self.sigma is 'variant':
            c9 = np.random.uniform(0., 0.3, [self.input_num*1])
        else:
            c9 = np.ones((self.input_num,))*self.sigma

        '''
        spot价格以100为基准，上下浮动5*std；
        
        '''
        if self.spot is 'variant':
            c10 = np.random.uniform(self.std_spot-5*self.stdi,self.std_spot+5*self.stdi,[self.input_num*1])
        else:
            c10 = np.ones((self.input_num,))*self.spot

        benchmark = np.c_[c1,c2,c3,c4,c5,c6,c7,c8,c9,c10]
        return  benchmark

    '''
    注：蒙特卡洛模拟的单次价格曲线随机生成，最小时间间隔为日
    '''

    def underlying_curve(self, s0, t, curve_number, mu, vol):
        first = np.ones(curve_number)*s0
        length = np.int(t*self.bussiness_month)
        Wt = np.random.normal(self.aver, self.stdi, [curve_number, length-1])
        curve = s0*np.exp(vol*Wt+(mu-vol**2/2)/self.bussiness_day).cumprod(1)
        curve = np.c_[first, curve]
        return curve

    def knock_out(self, upper_barrier, s0, frequency, underlying_curve, t):

        for i in range(np.int(self.bussiness_month * t / frequency)):

            index = int((i+1)*frequency)
            if underlying_curve[index-1]> s0*upper_barrier:
                return i+1
        return False

    def knock_in(self, lower_barrier, s0, frequency, underlying_curve, t):

        for i in range(int(self.bussiness_month * t / frequency)):
            index = int((i+1)*frequency)
            if underlying_curve[index-1]< s0*lower_barrier:
                return True
        return False

    def price_of_one_curve(self, one_product, one_curve):

        num = self.knock_out(one_product[3], self.std_spot, one_product[1], one_curve, one_product[0])
        if num:
            expire = num*self.nature_month/float(self.nature_day)
            discount_factor = np.exp(-self.r*expire)
            return self.argument * (1 + one_product[5] * expire)*discount_factor
        else:
            if self.knock_in(one_product[4], self.std_spot, one_product[2], one_curve, one_product[0]):
                receive = min(one_curve[-1] / self.std_spot, 1.)
                expire = one_product[0]/float(self.month_num)
                discount_factor = np.exp(-self.r*expire)
                return self.argument * receive * expire *discount_factor
            else:
                expire = one_product[0]/float(self.month_num)
                discount_factor = np.exp(-self.r*expire)
                return self.argument * (1 + one_product[6] * expire)*discount_factor

    def pricing(self, one_product):

        price = 0.
        curve = self.underlying_curve(one_product[-1], one_product[0], self.curve_num, one_product[7], one_product[8])
        for i in curve:

            price += self.price_of_one_curve(one_product, i)
        '''
                    one_curve = self.underlying_curve(one_product[7], one_product[0])
                    flag = self.knock_out(one_product[3], one_product[-1], one_product[1], one_curve, one_product[0])
                    if flag:
                        holder.append(self.argument*(1+one_product[5]*flag/360.)*np.exp(-self.r*flag/360.))
                    else:
                        if self.knock_in(one_product[4], one_product[-1], one_product[2], one_curve, one_product[0]):
                            receive = min(one_curve[-1]/one_product[-1], 1.)
                            holder.append(self.argument*receive*one_product[0]/360.*np.exp(-self.r*one_product[0]/360.))
                        else:
                            holder.append(self.argument*(1+one_product[6]*one_product[0]/360.)*np.exp(-self.r*one_product[0]/360.))
        '''
        return price/self.curve_num

    def label(self, bench):

        holder = []
        clock = 1

        time_total = 0.

        for i in bench:
            start = ti.time()

            holder.append(self.pricing(i))

            end = ti.time()
            consum = end - start
            time_total += consum

            if clock%100 is 0:
                print time_total
            clock += 1

            # print str(consum)+'s'
        print 'average time consumption: '+str(time_total/self.input_num)
        return np.array(holder)

    def cal_accuracy(self, list, value):

        length = int(self.input_num*(1-self.ratio))
        count = 0.
        for i in list:
            if i < value:
                count+= 1.
        return count/length

    @property
    def train_test(self):

        benchmarks = self.book()

        labels = self.label(benchmarks)

        train_set = benchmarks[0:int(self.input_num*self.ratio)]
        test_set = benchmarks[int(self.input_num*self.ratio):self.input_num]

        y_train = labels[0:int(self.input_num*self.ratio)].reshape([int(self.input_num*self.ratio),1])
        y_test = labels[int(self.input_num*self.ratio):self.input_num].reshape(int(self.input_num-int(self.input_num*self.ratio)),1)

        self.session.run(tf.global_variables_initializer())


        for rand in range(self.iteration):

            self.session.run(self.optimizer,feed_dict={self.X_train_holder: train_set, self.Y_train_holder: y_train})

        weight1, weight2, bias1, bias2 = self.session.run([self.w1, self.w2, self.b1, self.b2])

        hiden_layer = tf.nn.relu(tf.matmul(tf.cast(test_set, tf.float32), self.w1)+ self.b1)

        y_predict = tf.nn.relu(tf.matmul(hiden_layer, self.w2)+ self.b2)

        count = tf.divide(np.abs(y_predict-y_test),y_test)

        mat = count - self.tolerance
        result = self.session.run(mat)

        '''correct_prediction = tf.approximate_equal(y_predict, y_test, 0.001)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.accuracy = self.cal_accuracy(count, 0.001)'''

        return weight1, weight2, bias1, bias2, result.reshape(10,5)

if  __name__ == "__main__":

    print 'snowball.py is being called directly'

    factor_list = ['variant', 21, 1, 1., 0.86, 0.14, 0.14, 'variant', 'variant', 'variant']

    sample = SnowBall(factor_list)

    sample.create_network()

    a, b, c, d, accuracy = sample.train_test

    np.savetxt('/Users/wangpeng/tutu/htsc/product/snow/factor/weight1.txt', a)
    np.savetxt('/Users/wangpeng/tutu/htsc/product/snow/factor/weight2.txt', b)
    np.savetxt('/Users/wangpeng/tutu/htsc/product/snow/factor/bias1.txt', c)
    np.savetxt('/Users/wangpeng/tutu/htsc/product/snow/factor/bias2.txt', d)


    print 'weight1:', a
    print 'weihtt2:', b
    print 'bias1:', c
    print 'bias2:', d
    print 'accuracy matrix:', accuracy

else:

    print 'snowball.py is being imported from another file'
































