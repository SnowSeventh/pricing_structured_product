# -*- coding: UTF-8 -*-
import tensorflow as tf
from data_vector import vector
import time as ti
import numpy as np
from Auto_Call.auto_call import autocall
from Bussiness_Calender.Bcalender import BussinessCalender

class recursive:

    # 神经网络基本参数
    iteration = 100

    # 标准合约及其价格
    info= {
            'start': '2017/11/9',
            'end': '2018/11/15',
            'pricing_date': '2017/11/9', # 可变
            'spot': 100.,  # 可变
            'knockoutdays': [['2017/12/14'],['2018/1/11'],['2018/2/8'],['2018/3/8'],['2018/4/12'],['2018/5/10'],['2018/6/14'],
                             ['2018/7/12'],['2018/8/9'],['2018/9/13'],['2018/10/11']],
            'knockindays': 'all',
            'upper_barrier': 1.,
            'lower_barrier': 0.86,
            'interest_rate': 0.14,
            'knockoutrate': 0.14,
            'mu': -0.12, # 可变
            'vol': 0.15, # 可变
            'r': 0.
    }

    test = autocall(BussinessCalender(), info, argument=1)
    std_price = test.pricing(info)

    # 测试集准备
    db = vector(collection_name='new_record', amount= 1423)
    def fetch_train_set(self):
        start = ti.time()
        train_set = [self.db.vectorization(item) for item in self.db.fetch_data()]
        end = ti.time()
        consume = str(end - start) + 's'
        print '数据预处理结束, 耗时'+consume
        return train_set

    # learning rate
    global_step = tf.Variable(0, trainable=False)
    init_learning_rate = 0.01
    learning_rate = tf.train.exponential_decay(init_learning_rate,
                                               global_step=global_step,
                                               decay_steps=100,
                                               decay_rate=0.95)
    add_global = global_step.assign_add(1)


    # 建立网络参数容器

    with tf.name_scope('para_hiden_layer'):
        weight = tf.Variable(tf.random_uniform([3,3], -1., 1.), name='weight')
        bias = tf.Variable(tf.zeros([]), name='bias')
    with tf.name_scope('rnn'):
        u1 = tf.Variable(tf.random_uniform([6,3], -1., 1.), name='u1')
        b1 = tf.Variable(tf.zeros([]), name='b1')

    # 建立输入参数容器
    with tf.name_scope('Price'):
        price = tf.placeholder(tf.float32, name='price')
    with tf.name_scope('Spot'):
        spot = tf.placeholder(tf.float32, [1, 3], name='spot')
    with tf.name_scope('Pricing_date'):
        pricing_date = tf.placeholder(tf.float32, [1, 3], name='pricing_date' )
    with tf.name_scope('Mu'):
        mu = tf.placeholder(tf.float32, [1, 3], name='mu')
    with tf.name_scope('Vol'):
        vol = tf.placeholder(tf.float32, [1, 3], name='vol')

    # 传播函数
    def forward(self, input, weight, bias):
        out = tf.nn.tanh(tf.matmul(input, weight) + bias)
        return out
    # 将所有tensor按计算关系连接成flow
    def flow(self):
        with tf.name_scope('spot_out'):
            spot_out = self.forward(self.spot, self.weight, self.bias)
        with tf.name_scope('pricing_date_out'):
            pricing_date_out = self.forward(self.pricing_date, self.weight, self.bias)
        with tf.name_scope('mu_out'):
            mu_out = self.forward(self.mu, self.weight, self.bias)
        with tf.name_scope('vol_out'):
            vol_out = self.forward(self.vol, self.weight, self.bias)
        with tf.name_scope('node1_input'):
            node1_input = tf.concat([spot_out, mu_out], 1)
        with tf.name_scope('Node1'):
            node1 = self.forward(node1_input, self.u1, self.b1)
        with tf.name_scope('node2_input'):
            node2_input = tf.concat([vol_out, node1], 1)
        with tf.name_scope('Node2'):
            node2 = self.forward(node2_input, self.u1, self.b1)
        with tf.name_scope('node3_input'):
            node3_input = tf.concat([node2, pricing_date_out], 1)
        with tf.name_scope('Node3'):
            node3 = self.forward(node3_input, self.u1, self.b1)

        return node3
    # 效用函数、误差函数和预测价格
    def unti(self):
        node = self.flow()
        with tf.name_scope('price_'):
            price_ = tf.sqrt(tf.reduce_sum(tf.square(node)), name='prediction')
        with tf.name_scope('loss'):
            loss = tf.reduce_sum(tf.square(price_ - self.price + self.std_price))
        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)


        return price_, loss, optimizer

    # 训练部分
    def train(self):
        train_set = self.fetch_train_set()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=2)
            price_, loss, optimizer = self.unti()
            for i in range(self.iteration):
                loss_holder = 0.
                train = np.copy(train_set)
                print '-'*77
                for j in train:
                    _, lo, pri, add, rate = sess.run([optimizer, loss, price_, self.add_global, self.learning_rate],
                                                    feed_dict={self.price: j['price'],
                                                               self.spot: j['spot'],
                                                               self.pricing_date: j['pricing_date'],
                                                               self.mu: j['mu'],
                                                               self.vol: j['vol'],
                                                               })
                    loss_holder += lo
                if (i + 1) % 1 == 0:
                    print '\n'
                    saver.save(sess, './model_try/my-model', global_step=i + 1)
                    print '第' + str(i + 1) + '次训练模型已经保存'
                aver_loss = loss_holder/1423.
                print '第' + str(i + 1) + '次训练后，平均偏移误差为：'
                print aver_loss

    def fetch_test_set(self, start_id, end_id):
        temp = [self.db.fetch_one(index)[0] for index in np.arange(start_id, end_id + 1, 1)]
        return [self.db.vectorization(item) for item in temp]

    def restore_test(self):
        # 取出测试集
        start = ti.time()
        test = self.fetch_test_set(1425, 1616)
        test_set = np.copy(test)
        end = ti.time()
        print '取出测试集耗时：'+ str(end-start) + 's'+'\n' + '进入测试阶段'

        error_holder = []
        with tf.Session() as sess:
            # 重构模型
            saver = tf.train.import_meta_graph('./model_recursive/my-model-300.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./model_recursive/'))

            graph = tf.get_default_graph()
            # 将输入参数的容器重新构建
            price = graph.get_tensor_by_name('Price/price:0')
            spot = graph.get_tensor_by_name('Spot/spot:0')
            pricing_date = graph.get_tensor_by_name('Pricing_date/pricing_date:0')
            mu = graph.get_tensor_by_name('Mu/mu:0')
            vol = graph.get_tensor_by_name('Vol/vol:0')

            # 取出想要观察的变量
            loss = graph.get_tensor_by_name('loss/Sum:0')
            feed_list = [
                {
                    price: one['price'],
                    spot: one['spot'],
                    pricing_date: one['pricing_date'],
                    mu: one['mu'],
                    vol: one['vol']
                } for one in test_set]
            for piece in feed_list:
                pri, lo = sess.run([price, loss], piece)
                error_holder.append(lo/pri)
            return np.array(error_holder)
            # feed_list_price = [
            #     {
            #         price: two['price']
            #     } for two in test_set
            # ]
            # price_list = np.array([sess.run(price, feed) for feed in feed_list_price])
            # lo_list = np.array([sess.run(loss, feed) for feed in feed_list])
            # return lo_list/price_list


# 训练开始
# train = recursive()
# train.train()
# 测试
# test = recursive()
# result = test.restore_test()
# accuracy = np.where(result>0.05, 0, result)
# print np.count_nonzero(accuracy)/(1616.-1425.+1.)