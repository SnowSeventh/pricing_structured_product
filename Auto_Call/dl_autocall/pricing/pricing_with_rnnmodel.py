# -*- coding: UTF-8 -*-
import tensorflow as tf
from data_vector import vector

# 用户输入框
input= {
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
std_price = 0.952298142703
db = vector()
product = db.vectorization(input)

with tf.Session() as sess:
    # 重构模型
    saver = tf.train.import_meta_graph('./model_recursive/my-model-300.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./model_recursive/'))

    graph = tf.get_default_graph()
    # 将输入参数的容器重新构建
    spot = graph.get_tensor_by_name('Spot/spot:0')
    pricing_date = graph.get_tensor_by_name('Pricing_date/pricing_date:0')
    mu = graph.get_tensor_by_name('Mu/mu:0')
    vol = graph.get_tensor_by_name('Vol/vol:0')

    # 取出想要观察的变量
    price_ = graph.get_tensor_by_name('price_/prediction:0')
    feed_list = {
            spot: product['spot'],
            pricing_date: product['pricing_date'],
            mu: product['mu'],
            vol: product['vol']
        }
    print sess.run(price_, feed_list) + std_price

