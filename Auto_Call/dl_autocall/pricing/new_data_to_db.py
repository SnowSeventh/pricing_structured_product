# -*- coding: UTF-8 -*-
import numpy as np
import sys

from Auto_Call.auto_call import autocall
from Bussiness_Calender.Bcalender import BussinessCalender
import pymongo
import time as ti
import random as rd

'''
这次更新数据库，目的是使产品数据离散化，并只改变和greeks相关的四个变量，取值范围为：
- spot：基准值100，上下浮动百分之五，以0.1%为步长，取100个点；
- pricing_date：在续存期内，取四个点：0，0.25，0.5，0.75，基准值为0；
- mu：取两个点，-0.12，-0.4，基准值为-0.12；
- vol：取两个点，0.15，0.25，基准值为0.15；
因此，总数据量为100*4*4*4 = 1600。
'''

# 数据库设置
myclient = pymongo.MongoClient('mongodb://localhost:27017/')
mydb = myclient['pricing']
mycol = mydb['new_record']


# 设置标准合约以及四个参数的变化范围
std_info= {
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
spot_index = np.arange(-5., 5.1, 0.1) + std_info['spot']
pricing_date_index = np.array(['2017/11/9', '2018/2/19', '2018/5/21', '2018/8/20'])
mu_index = np.array([-0.12, -0.04])
vol_index = np.array([0.15, 0.25])

pricer = autocall(BussinessCalender(), std_info.copy(), argument= 1)

# 生成离散的产品样本空间
def func(x,y,z):
    q = x.copy()
    q[y] = z
    return q

def trans(index_holder, data, item):
    temp_holder = [data]*len(index_holder)
    record = [func(i,item,j) for i, j in zip(temp_holder, index_holder)]
    return record

trans_spot = trans(spot_index, std_info, 'spot')
trans_pricing_date = sum([trans(pricing_date_index, piece, 'pricing_date') for piece in trans_spot],[])
trans_mu = sum([trans(mu_index, piece, 'mu') for piece in trans_pricing_date], [])
trans_vol = sum([trans(vol_index, piece, 'vol') for piece in trans_mu], [])
rd.shuffle(trans_vol)
random_list = trans_vol

print '数据预处理结束\n'+'@'*77

# 取出产品样本，定价并上传
index = 1
for one in random_list:
    mirror = one.copy()
    try:
        start = ti.time()
        price = pricer.pricing(mirror)
        one['price'] = price
        one['_id'] = index
        mycol.insert_one(one)
        end = ti.time()
        consum = end - start
        i = np.copy(index)
        print '-' * 77
        print '第' + str(i) + '份合约已上传'
        print '\n'
        print '用时' + str(consum) + 's'
        index += 1
    except:
        i = np.copy(index)
        print '第' + str(i) + '份合约出错'
        print '\n'
        print sys.exc_info()
        index += 1




