# -*- coding: UTF-8 -*-
import numpy as np
import pymongo
import sys
from Auto_Call.auto_call import autocall
from Bussiness_Calender.Bcalender import BussinessCalender


info=  {
            'start': '2017/11/9',
            'end': '2018/11/15',
            'pricing_date': '2017/11/9', # 可变
            'spot': 100.,  # 可变
            'knockoutdays': [['2017/12/14'],['2018/1/11']],
            'knockindays': [['2018/9/13'],['2018/10/11']],
            'upper_barrier': 1.,
            'lower_barrier': 0.86,
            'interest_rate': 0.14,
            'knockoutrate': 0.14,
            'mu': -0.12, # 可变
            'vol': 0.15, # 可变
            'r': 0.
    }


host = 'mongodb://localhost:27017/'
'''
通用类vector有两个主要作用：1. 能根据需求从数据库中调取相应库相应collection中一定数量的数据；2. 数据预处理，能根据定价或者计算希腊值的不同需求
处理相应的数据，将数据向量化。以上两种功能都是通用方法。
'''
class vector:

    myclient = pymongo.MongoClient(host)

    def __init__(self, dbs_name = 'pricing', collection_name = 'record', amount = 100, target = 'pricing'):
        self.dbs_name = dbs_name
        self.collection_name = collection_name
        self.amount = amount
        self.target = target
        self.calender = BussinessCalender()

    def fetch_data(self):
        holder = []
        mydbs = self.myclient[self.dbs_name]
        mycol = mydbs[self.collection_name]
        for i in mycol.find().limit(self.amount):
            try:
                holder.append(i)
            except:
                print sys.exc_info()
        return holder

    def fetch_one(self,id):
        mydbs = self.myclient[self.dbs_name]
        mycol = mydbs[self.collection_name]
        myquery = {'_id':id}
        return mycol.find(myquery)

    def normalize_time(self, before, label):
        product = before.copy()
        length = (product['end'] - product['start']).days + 1.
        position = (product['pricing_date'] - product['start']).days
        outdays_holder = []
        indays_holder = []

        if label == 'out':
            # outdays
            for i in product['knockoutdays']:
                pp =  ((i - product['start']).days + 1)/length
                outdays_holder.append([0., pp, 0.])
            product['knockoutdays'] = np.array([outdays_holder])
            # indays
            if product['knockindays'] == 'all':
                timeseries = self.calender.timeseries(product['start'], product['end'])
                for i in timeseries:
                    pp = ((i - product['start']).days + 1)/length
                    indays_holder.append([0., pp, 0.])
                product['knockindays'] = np.array([indays_holder])
            else:
                for i in product['knockindays']:
                    pp = ((i - product['start']).days + 1)/length
                    indays_holder.append([0., pp, 0.])
                product['knockindays'] = np.array([indays_holder])
        else:
            # outdays
            if not product['knockoutdays'] == []:
                for i in product['knockoutdays']:
                    pp = ((i - product['start']).days + 1)/length
                    outdays_holder.append([0., pp, 0.])
                product['knockoutdays'] = np.array([outdays_holder])
            # indays
            if not product['knockindays'] == []:
                if product['knockindays'] == 'all':
                    timeseries = self.calender.timeseries(product['pricing_date'], product['end'])
                    for i in timeseries:
                        pp = ((i - product['start']).days + 1)/length
                        indays_holder.append([0., pp, 0.])
                    product['knockindays'] = np.array([indays_holder])
                else:
                    for i in product['knockindays']:
                        pp = ((i - product['start']).days + 1) / length
                        indays_holder.append([0., pp, 0.])
                    product['knockindays'] = np.array([indays_holder])
        # pricing_date, start, end
        # 更正：为了保证当定价日在起点时的数值为0，position不在加一或者减一；
        product['pricing_date'] = np.array([[0., position/length, 0.]])
        product['start'] = np.array([[0., 0., 0.]])
        product['end'] = np.array([[1., 0., 0.]])

        return product

    def normalize_value(self, before):
        product = before.copy()
        # upper barrier
        if product['knockoutdays'] == []:
            product['upper_barrier'] = []
        else:
            if not type(product['upper_barrier']) == list:
                length = len(product['knockoutdays'][0])
                product['upper_barrier'] = np.array([[[product['upper_barrier'], 0., 0.]]*length])
            else:
                upper_holder = []
                for i in product['upper_barrier']:
                    upper_holder.append([product['upper_barrier'], 0., 0.])
                product['upper_barrier'] = np.array([upper_holder])

        # lower barrier
        if product['knockindays'] == []:
            product['lower_barrier'] = []
        else:
            if not type(product['lower_barrier']) == list:
                length = len(product['knockindays'][0])
                product['lower_barrier'] = np.array([[[product['lower_barrier'], 0., 0.]]*length])
            else:
                lower_holder = []
                for i in product['lower_barrier']:
                    lower_holder.append([product['lower_barrier'], 0., 0.])
                product['lower_barrier'] = np.array([lower_holder])
        # spot
        product['spot'] = np.array([[product['spot']-info['spot'], 0., 0.]]) #重点：注意此处在向量化的过程中为了使三个维度的值大概处在合理近似范围
        # 将真实的spot除以100,更正，改为做差;再次更正，不能spot整体减去98，应该是第一维的值单独减去98;更正，减去std_price.

        return product

    def normalize_rate(self, before):
        product = before.copy()
        # knockoutrate
        if product['knockoutdays'] == []:
            product['knockoutrate'] = []
        else:
            if not type(product['knockoutrate']) == list:
                length = len(product['knockoutdays'][0])
                product['knockoutrate'] = np.array([[[0., 0., product['knockoutrate']]]*length])
            else:
                rate_holder = []
                for i in product['knockoutrate']:
                    rate_holder.append([0., 0., product['knockoutrate']])
                product['knockoutrate'] = np.array([rate_holder])
        # interest, mu, vol, r
        product['interest_rate'] = np.array([[0., 0., product['interest_rate']]])
        product['mu'] = np.array([[0., 0., product['mu'] - info['mu']]])
        product['vol'] = np.array([[0., 0., product['vol'] - info['vol']]])
        product['r'] = np.array([[0., 0., product['r']]])

        return product

    def vectorization(self, product):
        #copy神技不解释
        mirror = product.copy()
        '''
        # 纯价值型变量转化，v；
        mirror['sopt'] = np.array([mirror['spot'], 0., 0.])
        if type(mirror['upper']) is list:
            mirror['upper'] = np.array(mirror['upper'])*self.std_price
            att1 = np.zeros([mirror['upper'].shape[0], 2])
            mirror['upper'] = np.c_[mirror['upper'], att1]
        else:
            mirror['upper'] = np.array([mirror['upper']*self.std_price, 0., 0.])
        if type(mirror['lower']) is list:
            mirror['lower'] = np.array(mirror['lower'])*self.std_price
            att2 = np.zeros([mirror['lower'].shape[0], 2])
            mirror['lower'] = np.c_[mirror['lower'], att2]
        else:
            mirror['lower'] = np.array([mirror['lower']*self.std_price, 0., 0.])


        #mirror['upper'] = np.array([mirror['upper'], 0., 0.])
        #mirror['lower'] = np.array([mirror['lower'], 0., 0.])
        '''
        '''
        在将产品向量化之前，很多东西需要处理，也就是autocall中做过的preparation，这部分可以将outdays、indays、outbarrier和inbarrier切割
        成pricing_date之下真正需要考虑的数据集，可以调用autocall中已有的功能部分实现，剩下的诸如将日期数字化则在这一部分从头开始写；
        '''
        model = autocall(BussinessCalender(), mirror)
        label = model.dtype(mirror['pricing_date'], mirror['start'], mirror['end'])
        after = model.prepareforinput(label, mirror)
        t_done = self.normalize_time(after, label)
        v_done = self.normalize_value(t_done)
        r_done = self.normalize_rate(v_done)
        return r_done

# test， 搞定结果和形式都没问题，已经达到要求

# test = vector(amount=10)
# # test_set = test.fetch_one(3)
# # print test_set[0]
# result = test.vectorization(info)
# print result
