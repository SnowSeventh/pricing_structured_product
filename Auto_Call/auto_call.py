# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
# from datetime import date, datetime, time, timedelta
import time as ti
from Bussiness_Calender import Bcalender
from Bussiness_Calender.Bcalender import BussinessCalender
import sys



'''
这次重写的autocall模块，将神经网络部分从中剥离，但是加入计算greeks的功能。class以一份合约为样本，一切参数基于真实的合约参数，尤其是日历的概念。
定价的方法还是维持蒙特卡洛不变，由于内容过少，这一部分就不再单独模块化了，直接写入到产品中，以后的其他产品同理。
'''
bookdictionary = {}


class autocall:

    '''
    合约类参数，由于此次强调实用性和专业性，所以，即便对计算无实际意义的合约参数和类属性，保留属性，并且在生成之后的产品对象时，参数以dictionary的格式传入
    '''

    start = None  # 起始日--------------------✔️
    end = None  # 到期日----------------------✔️
    pricing_date = None #定价日---------------✔️-theta, 定价日的概念极其重要，具体而言，定价日分三类：1. 定价日处于合约期之前，这种情况下，
    # 我们要知道该日的标的物收盘价，并以其为spot蒙特卡洛到期初观察日的价格，生成合约期的标的物价格曲线，然后进行定价；2. 定价日处于合约期中（包括
    # 左右边界），这时候我们要知道定价日的spot并靠它生成之后的价格曲线；3. 定价日在合约期之后，这种情况考虑意义不大，因为历史价格都已确定，没有定价
    #的必要了
    spot = None  # 标的物初期价格--------------✔️-delta, gamma
    knockoutdays = None #敲出观察日------------×
    knockindays = None #敲入观察日-------------×
    upper_barrier = None #敲出障碍-------------×️
    lower_barrier = None #敲入障碍-------------×️
    interest_rate = None #票面利率，年化--------✔️
    knockoutrate = None #敲出收益率，年化--------×️
    mu = None #期望收益率，年化，r+repo+div------✔️-rho
    vol = None #波动率-------------------------✔️-vega
    r = None #无风险收益率，年化-----------------✔️-rho
    calender = None
    '''
    做标记的参数都是合约最关键的数值参数，是进行后续计算必不可少的参量；✔表示该参数为数值，类型与意义相符；×表示该数据可能为list结构也可能为数值，
    需要在使用前进行判断，切记。
    '''
    year_count = None  # 年度计息天数----------
    argument = None #本金---------------------
    std_spot = None #标的物价格变化基准---------
    curve_num = None #模特卡洛曲线数量----------
    delta_t = 1./252
    stdi = (delta_t)**0.5 #蒙特卡洛随机数标准差，由于标的物价格曲线逐日生成，所以delta t应为年化后的时间间隔，全年交易日按252计算------------
    aver = 0. #蒙特卡洛随机数均值--------------
    tolerance = 0.001 #准许误差-----------------
    '''
    以上这部分参数相对固定，与产品的变化相关性不太大，所以可以在初始化对象的时候赋默认的参数，这样也留下了自定义修改的可能；
    '''
    '''
    非计算属性
    '''

    name = None #产品名称
    code = None #产品代码
    underlying = None #标的物
    distribution = None #发行机构
    book = None #登记机构
    max_amount = None #最高发行规模
    min_amount = None #最低发行规模
    list_date = None #产品募集期
    duration = None #产品期限
    real_duration = None #实际续存期
    interest_count = None #到期计息天数
    real_interest_count = None #实际计息天数
    face_value = None #收益凭证面值
    distribution_price = None #发行价格
    min_call = None #最低认购金额
    call_date = None #认购日
    book_date = None #登记日
    first_examination_date = start #期初观察日
    last_examination_date = end #期末观察日
    place = None #发行场所
    risk_level = None #产品风险等级
    max_num = None #投资者人数上限
    receive = None #募集方式
    target = None #用途
    others = None #其他文字性内容
    underlying_code = None #标的物代码
    decimal = None #小数位数
    observation_date = None #观察日界定
    receive_level = None #收益表现水平
    knockout = None #敲出事件
    knockin = None #敲入事件

    def __init__(self, calender, inputdict, year= 365, argument= 100, std_spot= 100., curve_num= 50000, bookdict= bookdictionary):
        self.year_count = year
        self.argument = argument
        self.std_spot = std_spot
        self.curve_num = curve_num

        self.calender = calender

        '''
        这里应该把bookdict中的键值取出来并赋给上面的所有非计算属性，暂且空出来，不影响开发过程
        '''
        self.start = inputdict['start']
        self.end = inputdict['end']
        self.pricing_date = inputdict['pricing_date']
        self.spot = inputdict['spot']
        self.knockoutdays = inputdict['knockoutdays']
        self.knockindays = inputdict['knockindays']
        self.upper_barrier = inputdict['upper_barrier']
        self.lower_barrier = inputdict['lower_barrier']
        self.interest_rate = inputdict['interest_rate']
        self.knockoutrate = inputdict['knockoutrate']
        self.mu = inputdict['mu']
        self.vol = ['vol']
        self.r = ['r']

    '''
    蒙特卡洛生成标的物价格曲线，注：这里的t指的是一条曲线所含的交易日总天数，可以利用日历模块得出，区间边界可在生成时视情况而定。
    更新：只生成价格曲线不行，由于要计算的是实际产品，所以必须按照真实日期生成价格的时间序列，返回dataframe，传入参数时必须传入
    真实的start和end date。
    '''
    '''
    注：以下几个计算产品价格的功能函数：underlying_curve(), knock_out(), knock_in(), one_curve(), pricing(), 在调用
    的过程中，势必要传入5个日期相关的参数：start, end, pricing date, knockoutdays, knockindays. 而这几个参数都是由input字典
    通过字符串格式传入的（input字典中写字符串格式的日期更为方便），所以暂且不在各个子函数的内部修改日期类型，只在最后定价的时候统一的将
    字符串换乘日期格式。
    '''
    # 不行，得修改，矩阵第一列应该是s0;已修改；
    def underlying_curve(self, s0, start, end, curve_num, mu, vol):
        dateseries = self.calender.timeseries(start, end)
        t = len(dateseries)
        first = np.ones(curve_num)*s0
        np.random.seed(0)
        Wt = np.random.normal(self.aver, self.stdi, [curve_num, t-1])
        curve = s0*np.exp(vol*Wt + (mu - vol**2/2)*self.delta_t).cumprod(1)
        curve = np.c_[first, curve]
        curve = pd.DataFrame(curve, columns=dateseries)
        return curve
    # 也得修改，当定价日在合约期内，剩下的合约期可能一个敲出观察日都不含，也就是说outdate可能是空list，得加入这种情况的判断，结果就是
    # 绝对不敲出了。已修改。
    def knock_out(self, outdate, upper_barriers, std_price, underlying_curve):
        if outdate == []:
            return False
        else:
            if type(upper_barriers) is list:
                lg = len(outdate)
                for i in range(lg):
                    index = outdate[i]
                    if underlying_curve[index] > std_price*upper_barriers[i]:
                        return index
                return False
            else:
                for i in outdate:
                    if underlying_curve[i] > std_price*upper_barriers:
                        return i
                return False
    # 同上，已修改。
    def knock_in(self, indate, lower_barriers, std_price, underlying_curve):
        if indate == []:
            return False
        else:
            if indate is 'all':
                for i in underlying_curve.index:
                    if underlying_curve[i] < std_price*lower_barriers:
                        return True
                return False
            else:
                if type(lower_barriers) is list:
                    lg = len(indate)
                    for i in range(lg):
                        index = indate[i]
                        if underlying_curve[index] < std_price*lower_barriers[i]:
                            return True
                    return False
                else:
                    for i in indate:
                        if underlying_curve[i] < std_price*lower_barriers:
                            return True
                    return False

    def dtype(self, pricing_date, left, right):
        b = self.calender.stringtodate(pricing_date)
        a = self.calender.stringtodate(left)
        c = self.calender.stringtodate(right)
        if (b-a).days >= 0:
            return 'in'
        else:
            return 'out'

    '''
    单一价格生成函数的参数不是产品输入参数，而是符合计算要求的逻辑参数，由pricing函数调用前预生成，这里就先搁置。
    '''
    def price_one_curve(self, label, one_curve, start, end, pricing_date, outdate, indate, upper, lower,
                        outrate, divident, r):
        outday = self.knock_out(outdate, upper, self.std_spot, one_curve)
        if label is 'in':
            if outday:
                if type(outrate) is list:
                    receive = outrate[outdate.index(outday)]
                    expire = (outday - pricing_date).days/float(self.year_count)
                    discount_factor = np.exp(-r*expire)
                    return self.argument*(1 + receive*expire)*discount_factor
                else:
                    receive = outrate
                    expire = (outday - pricing_date).days/float(self.year_count)
                    discount_factor = np.exp(-r * expire)
                    return self.argument * (1 + receive * expire) * discount_factor
            else:
                if self.knock_in(indate, lower, self.std_spot, one_curve):
                    receive = min(one_curve[end]/self.std_spot, 1.)
                    expire = (end - pricing_date).days/float(self.year_count)
                    discount_factor = np.exp(-r*expire)
                    return self.argument*receive*expire*discount_factor
                else:
                    receive = divident
                    expire = (end - pricing_date).days/float(self.year_count)
                    discount_factor = np.exp(-r*expire)
                    return self.argument*(1 + receive*expire)*discount_factor
        else:
            if outday:
                if type(outrate) is list:
                    receive = outrate[outdate.index(outday)]
                    interest_countdays = (outday - start).days/float(self.year_count)
                    expire = (outday - pricing_date).days/float(self.year_count)
                    discount_factor = np.exp(-r*expire)
                    return self.argument*(1 + receive*interest_countdays)*discount_factor
                else:
                    receive = outrate
                    interest_countdays = (outday - start).days / float(self.year_count)
                    expire = (outday - pricing_date).days/float(self.year_count)
                    discount_factor = np.exp(-r * expire)
                    return self.argument * (1 + receive * interest_countdays) * discount_factor
            else:
                if self.knock_in(indate, lower, self.std_spot, one_curve):
                    receive = min(one_curve[end]/self.std_spot, 1.)
                    interest_countdays = (end - start).days/float(self.year_count)
                    expire = (end - pricing_date).days/float(self.year_count)
                    discount_factor = np.exp(-r*expire)
                    return self.argument*receive*interest_countdays*discount_factor
                else:
                    receive = divident
                    interest_countdays = (end - start).days / float(self.year_count)
                    expire = (end - pricing_date).days/float(self.year_count)
                    discount_factor = np.exp(-r*expire)
                    return self.argument*(1 + receive*interest_countdays)*discount_factor

    def inputtrans(self, one_product):
        one_product['start'] = self.calender.stringtodate(one_product['start'])
        one_product['end'] = self.calender.stringtodate(one_product['end'])
        one_product['pricing_date'] = self.calender.stringtodate(one_product['pricing_date'])
        one_product['knockoutdays'] = Bcalender.element_wise_op(one_product['knockoutdays'], self.calender.stringtodate)
        if type(one_product['knockindays']) is list:
            one_product['knockindays'] = Bcalender.element_wise_op(one_product['knockindays'], self.calender.stringtodate)
        return one_product

    def prepareforinput(self, label, one_product):
        one = self.inputtrans(one_product)
        if label is 'in':
            # out部分
            # 1. outlist空了
            firstout = self.calender.cut(one['knockoutdays'], one['pricing_date'])
            if firstout is 'empty':
                one['knockoutdays'] = []
            # 2. outlist非空
            else:
                one['knockoutdays'] = one['knockoutdays'][firstout:]
                if type(one['upper_barrier']) is list:
                    one['upper_barrier'] = one['upper_barrier'][firstout:]
                if type(one['knockoutrate']) is list:
                    one['knockoutrate'] = one['knockoutrate'][firstout:]
            # in部分
            # 1. indays是list
            if type(one['knockindays']) is list:
                firstin = self.calender.cut(one['knockindays'], one['pricing_date'])
                if firstin is 'empty':
                    one['knockindays'] = []
                else:
                    one['knockindays'] = one['knockindays'][firstin:]
                    if type(one['lower_barrier']) is list:
                        one['lower_barrier'] = one['lower_barrier'][firstin:]
            # 2. indays是all，不需要切割
        else:
            return one
        return one

    def pricing(self, one_product):
        price = 0.
        label = self.dtype(one_product['pricing_date'], one_product['start'], one_product['end'])
        if label is 'in':
        # 开始处理输入信息，包括日期格式的转化和有效时间程度的切割，使得数据可以使用.
            after = self.prepareforinput(label, one_product)
            curve = self.underlying_curve(after['spot'], after['pricing_date'], after['end'], self.curve_num, after['mu'], after['vol'])
            for i in range(self.curve_num):
                one_curve = curve.loc[i]
                price += self.price_one_curve(label, one_curve, after['start'], after['end'], after['pricing_date'], after['knockoutdays'],
                                              after['knockindays'], after['upper_barrier'], after['lower_barrier'], after['knockoutrate'],
                                              after['interest_rate'], after['r'])
            return price/self.curve_num
        else:
            after = self.prepareforinput(label, one_product)
            # 注：此处传入underlying_curve中的spot为用户输入的产品参数，也就是start当日的标的物收盘价，这个价格在当下的定价日是不存在的，也即是需要预测的
            #未来价格，但方便起见这里写成了用户输入。
            curve = self.underlying_curve(after['spot'], after['start'], after['end'], self.curve_num, after['mu'], after['vol'])
            for i in range(self.curve_num):
                one_curve = curve.loc[i]
                price += self.price_one_curve(label, one_curve, after['start'], after['end'], after['pricing_date'],
                                              after['knockoutdays'],
                                              after['knockindays'], after['upper_barrier'], after['lower_barrier'],
                                              after['knockoutrate'],
                                              after['interest_rate'], after['r'])
            return price/self.curve_num
    # 接下来，开始通过产品价格计算greeks
    # delta
    def delta(self, product1, product2, deltashift):
        pv2 = self.pricing(product2)
        pv1 = self.pricing(product1)
        return (pv2-pv1)/(2*deltashift)

    def greeks(self, product, shift, name):
        if name is 'delta':
            product1 = product.copy()
            product2 = product.copy()
            product1['spot'] = product1['spot']*(1-shift)
            product2['spot'] = product2['spot']*(1+shift)
            return (self.pricing(product2)-self.pricing(product1))/(2*shift)
        if name is 'gamma':
            product1 = product.copy()
            product2 = product.copy()
            product1['spot'] = product1['spot'] * (1 - shift)
            product2['spot'] = product2['spot'] * (1 + shift)
            return (self.pricing(product2)+self.pricing(product1)-2*self.pricing(product))/shift
        # if name is 'theta':
        '''
        有点疑问，为什么是加一个自然日，先空出来
        '''
        if name is 'vega':
            product1 = product.copy()
            product1['vol'] = product1['vol']+shift
            return self.pricing(product1)-self.pricing(product)
        if name is 'rho':
            product1 = product.copy()
            product1['r'] = product1['r']+shift
            return self.pricing(product1)-self.pricing(product)
info= {
    'start': '2017/11/9',
    'end': '2018/11/15',
    'pricing_date': '2017/11/9',
    'spot': 100.,
    'knockoutdays': [['2017/12/14'],['2018/1/11'],['2018/2/8'],['2018/3/8'],['2018/4/12'],['2018/5/10'],['2018/6/14'],
                     ['2018/7/12'],['2018/8/9'],['2018/9/13'],['2018/10/11']],
    'knockindays': 'all',
    'upper_barrier': 1.,
    'lower_barrier': 0.86,
    'interest_rate': 0.14,
    'knockoutrate': 0.14,
    'mu': 0.1,
    'vol': 0.2,
    'r': 0.1
}

# test = autocall(BussinessCalender(), info)
# start = ti.time()
# print test.pricing(info)
# end = ti.time()
# print str(end - start)+'s'




