# -*- coding: UTF-8 -*-
from datetime import date, datetime, time, timedelta
import calendar
import numpy as np
import pandas as pd


def element_wise_op(array, op):
    holder = []
    l = len(array)
    for i in array:
        holder.append(op(i[0]))
    return holder
    # for i in np.nditer(array, op_flags=['readwrite']):
    #     i[...] = op(i)

class BussinessCalender:

    '''
    默认处理Y/m/d型的字符串日期，且转化成date之后只保留年月日
    '''
    def stringtodate(self,stringdate,format = '%Y/%m/%d'):
       return datetime.strptime(stringdate,format).date()

    '''
    切割日期list，传入模糊值，返回比这个值大的最近的元素的位置
    '''
    def cut(self, list, value):
        for i in list:
            if (i-value).days >0:
                return list.index(i)
        return 'empty'
    '''
    通过pandas读取csv得到日期array，默认为本地目录下日历文件，且所有节日都转化成日期格式
    '''
    def loadfile(self, path ='/Users/wangpeng/tutu/htsc/product/Bussiness_Calender/holiday_test.csv'):
        df = pd.read_csv(path, header=None)
        holiday_array = df.values.tolist()
        holiday_array = element_wise_op(holiday_array, self.stringtodate)
        return holiday_array

    def isweekend(self, date):
       if datetime.isoweekday(date) in [6,7]:
           return True
       else:
           return False

    def isholiday(self, date):
        if date in self.loadfile():
            return True
        else:
            return False

    def nextbussinessday(self, date):
        shift = timedelta(days=1)
        nextday = date + shift
        while self.isholiday(nextday) or self.isweekend(nextday):
            nextday += shift
        return nextday

    def prevbussinessday(self, date):
        shift = timedelta(days= -1)
        previday = date + shift
        while self.isholiday(previday) or self.isweekend(previday):
            previday -= shift
        return previday

    '''
    按日漂移函数，通过传入一个日期（确定的日期格式），可以返回相应的天数之前或之后的第一个交易日，默认返回其后的第一个交易日
    '''
    def Edate(self, date, num = 1):
        result = None
        if num >= 0:
            for i in range(num):
                result = self.nextbussinessday(date)
            return result
        else:
            for j in range(num):
                result = self.prevbussinessday(date)
            return result

    '''
    在做好以上功能后，大部分产品的续存期时间序列都可以生成；然而，对于有敲入和敲出的期权，必须同时按照敲入和敲出的频率，生成
    相应的观察时间序列，这就涉及到更加宽泛的时间漂移功能，包括Emonth，Eweek等等；同时由于生成时间序列的要求不同（modified following、month end
    或者是每个月第几个星期几），所以，函数实在难以确定。为了降低现阶段的难度，观察日暂时不能自动生成，而是由定价人员人为的确定，即
    load进一份含有敲入和敲出观察日的文件，格式问题同上。
    '''
    def load_target_date(self, filepath='/Users/wangpeng/tutu/htsc/product/Bussiness_Calender/knockoutdate.csv'):
        df = pd.read_csv(filepath, header=None)
        knock_array = df.values.tolist()
        knock_array = element_wise_op(knock_array, self.stringtodate)
        return knock_array

    '''
    生成时间序列，基于起始日（也为初期观察日）和到期终止日（也为期末观察日，即没有中间敲出的完整到期终止日）。注意，这两个日期必须为交易日，并且包含
    在生成的时间序列中。至于产品期限和计息长度对区间边界的界定，不需要再日历部分做好，生成时间序列即可，是否使用边界点可在定价模块中写。
    '''
    def timeseries(self, start, end):
        left = np.copy(start)
        left = left.tolist()
        right = np.copy(end)
        holder = []
        while (right - left).days >= 0:
            holder.append(left)
            left = self.nextbussinessday(left)
        return holder

    '''
    理论上来说，以上功能函数能满足目前几乎所有产品的日期相关功能，可以生成真实的交易日时间序列，并应用到之后的定价和风控过程中；美中不足的是还有很多的
    日期漂移函数和固定频率观察函数功能的缺失，但由于没有统一的标准，这部分暂且放下。
    '''








