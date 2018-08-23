from Bussiness_Calender import Bcalender
import numpy as np
test = Bcalender.BussinessCalender()
rec = test.loadfile()
print np.array(test.timeseries(rec[0],rec[1])).reshape(6,4)