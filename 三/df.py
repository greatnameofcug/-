import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
 
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.signal as signal



eta = [0.01]
batch_size = [10]
x = np.linspace(0,1000000,1000)
Data1 = []
Data2 = []
title = ['[323]','[333]','[343]','[363]','[393]','[3123]','[3723]']
for t in title:
    data1=[]
    data2=[]
    f1 = open('C:/Users/14775/Desktop/模式识别导论/作业/三/'+t+'1000000_0.01_10.txt','r')
    data = f1.readlines()
    f1.close()
    for line in data:
        try:
            data1.append(float((line.strip('\n'))))
        except:
            pass
    data2=data1[1000000:1001000]
    data1=data1[0:999999]
    data1 = np.array(data1)
    data2 = np.array(data2)
    data1 = signal.medfilt(data1,7)
    Data1.append(data1)
    Data2.append(data2)


fig = plt.figure()

axes = fig.subplots(nrows=1, ncols=1, sharex = True,sharey = True)
cc = 6
for ax in fig.axes:
    ax.set_title(title[cc],fontsize='xx-small',horizontalalignment='right')
    ax.plot(Data1[cc],linewidth=0.1,label='loss')
    ax.plot(x,Data2[cc],label='acc')
    cc += 1

lines, labels = fig.axes[-1].get_legend_handles_labels()
    
fig.legend(lines, labels)
plt.tight_layout()
plt.show()
'''
f,((ax11,ax12,ax13),(ax21,ax22,ax23),(ax31,ax32,ax33),(ax41,ax42,ax43),(ax51,ax52,ax53)) = plt.subplots(5,3,sharex = True,sharey = True)

ax11.set_title(title[0])
ax11.plot(Data1[0],linewidth=0.5)
ax11.plot(x,Data2[0])

ax12.set_title(title[1])
ax12.plot(Data1[1],linewidth=0.5)
ax12.plot(x,Data2[1])

ax13.set_title(title[2])
ax13.plot(Data1[2],linewidth=0.5)
ax13.plot(x,Data2[2])

ax21.set_title(title[3])
ax21.plot(Data1[3],linewidth=0.5)
ax21.plot(x,Data2[3])

ax22.set_title(title[4])
ax22.plot(Data1[4],linewidth=0.5)
ax22.plot(x,Data2[4])

ax23.set_title(title[5])
ax23.plot(Data1[5],linewidth=0.5)
ax23.plot(x,Data2[5])

ax31.set_title(title[6])
ax31.plot(Data1[6],linewidth=0.5)
ax31.plot(x,Data2[6])

ax32.set_title(title[7])
ax32.plot(Data1[7],linewidth=0.5)
ax32.plot(x,Data2[7])

ax33.set_title(title[8])
ax33.plot(Data1[8],linewidth=0.5)
ax33.plot(x,Data2[8])

ax41.set_title(title[9])
ax41.plot(Data1[9],linewidth=0.5)
ax41.plot(x,Data2[9])

ax42.set_title(title[10])
ax42.plot(Data1[10],linewidth=0.5)
ax42.plot(x,Data2[10])

ax43.set_title(title[11])
ax43.plot(Data1[11],linewidth=0.5)
ax43.plot(x,Data2[11])

ax51.set_title(title[12])
ax51.plot(Data1[12],linewidth=0.5)
ax51.plot(x,Data2[12])

ax52.set_title(title[13])
ax52.plot(Data1[13],linewidth=0.5)
ax52.plot(x,Data2[13])

ax53.set_title(title[14])
ax53.plot(Data1[14],linewidth=0.5)
ax53.plot(x,Data2[14])
plt.tight_layout()
plt.show()


plt.plot(x,data2)
plt.plot(data1)
plt.show()
print(data1[-10:-1])

plt.plot(x,data2)
plt.plot(data1)
plt.show()
print(data1[-10:-1])
'''
'''
x = np.linspace(-6,9,1500)
interval0 = [1 if (i<-2) else 0 for i in x]
interval1 = [1 if (i>-2 and i<3) else 0 for i in x]
interval2 = [1 if (i>3) else 0 for i in x]

y = 1/(6*abs(x+4))*interval0 + 1/(6*abs(x))*interval1 + 1/(6*abs(x-6))*interval2
A = np.random.rand(10, 10)
print(A)
A[A > 0.5] = 5
print(A)
for i in range(5):
    print(i)


plt.plot(x,y,label='p(x)')

plt.legend(loc = 'upper right')
plt.xlim(-6,9)
plt.ylim(0,5)
plt.show()
'''

