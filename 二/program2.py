

import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import math
import copy

#数据
w_1 = [[0.1, 1.1], [6.8, 7.1], [-3.5, -4.1], [2.0, 2.7], [4.1, 2.8], [3.1, 5.0], [-0.8, -1.3], [0.9, 1.2], [5.0, 6.4], [3.9, 4.0]]
w_2 = [[7.1, 4.2], [-1.4, -4.3], [4.5, 0.0], [6.3, 1.6], [4.2, 1.9], [1.4, -3.2], [2.4, -4.0], [2.5, -6.1], [8.4, 3.7], [4.1, -2.2]]
w_3 = [[-3.0, -2.9], [0.5, 8.7], [2.9, 2.1], [-0.1, 5.2], [-4.0, 2.2], [-1.3, 3.7], [-3.4, 6.2], [-4.1, 3.4], [-5.1, 1.6], [1.9, 5.1]]
w_4 = [[-2.0, -8.4], [-8.9, 0.2], [-4.2, -7.7], [-8.5, -3.2], [-6.7, -4.0], [-0.5, -9.2], [-5.3, -6.7], [-8.7, -6.4], [-7.1, -9.7], [-8.0, -6.3]]

#规范化增广样本
def samples_trans(w,key):
    # 复制样本,防止后续操作改变原始样本
    ww = copy.deepcopy(w)

    for i in ww: i.append(1)
    ww = np.array(ww)
    if key == 1:
        ww = -ww
    return ww

#计算权重a
def batch_perception(w1, w2, eta, theta):
    w1 = samples_trans(w1,0)
    w2 = samples_trans(w2,1)
    w = np.concatenate([w1,w2])

    a = np.zeros_like(w[1])
    
    count = 0
    while True:
        y = np.zeros_like(w[1])
        for i in w:
            if a.T.dot(i).T<=theta:
                y+=i
        #输出每一次迭代的解向量a和sum(y)
        print(count,'\t',a,'\t',y)
        if all(y == 0)or count>2000:
            break
        
        a += eta*y
        count += 1
        

batch_perception(w_2,w_3,1,0.01)


'''
    w_1 = copy.deepcopy(w1)
    w_2 = copy.deepcopy(w2)

    # 增广
    for i in w_1: i.append(1)
    for i in w_2: i.append(1)

    # 规范化
    w_1 = np.array(w_1)
    w_2 = np.array(w_2)
    w_2 = -w_2
    w = np.concatenate([w_1, w_2])

    return w
'''