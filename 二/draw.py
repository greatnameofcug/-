import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
 
 
# 数据
data = np.arange(24).reshape((8, 3))
# data的值如下：
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]
#  [12 13 14]
#  [15 16 17]
#  [18 19 20]
#  [21 22 23]]
x1_train = np.array([[1.58, 2.32, -5.8], [0.67, 1.58, -4.78], [1.04, 1.01, -3.63],
                   [-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73], [1.39, 3.16, 2.87],
                   [1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [0.45, 1.33, -4.38],
                   [-0.76, 0.84, -1.96]])
x2_train = np.array([[0.21, 0.03, -2.21], [0.37, 0.28, -1.8], [0.18, 1.22, 0.16],
                   [-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39], [0.74, 0.96, -1.16],
                   [-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [0.44, 1.31, -0.14],
                   [0.46, 1.49, 0.68]])
x3_train = np.array([[-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [1.55, 0.99, 2.69],
                   [1.86, 3.19, 1.51], [1.68, 1.79, -0.87], [3.51, -0.22, -1.39],
                   [1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [0.25, 0.68, -0.99],
                   [0.66, -0.45, 0.08]])
x1 = x1_train[:, 0]  # [ 0  3  6  9 12 15 18 21]
y1 = x1_train[:, 1]  # [ 1  4  7 10 13 16 19 22]
z1 = x1_train[:, 2]  # [ 2  5  8 11 14 17 20 23]
x2 = x2_train[:, 0]  # [ 0  3  6  9 12 15 18 21]
y2 = x2_train[:, 1]  # [ 1  4  7 10 13 16 19 22]
z2 = x2_train[:, 2]  # [ 2  5  8 11 14 17 20 23]
x3 = x3_train[:, 0]  # [ 0  3  6  9 12 15 18 21]
y3 = x3_train[:, 1]  # [ 1  4  7 10 13 16 19 22]
z3 = x3_train[:, 2]  # [ 2  5  8 11 14 17 20 23]
# 绘制散点图
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x1, y1, z1, c='r')
ax.scatter(x2, y2, z2, c='g')
ax.scatter(x3, y3, z3, c='b')
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
plt.show()

'''
import matplotlib.pyplot as plt
import numpy as np
 
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
'''
'''
plt.plot(x,y,label='p(x)')

plt.legend(loc = 'upper right')
plt.xlim(-6,9)
plt.ylim(0,5)
plt.show()
'''

