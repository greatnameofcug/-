import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import math
#生成样本点和采样间隔
sampleNo = 50
mu = 5
std_var = 1
np.random.seed(0)
s = np.random.normal(mu, std_var, sampleNo )
x = np.linspace(0,10,1000)

#高斯
def get_phi_Gauss(u):
    phi = np.exp(-u*u.T/2)
    return phi

 
#p(x)高斯
def get_px_Gauss(x, y, h):
    phi = 0
    n = len(y)
    for i in range(n):
        # print("xi[i]", xi[i])
        u = (x-y[i])/h
        phi += get_phi_Gauss(u)
        
    px = 1/sampleNo/h*phi/math.sqrt(2*np.pi)
    return px

#方窗
def Kernel(x, xi, h):
    delta_x = abs(x-xi)
    #plt.plot(delta_x,color='red')
    delta_x[delta_x<=h/2]=0
    delta_x[delta_x>=h/2]=2
    delta_x[delta_x==0]=1
    delta_x[delta_x==2]=0
    #plt.plot(delta_x)
    #plt.show()
    
    return delta_x
#p(x)方窗
def get_px(x, y, h):
    n = len(y)
    K = 0
    for i in range(n):
        # print("xi[i]", xi[i])
        K += Kernel(x, y[i], h)
    print(K.shape)
    px = 1/sampleNo*K/h
    return px

get_px(x, s, 0.5)

f,((ax11,ax12,ax13),(ax21,ax22,ax23)) = plt.subplots(2,3,sharex = True,sharey = True)

ax11.set_title("h=0.1")
ax11.plot(x,get_px(x, s, 0.1),linewidth=0.5)
ax12.set_title("h=0.5")
ax12.plot(x,get_px(x, s, 0.5),linewidth=0.5)
ax13.set_title("h=1")
ax13.plot(x,get_px(x, s, 1),linewidth=0.5)
ax21.set_title("h=2")
ax21.plot(x,get_px(x, s, 2),linewidth=0.5)
ax22.set_title("h=5")
ax22.plot(x,get_px(x, s, 5),linewidth=0.5)
ax23.set_title("h=10")
ax23.plot(x,get_px(x, s, 10),linewidth=0.5)
 
plt.show()






#method 3 :easy to define structure
#这种方式不能生成指定跨行列的那种
import matplotlib.pyplot as plt
#(ax11,ax12),(ax13,ax14)代表了两行
#f就是figure对象,
#sharex：是否共享x轴
#sharey:是否共享y轴
f,((ax11,ax12,ax13),(ax21,ax22,ax23)) = plt.subplots(2,3,sharex = True,sharey = True)

ax11.set_title("h=0.1")
ax11.plot(x,get_px_Gauss(x, s, 0.1))
ax12.set_title("h=0.5")
ax12.plot(x,get_px_Gauss(x, s, 0.5))
ax13.set_title("h=1")
ax13.plot(x,get_px_Gauss(x, s, 1))
ax21.set_title("h=2")
ax21.plot(x,get_px_Gauss(x, s, 2))
ax22.set_title("h=5")
ax22.plot(x,get_px_Gauss(x, s, 5))
ax23.set_title("h=10")
ax23.plot(x,get_px_Gauss(x, s, 10))
 
plt.show()
