import numpy as np
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
#from mnist import load_mnist
from collections import OrderedDict
import matplotlib.pyplot as plt


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Sigmoid:

    def __init__(self):
        self.out = None
        self.loss = None
        self.y = None # 网络的输出
        self.t = None # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = sigmoid(x)
        out = sigmoid(x)
        self.out = out
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss
        '''
        out = sigmoid(x)
        self.out = out
        return out
        '''

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 监督数据是one-hot-vector的情况
            #print(self.out.shape)
            #print((dout * (1.0 - self.out) * self.out).shape)
            #print((self.y - self.t).shape)
            dx = dout * (1.0 - self.out) * self.out * (self.y - self.t) / batch_size
            #dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx
        '''
        dx = dout * (1.0 - self.out) * self.out

        return dx
        '''

#未完善
class Tanh:

    def __init__(self):
        self.out = None

    def forward(self, x):
        out = tanh(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (2.0 - self.out) * self.out
        #dx = dout * (1.0 - self.out) * self.out

        return dx

class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 权重和偏置参数的导数
        self.dW = None
        self.db = None

    def forward(self, x):
        # 对应张量
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmax的输出
        self.t = None # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    
def tanh(x):
    return 2 / (1 + np.exp(-2*x))
    #return 2*sigmoid(2*x)-1

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        batch_size = y.shape[0]
        return 0.5 * np.sum((y-t)**2)/batch_size
        #t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 初始化权重
        #hidden_size = [1,2,3,4,5]
        self.params = {}
        l = len(hidden_size)
        for i in range(l-1):
            no = i + 1
            w = 'W' + str(no)
            b = 'b' + str(no)
            self.params[w] = weight_init_std * np.random.randn(hidden_size[i],hidden_size[i+1])   #123
            self.params[b] = np.zeros(hidden_size[i+1])


        '''
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)
        '''
        # 生成层
        self.layers = OrderedDict()
        for i in range(l-1):
            affine = 'Affine' + str(i + 1)
            #sgm = 'Sigmoid' + str(i + 1)
            th = 'Tanh' + str(i + 1)
            w = 'W' + str(i + 1)
            b = 'b' + str(i + 1)
            self.layers[affine] = Affine(self.params[w], self.params[b])

            if i != l-2:
                #self.layers[sgm] = Sigmoid()
                self.layers[th] = Tanh()

        '''
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        '''
        self.lastLayer = Sigmoid()
        #self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads



# 读入数据
#(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
x_train = np.array([[1.58, 2.32, -5.8], [0.67, 1.58, -4.78], [1.04, 1.01, -3.63],
                   [-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73], [1.39, 3.16, 2.87],
                   [1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [0.45, 1.33, -4.38],
                   [-0.76, 0.84, -1.96],
                   [0.21, 0.03, -2.21], [0.37, 0.28, -1.8], [0.18, 1.22, 0.16],
                   [-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39], [0.74, 0.96, -1.16],
                   [-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [0.44, 1.31, -0.14],
                   [0.46, 1.49, 0.68],
                   [-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [1.55, 0.99, 2.69],
                   [1.86, 3.19, 1.51], [1.68, 1.79, -0.87], [3.51, -0.22, -1.39],
                   [1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [0.25, 0.68, -0.99],
                   [0.66, -0.45, 0.08]])
t_train = np.array([[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],
                    [0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],
                    [0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
print(x_train.shape,t_train.shape)

network = TwoLayerNet(input_size=3, hidden_size=[3,72,3], output_size=3)
#network = TwoLayerNet(input_size=3, hidden_size=3, output_size=3)

iters_num = 1000000
train_size = x_train.shape[0]
batch_size = 10
learning_rate = 0.01

train_loss_list = []
train_acc_list = []
test_acc_list = []

#iter_per_epoch = max(train_size / batch_size, 1)
iter_per_epoch = 1000


for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 梯度
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        #test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        #test_acc_list.append(test_acc)
        print(i, train_acc, loss)
print(network.params)
plt.plot(train_loss_list)
plt.show()


f = open('C:/Users/14775/Desktop/模式识别导论/作业/三/'+'[3723]'+str(iters_num)+'_'+str(learning_rate)+'_'+str(batch_size)+'.txt', 'w')
f.write('train_loss_list\n')
for i in train_loss_list:
    f.write(str(i)+'\n')
f.write('train_acc_list\n')
for i in train_acc_list:
    f.write(str(i)+'\n')
f.close()
