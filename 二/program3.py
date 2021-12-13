import numpy as np 
import copy

def MSE_multi(wi):
    w_i = copy.deepcopy(wi)    

    #回归值矩阵y,增广X矩阵
    y = np.zeros((len(w_i), len(w_i)*len(w_i[0])))
    X = []
    for idx, i in enumerate(w_i):
        for j in i: 
            j.append(1)
            X.append(j)
        y[idx, idx*len(w_i[0]):(idx+1)*len(w_i[0])] = 1

    X = np.array(X).T
    il = X.shape[0]
    lad = 1e-10
    # 计算W
    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(X, X.T))+lad*np.identity(il), X), y.T)

    return W

def test(w_test, W):

    w_t = copy.deepcopy(w_test)
    f_cnt = 0
    for idx_i, i in enumerate(w_t):
        for idx_j, j in enumerate(i):
            j.append(1)
            j = np.array(j)
            #决策
            if np.argmax(np.matmul(W.T, j)) != idx_i: f_cnt += 1

    f_ratio = f_cnt / ((idx_i+1)*(idx_j+1))

    return 1-f_ratio




if __name__ == "__main__":
    w_1 = [[0.1, 1.1], [6.8, 7.1], [-3.5, -4.1], [2.0, 2.7], [4.1, 2.8], [3.1, 5.0], [-0.8, -1.3], [0.9, 1.2], [5.0, 6.4], [3.9, 4.0]]
    w_2 = [[7.1, 4.2], [-1.4, -4.3], [4.5, 0.0], [6.3, 1.6], [4.2, 1.9], [1.4, -3.2], [2.4, -4.0], [2.5, -6.1], [8.4, 3.7], [4.1, -2.2]]
    w_3 = [[-3.0, -2.9], [0.5, 8.7], [2.9, 2.1], [-0.1, 5.2], [-4.0, 2.2], [-1.3, 3.7], [-3.4, 6.2], [-4.1, 3.4], [-5.1, 1.6], [1.9, 5.1]]
    w_4 = [[-2.0, -8.4], [-8.9, 0.2], [-4.2, -7.7], [-8.5, -3.2], [-6.7, -4.0], [-0.5, -9.2], [-5.3, -6.7], [-8.7, -6.4], [-7.1, -9.7], [-8.0, -6.3]]
    # Train
    wi = [w_1[:8], w_2[:8], w_3[:8], w_4[:8]]
    W = MSE_multi(wi)
    print (W)
    # Test
    w_test = [w_1[8:], w_2[8:], w_3[8:], w_4[8:]]
    f_ratio = test(w_test, W)
    print (f_ratio)