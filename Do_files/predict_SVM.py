import numpy as np
from SVM import SVM

def predict_SVM():
    
    '''
    データ作成とモデルフィッティング
    '''
    N = 2000
    cls1 = np.random.randn(1000,2)
    cls2 = np.random.randn(1000,2)+np.array([5,5])

    # データ行列Xを作成
    X = np.vstack((cls1, cls2))
    T =[]
    for i in range(int(N/2)):
        T.append(1.0)

    for i in range(int(N/2)):
        T.append(-1.0)
    T = np.array(T)

    ##モデルフィッティング
    model = SVM()
    model.fit(X , T)
    pred_list =  np.sign(model.w0 + np.dot(X , model.w))

    ##predict
    ok = []
    for i in range(len(X)):
        if T[i] == pred_list[i]:
            ok.append(1)

        else:
            ok.append(0)

    acc_SVM = np.sum(ok)/len(ok)
    print('Accuracy is {}'.format(acc_SVM))

if __name__ == '__main__':
    predict_SVM()
