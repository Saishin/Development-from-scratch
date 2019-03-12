import numpy as np
from SVM import SVM

def predict_SVM():
    N = 100
    cls1 = []
    cls2 = []

    mean1 = [-1, 2]
    mean2 = [1, -1]
    cov = [[1.0,0.8], [0.8, 1.0]]

    cls1.extend(np.random.multivariate_normal(mean1, cov, int(N/2)))
    cls2.extend(np.random.multivariate_normal(mean2, cov, int(N/2)))

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
