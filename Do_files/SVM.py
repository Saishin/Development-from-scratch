import numpy as np
import cvxopt
from cvxopt import matrix


class SVM():

    def fit(self, X ,T):

        '''
         二次計画法を定義して解く
        '''
        ##行列作成
        m = np.zeros((X.shape[0] , X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                m[i , j] = T[i] * T[j] * np.dot(X[i] , X[j])

        ##行列を定義する
        P = matrix(m)
        q = matrix(np.array([-1.0 for i in range(X.shape[0])]))
        G = matrix(-1.0*np.eye(X.shape[0]))
        h = matrix(np.array([.0 for i in range(X.shape[0])]))
        A = matrix(T , (1,T.shape[0]))
        b = matrix(0.0)


        ##二次計画法で解く。
        sol=cvxopt.solvers.qp(P,q,G,h,A,b)
        a = np.array(sol['x'])

        # サポートベクトルのインデックスを抽出
        S = []
        for i in range(len(a)):
            if a[i] != 0:
                S.append(i)

        # wを計算
        self.w = np.zeros(2)
        for n in S:
            self.w += a[n] * T[n] * X[n]

        # w0を計算
        sums = 0
        for n in S:
            temp = 0
            for m in S:
                temp += a[m] * T[m] * np.dot(X[n], X[m])
            sums += (T[n] - temp)
        self.w0 = sums / len(S)

        return self

if __name__ =='__main__':
    SVM()
