import numpy as np
import pandas as pd
import support

'''
ここでは、説明変数を無視したただ単純に大多数のラベルを返すモデルを作成する
'''

class ZeroRule():
    def __init__(self):
        self.r = None

    def fit(self,x,y):
        selr.r = np.mean(y,axis=0)
        return self.r

    def predict(self,x):
        ##データ数x予測ラベルの(1,)の型を作って、実際のラベルをそこに代入
        ##shape変換でも対応できそう
        z = np.zeros(len(x),self.r.shape[0])
        return z+self.r

    def __str__(self):
        return str(self.r)

if __name__ == '__main__':
    ps = support.get_base_args()
    args = ps.parse_args()

    df = df
