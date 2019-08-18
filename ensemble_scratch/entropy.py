import numpy as np
from collections import Counter

def deviation_org(y):
    '''
    分割する際の指標として、分散を用いる
    分割する値の前後で標準偏差がもっとも小さいものを用いることで、目的変数を最も分離したものを考える
    '''
    d = y - y.mean()
    s = d**2
    return np.sqrt(s.mean())

def gini_org(y):
    '''
    クラス分類の際にどのクラスかを判定させる関数
    ジニ不純度の計算.計算が少し遅いver
    '''
    i = y.argmax(axis=1)
    clz = set(i)
    c = Counter(i)
    size = y.shape[0]
    score=0.0
    for val in clz:
        score += (c[val]/size)**2
    return 1.0-score

def gini(y):
    '''
    ジニ不純度の計算.計算が早いver.所属ラベルの配列が二次元配列で与えられることを利用
    ラベル軸ごとに計算せずともテンソルで合計を取れば、データ個数をカウントできる
    '''
    m  = y.sum(axis=0)
    size = y.shape[0]
    e = [(p/size)**2 for p in m]
    return 1.0 - np.sum(e)

def infgain(y):
    '''
    情報利得度の計算.これもテンソルを利用
    分割後に目的変数に含まれる情報量が少なくなるように分割点を求める
    '''
    m = y.sum(axis=0)
    size = y.shape[0]
    e = [(p/size)*np.log2(p/size) for p in m if p != 0.0]
    return -np.sum(e)
