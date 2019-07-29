import re
import numpy as np
import pandas as pd
import support
from sklearn.model_selection import KFold,cross_validate
from sklearn.svm import SVC,SVR
from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
from sklearn.neighbors import KNeighborsClassifier,KneighborsRegressor
from sklearn.neural_network import MLPClassifer,MLPRegressor

'''
ベンチマークのモデル作成コード
使うモデルは、SVM,SVC,ガウシアンプロセスのクラス分類と回帰
k近傍法のクラス分類と回帰
MLPのクラス分類と回帰
'''

def main():
    '''
    ベンチマークのアルゴリズムとアルゴリズムを実装したモデルの一覧
    '''
    models = [
    ('SVM',SVC(random_state=1),SVR()),
    ('GaussianProcess',GaussianProcessClassifier(random_state=1),
    GaussianProcessRegressor(normalize_y=True,alpha=1,random_state=1)),
    ('KNeighbors',KNeighborsClassifier(),KneighborsRegressor()),
    ('MLP',MLPClassifer(random_state=1),MLPRegressor(hidden_layer_size=(5),solve='lbfgs',random_state=1)),
    ]

    '''
    データセットの用意
    検証用のデータセットのファイルを定義する.
    検証用のデータセットはファイルによって区切り文字、ヘッダーの行数、インデックスとなる列の違いがあるためにそれぞれに対応させる
    '''
    

if __name__ == '__main__':
