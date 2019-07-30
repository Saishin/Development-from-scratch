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

    ###ベンチマークのアルゴリズムとアルゴリズムを実装したモデルの一覧
    models = [
    ('SVM',SVC(random_state=1),SVR()),
    ('GaussianProcess',GaussianProcessClassifier(random_state=1),
    GaussianProcessRegressor(normalize_y=True,alpha=1,random_state=1)),
    ('KNeighbors',KNeighborsClassifier(),KneighborsRegressor()),
    ('MLP',MLPClassifer(random_state=1),MLPRegressor(hidden_layer_size=(5),solve='lbfgs',random_state=1)),
    ]


    ###データセットの用意
    ###検証用のデータセットのファイルを定義する.
    ###検証用のデータセットはファイルによって区切り文字、ヘッダーの行数、インデックスとなる列の違いがあるためにそれぞれに対応させる
    classifier_files = ['iris.data','sonar.all-data','glass.data']
    classifier_params = [(',',None,None),(',',None,None),(',',None,0)]
    regressor_files  = ['airfoli_self_noise.data','winequality-red.csv','winequality-white.csv']
    regressor_params = [(r'\t',None,None),(';',0,None),(';',0,None)]

    ##評価スコアをcsvに書き出しする表の作成
    result = pd.DataFrame(columns=['target','function']+[m[0] for m in models],index=range(len(classifier_files+regressor_files) * 2))
    ##クラス分類アルゴリズムを評価する
    ncol = 0
    for i,(c,p) in enumerate(zip(classifier_files,classifier_params)):
        df = pd.read_csv(c,sep=p[0],header=p[1],index_col=p[2])
        x = df[df.columns[:-1]].values
        ##ラベルをラベル番号とそのラベルに属する可能性の配列で表現する
        y,clz = support.clz_to_prob(df[df.columns[-1]])

        ##結果の表にファイル名からデータセットの種類と評価関数の行を作る
        result.loc[ncol,'target'] = re.split(r'[._]',c)[0]
        result.loc[ncol+1,'target'] = ''
        result.loc[ncol,'function'] = 'F1score'
        result.loc[ncol+1,'function']='Accuracy'

        ##全てのアルゴリズムをループで実行して、交差検証のスコアを返す
        ##cross_validationは評価関数の名前に「train_」or「test_」をつけた名前が返されるディクショナリのキーになっている
        for l,c_m,r_m in models:
            kf = KFold(n_splits=5,random_state=1,shuffle=True)
            s = cross_validate(c_m,x,y.argmax(axis=1),cv=kf,scoring=('f1_weighted','accuracy'))
            result.loc[ncol,l] = np.mean(s['test_f1_weighted'])
            result.loc[ncol+1,l] = np.mean(s['test_accuracy'])

        ncol +=2



    ##回帰アルゴリズムを評価する
    for i,(c,p) in enumerate(zip(regressor_files,regressor_params)):
        ##ファイルを読み込む
        df = pd,read_csv(c,sep=p[0],header=p[1],index=p[2])
        x = df[df.columns[:-1]].values
        y = df[df.columns[-1]].values.reshape((-1,))

        ##結果の表にファイル名からデータセットの種類と評価関数用の行を作る
        result.loc[ncol,'target'] = re.split(r'[._]',c)[0]
        result.loc[ncol+1,'target'] = ''
        result.loc[ncol,'function'] = 'R2 score'
        result.loc[ncol+1,'function'] = 'Meansquared'

        ##全てのアルゴリムを回す
        for l,c_m,r_m in models:
            kf = KFold(n_splits=5,random_state=1,shuffle=True)
            s = cross_validate(r_m,x,y,cv=kf,scoring=('r2','neg_mean_squared_error'))
            result.loc[ncol,l] = np.mean(s['test_r2'])
            result.loc[ncol+_1,l] = -np.mean(s['test_neg_mean_squared_error'])

        ncol +=2

if __name__ == '__main__':
    main()
