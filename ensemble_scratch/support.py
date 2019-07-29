import numpy as np


'''
データセットとしては、UCIのデータセットから、6つのデータセットをダウンロード
クラス分類用に3つ、回帰ように3つのデータセットを使用して、アルゴリズム評価を行う。
'''

def clz_to_prob(clz, scratch=True):
    '''
    正解ラベルを受け取って、正解ラベルのワンホット配列の作成
    クラスのらラベル名も同時に返す。
    '''
    if scratch:
        ##完全にスクラッチで書くver
        l = list(set(clz))
        m = [l.index for c in clz]
        z = np.zeros((len(clz),len(l)))
        for i,j in enumerate(m):
            z[i,j] = 1.0
        return z, list(map(str,l))

    else:
        ##ライブラリを使う時
        l = list(set(clz))
        z = np.identity(len(set(clz)))[clz]
        return z, list(map(str,l))

def prob_to_clz(pred,cl):
    '''
    pred:予測クラス確率
    cl:正解ラベルのリスト
    モデルのクラス出力の確率からクラスラベルを作成する
    '''
    max_index = pred.argmax(axis=1)
    return [cl[z] for z in max_index]

def get_base_args():
    '''
    オプション引数を自作で定義する
    引数はデータファイルの名前と、ファイルの区切り文字、ヘッダーの行とindex,回帰かどうか、交差検証を行うかどうか
    -i:データファイルの名前
    -s:ファイルの区切り文字
    -e:ヘッダーの行
    -x:ヘッダーのindex
    -r:回帰かどうか
    -c:交差検証を行うかどうか
    '''
    import argparse

    ps = argparse.ArgumentParse(description='ML Test')
    ps.add_argument('--input data','-i',help='Train File')
    ps.add_argument('--separator','-s',help='CSV separator')
    ps.add_argument('--heder row','-e',help='CSV header')
    ps.add_argument('--header index','-x',help='CSV index_col')
    ps.add_argument('--regression','-r',help='Regression')
    ps.add_argument('--crossvalidate','-c',help='use cross validation')
    retrun ps

def report_classifer(plf,x,y,clz,cv=Ture):
    '''
    plf:モデル
    x:input data
    y:target data
    clz:クラス名
    cv:クロスバリデーションの有無
    n:各クロスバリデーションの結果をデータセット量に応じて、重み付け
    '''
    import warnings
    from sklearn.metrics import classification_report,f1_score,accuracy_score
    from sklearn.exceptions import UnderfinedMetricWarning
    from sklearn.model_selection import KFold
    if not cv:
        ##クロスバリデーションなしの時のモデルスコア
        plf.fit(x,y)
        print('Model')
        print(str(plf))
        z = plf.predict(x)
        z = z.argmax(axis=1)
        y = y.argmax(axis=1)
        with warnings.cath_warnings():
            warnings.simplefiter('ignore',category=UnderfinedMetricWarning)
            ##各ラベルごとの評価指標を見る
            report = classification_report(y,z,target_names=clz)
        print('Train score')
        print(report)

    else:
        ##交差検証のスコアを示す
        ##分割したデータの個数で平均を取り、その値で評価
        kf = KFold(n_split = 10,random_state=1,shuffle=True)
        f1=[]
        acc=[]
        n=[]
        for train_index,test_index in kf.split(x):
            x_train,x_test = x[train_index],x[test_index]
            y_train,y_test = y[train_index],y[test_index]
            plf.fit(x_train,y_train)
            z = plf.predict(x_test)
            z = z.argmax(axis=1)
            y_test = y_test.argmax(axis=1)
            f1.append(f1_score(y_test,z,average='weighted'))
            acc.append(accuracy_score(y_test,z))
            n.append(len(x_test/len(x)))
        print('CV score')
        print('f1_score = %f'%(np.average(f1,weights=n)))
        print('Accuracy score%f'%(np.average(acc,weights=n)))

def report_regression(plf,x,y,cv=True):
    '''
    回帰を行うための評価関数
    n:各クロスバリデーションの結果をデータセット量に応じて、重み付け
    '''
    from sklearn.metrics import r2_score,explained_variance_score,mean_absolute_error,mean_squared_error
    from sklearn.model_selection import KFold
    if not cv:
        ##交差検証なしのモデリングコード
        plf.fit(x,y)
        print('Model')
        print(str(plf))
        z = plf.predict(x)
        print('Train score')
        r2 = r2_score(y,z)
        print('R2 score:%f'%r2)
        ev = explained_variance_score(y,z)
        print('explained_variance_score:%f'%ev)
        mean_ab_error = mean_absolute_error(y,z)
        print('mean absolute error:%f'%mean_ab_error)
        mean_sq_error = mean_squared_error(y,z)
        print('mean_squared_error:%f'%mean_sq_error)

    else:
        ##交差検証ありのモデリングコード
        kf = KFold(n_split=10,random_state=1,shuffle=True)
        r2=[]
        mean_sq=[]
        n=[]
        for train_index,test_index in kf.split(x):
            x_train,x_test = x[train_index],x[test_index]
            y_train,y_test = y[train_index],y[test_index]
            plf.fit(x_train,y_train)
            z = plf.predict(x_test)
            r2.append(r2_score(y_test,z))
            mean_sq.append(mean_squared_error(y_test,z))
            n.append(len(x_test)/len(x))
        print('CV score')
        print('R2 score = %f'%np.average(r2,weights=n))
        print('mean squared error = %f'%np.average(mean_sq,weights=n))
