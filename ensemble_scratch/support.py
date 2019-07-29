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
    ps.add_argument('--heder row','-e',help='header row')
    ps.add_argument('--header index','-x',help='header index')
    ps.add_argument('--reg')
