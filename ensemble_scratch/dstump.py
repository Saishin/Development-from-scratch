import numpy as np
import support
import entropy
from zeror import ZeroRule
from linear import Linear

class DecisionStump:
    '''
    最もシンプルな決定木.ノード一つと葉が2つの深さ1の決定木
    決定木の繰り返される部分
    '''
    def __init__(self,metric = entropy.gini, leaf = ZeroRule):
        '''
        metric:使用するmetric関数の定義
        leaf:葉のモデル
        left,right:葉の左右のインスタンス
        feat_index,feat_val:分割に使用する目的変数の次元の位置と変数
        score:分割の際のmetric関数の値
        '''
        self.metric = metric
        self.leaf = leaf
        self.left = None
        self.right = None
        self.feat_index = 0
        self.feat_val = np.nan
        self.score = np.nan

    def make_split(self,feat,val):
        '''
        入力されたデータを葉に分割させるための関数
        featをval以下と以上で分割するindexを返す
        '''
        left,right = [],[]
        for i,v in enumerate(feat):
            if v < val:
                left.append(i)
            else:
                right.append(i)
        return left,right

    def make_loss(self,y1,y2,l,r):
        '''
        分割した後のスコア計算する関数
        y1,y2が左右に分割した目的変数。l,rがその全体の中でのindex
        yをy1,y2に分割際のmetrics関数の重み付き合計を返す
        '''
        if y1.shape[0] == 0 or y2.shape[0] == 0:
            return np.inf
        total = y1.shape[0]+y2.shape[0]
        m1 = self.metric(y1)*(y1.shape[0]/total)
        m2 = self.metric(y2)*(y2.shape[0]/total)
        return m1+m2

    def split_tree(self,x,y):
        '''
        データを分割して、左右の枝に属するインデックスを返す
        '''
        self.feat_index = 0
        self.feat_val = np.inf
        score = np.inf
        ##左右のindex
        left,right = list(range(x.shape[0])),[]
        ##特徴量の全てで最もよく分割する値を探す
        for i in range(x.shape[1]):
            feat = x[:,i]
            for val in feat:
                l,r = self.make_split(feat,val)
                loss  = self.make_loss(y[l],y[r],l,r)
                if score > loss:
                    score = loss
                    left = l
                    right = r
                    self.feat_index = i
                    self.feat_val = val
        ##全特徴量のうちで分割として最も良いもの
        self.score = score
        return left,right

    def fit(self,x,y):
        '''
        データを左右に振り分けて、それぞれの葉の学習を行う。
        leafには指定モデルが入っている
        '''
        ##左右の葉を作成
        self.left = self.leaf()
        self.right = self.leaf()
        ##データを左右の葉に振り分け
        left,right = self.split_tree(x,y)
        ##左右の葉で学習
        if len(left) > 0:
            self.left.fit(x[left],y[left])
        if len(right) > 0:
            self.right.fit(x[right],y[right])
        return self

    def predict(self,x):
        '''
        深さ1の決定木だから全特徴量から最も分割性能が良いものだけを使用
        '''
        feat = x[:,self.feat_index]
        val = self.feat_val
        l,r = self.make_split(feat,val)
        ##左右の葉を作って、それぞれの葉で予測
        z = None
        if len(l) > 0 and len(r) > 0:
            left = self.left.predict(x[l])
            right = self.right.predict(x[r])
            z = np.zeros((x.shape[0],left.shape[1]))
            z[l] = left
            z[r] = right
        elif len(l) >0 :
            ##全てleftの葉の時
            z = self.left.predict(x)
        elif len(r) > 0:
            ##全てrightの葉の時
            z = self.right.predict(x)
        return z

    def __str__(self):
        ##深さ1のdecisionstumpの分割ルールの表示
        return '\n'.join([
        ' if feat[%d] <= %f then :'%(self.feat_index,self.feat_val),
        ' %s '%(self.left,),
        ' else',
        ' %s '%(self.right,)
        ])

def main():
    import pandas as pd
    '''
    実行時に引数を取れるように設定
    '''
    ps = support.get_base_args()
    ps.add_argument('--metric','-m',default='',help='Metric function')
    ps.add_argument('--leaf','-l',default='',help='Leaf class')
    args = ps.parse_args()

    df = pd.read_csv(args.input,sep=args.separator,header=args.header,index_col=args.indexcol)
    x = df[df.columns[:-1]].values

    ##argsの引数によってどの関数を使うか定義
    ##metric
    if args.metric == 'div':
        mt = entropy.deviation_org
    elif args.metric == 'infgain':
        mt = entropy.infgain
    elif args.metric == 'gini':
        mt = entropy.gini
    else:
        mt = None

    ##leaf
    if args.leaf == 'zeror':
        lf = ZeroRule
    elif args.leaf == 'linear':
        lf = Linear
    else:
        lf = None

    ##回帰かクラス分類か
    if not args.regression:
        ##クラス分類
        y,clz = support.clz_to_prob(df[df.columns[-1]])
        if mt is None:
            mt = entropy.gini
        if lf is None:
            lf = ZeroRule
        plf = DecisionStump(metric=mt,leaf=lf)
        support.report_classifer(plf,x,y,clz,args.crossvalidate)

    else:
        y = df[df.columns[-1]].values.reshape((-1,1))
        if mt is None:
            mt = entropy.deviation_org
        if leaf is None:
            lf = Linear
        plf = DecisionStump(metric=mt,leaf=lf)
        support.report_regressor(plf,x,y,args.crossvalidate)

if __name__ == '__main__':
    main()
