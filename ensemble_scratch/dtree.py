import numpy as np
import support
import entropy
from zeror import ZeroRule
from linear import Linear
from dstump import DecisionStump

class DecisionTree(DecisionStump):
    '''
    DecisionStumpを継承して、最大深さと現在の木の深さの変数を作成
    DecisionTreeクラスは決定木内の一つのノードで、葉となるノードを自分自身のクラスで置き換えて、
    可変の木にする
    predictは継承したDecisionStumpのpredictを使用する
    '''
    def __init__(self,max_depth=5,metric=entropy.gini,leaf=ZeroRule,depth=1):
        super().__init__(metric=metric,leaf=leaf)
        self.max_depth = max_depth
        self.depth = depth

    def fit(self,x,y):
        ##左右の葉を作成する。
        self.left = self.leaf()
        self.right = self.leaf()
        ##データを左右に分割する
        left,right = self.split_tree(x,y)
        if self.depth < self.max_depth:
            if len(left) > 0:
                self.left = self.get_node()
            if len(right) > 0:
                self.right = self.get_node()
        ##左右のノードを学習させる
        if len(left) > 0:
            self.left.fit(x[left],y[left])
        if len(right) > 0:
            self.right.fit(x[right],y[right])
        return self

    def get_node(self):
        '''
        新しいノードを作成する
        '''
        return DecisionTree(max_depth = self.max_depth,metric=self.metric,leaf=self.leaf,depth=self.depth+1)

    def split_tree_fast(self,x,y):
        '''
        DecisionStumpのsplit_treeは一つずつの変数に対して、処理を回すため遅い
        numpyのブロードキャストを使って高速化を図る
        '''
        self.feat_index = 0
        self.feat_val = np.inf
        score = np.inf

        ##データの前準備
        ytil = y[:,np.newaxis]
        xindex = np.argsort(x , axis=0)
        ##sortしたxに合わせて、yの並び替え
        ysort = np.take(ytil,xindex,axis=0)
        for f in range(x.shape[0]):
            l = xindex[:f , : ]
            r = xindex[f: , : ]
            ly = ysort[:f , : ,0, : ]
            ry = ysort[f: , : ,0, : ]
            ##全ての次元のスコアを求める
            loss = [self.make_loss(ly[: , yp , :] , ry[: , yp , :] , l[: , yp], r[: , yp] ) if x[xindex[f-1,yp],yp] != x[xindex[f,yp],yp] else np.inf for yp in range(x.shape[1])]
            ##最小のスコアになる次元を求める
            i  = np.argmin(loss)
            if score > loss[i]:
                score = loss[i]
                self.feat_index = i
                self.feat_val = x[xindex[f , i] , i]
        ##実際に分割するindedxを取得
        filter = x[: , self.feat_index] < self.feat_val
        left = np.where(filter)[0].tolist()
        right = np.where(filter == False)[0].tolist()
        self.score = score
        return left,right

    def split_tree(self,x,y):
        ##高速動作の関数でオーバーロード
        return self.split_tree_fast(x , y)

    def print_leaf(self , node , d=0):
        ##分岐条件を明示
        if isinstance(node,DecisionTree):
            return '\n'.join([
                '%s if feat[%d] <= %f then: '%('+'*d , node.feat_index, node.feat_val ),
                self.print_leaf(node.left, d+1),
                '%s else'%('l'*d,),
                self.print_leaf(node.right, d+1)
            ])
        else:
            return '%s %s '%('l'*(d-1),node)

    def __str__(self):
        return self.print_leaf(self)

def main():
    import pandas as pd
    ps = support.get_base_args()
    ps.add_argument('--metric',  '-m' , default='' , help='Metric function')
    ps.add_argument('--leaf' , '-l', default='',help='Leaf class')
    ps.add_argument('--depth' , '-d' , type=int , default=5,help='Max Tree Depth')
    args = ps.parse_args()

    df = pd.read_csv(args.input,  sep=args.separator , header=args.header,  index_col = args.indexcol)
    x = df[df.columns[:-1]].values

    ##分岐関数の指定
    if args.metric == 'div':
        mt  = entropy.deviation_org
    elif args.metric == 'infgain':
        mt = entropy.infgain
    elif args.metric == 'gini':
        mt = entropy.gini
    else:
        mt = None

    ##葉の関数の指定
    if args.leaf == 'zeror':
        lf = ZeroRule
    elif args.leaf == 'linear':
        lf = Linear
    else:
        lf  =None

    ##回帰かクラス分類か
    if not args.regression:
        y,clz = support.clz_to_prob(df[df.columns[:-1]])
        if mt is None:
            mt = entropy.gini
        if lf is None:
            lf = ZeroRule
        plf = DecisionTree(metric=mt , leaf=lf, max_depth=args.depth)
        support.report_classifer(plf,x,y,clz,args.crossvalidate)

    else:
        y = df[df.columns[-1]].values.reshape((-1,1))
        if mt is None:
            mt = entropy.deviation_org
        if lf is None:
            lf = Linear
        plf = DecisionTree(metric=mt  ,leaf=lf , max_depth=args.depth)
        support.report_regressor(plf,x,y,args.crossvalidate)

if __name__ == '__main__':
    main()
