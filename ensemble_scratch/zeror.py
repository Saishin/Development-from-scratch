import numpy as np
import pandas as pd
import support

'''
ここでは、説明変数を無視したただ単純に大多数のラベルを返すモデルを作成する
実行するときは、
python zeror.py -i iris.data
のようにsupportで書いたオプションの後に指定していくことで、オプションを指定していく。
'''

class ZeroRule():
    def __init__(self):
        self.r = None

    def fit(self,x,y):
        selr.r = np.mean(y,axis=0)
        return self.r

    def predict(self,x):
        ##データ数x予測ラベルの(-1,1)の型を作って、実際のラベルをそこに代入
        z = np.zeros((len(x),self.r.shape[0]))
        return z+self.r

    def __str__(self):
        return str(self.r)

if __name__ == '__main__':
    ps = support.get_base_args()
    args = ps.parse_args()

    df = pd.read_csv(args.input, sep=args.separator,header=args.header,index=args.indexcol)
    x = df[df.columns[:-1]].values

    if not args.regression:
        y,clz = support.clz_to_prob(df[df.columns[-1]])
        plf = ZeroRule()
        support.report_classifer(plf,x,y,clz,args.crossvalidate)

    else:
        y = df[df.columns[-1]].values.reshape((-1,1))
        plf = ZeroRule()
        support.report_regressor(plf,x,y,args.crossvalidate)
