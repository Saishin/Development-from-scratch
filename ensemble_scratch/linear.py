import numpy as np
import support

class Linear:
    def __init__(self,epochs=20,lr=0.01,earlystop=None):
        '''
        epoch:エポック回数
        lr:学習率
        earlystop:学習の停止設定
        beta,norm:学習データを正則化するためそれを元のデータの尺度に戻すため
        '''
        self.epochs = epochs
        self.lr = lr
        self.earlystop = earlystop
        self.beta = None
        self.norm = None

    def fitnorm(self,x,y):
        '''
        学習の前にデータに含まれる値の範囲を0-1に変換するためにデータから、最小と最大を取得
        '''
        self.norm = np.zeros((x.shape[1]+1,2))
        self.norm[0,0] = np.min(y)
        self.norm[0,1] = np.max(y)
        self.norm[1:,0] = np.min(x,axis=0)
        self.norm[1:,1] = np.max(x,axis=0)

    def normalize(self,x,y=None):
        '''
        データを正則化する
        特徴量の次元数分正則化の分母も必要になるため最初にmax-minで次元数分の配列を作成して利用
        '''
        l = self.norm[1:,1]-self.norm[1:,0]
        l[l==0] = 1
        p  = (x - self.norm[1:,0])/l
        q = y
        if y is not None and not self.norm[0,1] ==self.norm[0,0]:
            q = (y - self.norm[0,0])/(self.norm[0,1]-self.norm[0,0])
        return p, q

    def r2(self,y,z):
        ##earlystop用にr2スコアの計算
        y = y.reshape((-1,))
        z = z.reshape((-1,))
        mn = ((y-z)**2).sum(axis=0)
        dn  =((y - y.mean())**2).sum(axis=0)
        if dn ==0:
            return np.inf
        return 1.0-mn/dn

    def fit(self,x,y):
        '''
        勾配降下法でのパラメータ推定
        最初にデータの正則化を行う
        '''
        self.fitnorm(x,y)
        x,y = self.normalize(x,y)
        ##切片を係数項に含める
        self.beta = np.zeros((x.shape[1]+1,))
        ##学習
        for _ in range(self.epochs):
            ##各レコードのデータをfitさせていく
            for p,q in zip(x,y):
                z = self.predict(p.reshape((1,-1)),normalized=True)
                z = z.reshape((1,))
                err = (z - q)*self.lr
                delta = p*err
                ##パラメータ更新
                self.beta[0] -= err
                self.beta[1:] -= delta
            if self.earlystop is not None:
                z = self.predict(x,normalized=True)
                s = self.r2(y,z)
                if self.earlystop <= s:
                    break
        return self

    def predict(self,x,normalized=False):
        '''
        線形回帰のモデル実行
        まずは値の範囲を0-1に変換
        '''
        if not normalized:
            x,_ = self.normalize(x)
        ##全データの長さに対応した配列を作って、それぞれに切片項を足す
        z = np.zeros((x.shape[0],1))+self.beta[0]
        for i in range(x.shape[1]):
            ##各変数に対応する係数をかける
            c = x[:,i]*self.beta[i+1]
            ##切片項と変数とパラメータのかけたものを足して、線形回帰を作成
            z += c.reshape((-1,1))
        ##0-1に正則化した値を元に戻す
        if not normalized:
            z = z *(self.norm[0,1] - self.norm[0,0])+self.norm[0,0]
        return z

    def __str__(self):
        ##モデル内容の表示
        if type(self.beta) is not type(None):
            s = ['%f'%self.beta[0]]
            e = [' +feat[%d]*%f'%(i+1,j) for i,j in enumerate(self.beta[1:])]
            s.extend(e)
            return ''.join(s)
        else:
            return '0.0'

def main():
    import pandas as pd
    ps = support.get_base_args()
    ps.add_argument('--epochs','-p',type=int,default=20,help='Num of Epochs')
    ps.add_argument('--learningrate','-l',type=float,default=0.01,help='Learning rate')
    ps.add_argument('--earlystop','-a',action='store_true',help='Early stopping')
    ps.add_argument('--stoppingvalue','-v',type=float,default=0.01,help='Early stopping value')
    args = ps.parse_args()

    df = pd.read_csv(args.input,sep=args.separator,header=args.header,index_col=args.indexcol)
    x = df[df.columns[:-1]].values

    if not args.regression:
        print('Not support')
    else:
        y = df[df.columns[-1]].values.reshape((-1,1))
        if args.earlystop:
            plf  =Linear(epochs=args.epochs,lr=args.learningrate,earlystop=args.stoppingvalue)
        else:
            plf = Linear(epochs=args.epochs,lr=args.learningrate)
        support.report_regressor(plf,x,y,args.crossvalidate)



if __name__ == '__main__':
    main()
