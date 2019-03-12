import numpy as np

class NN():
    '''
    3層のNNにおいて活性化関数をシグモイド関数として実装
    '''

    def __init__(self , input_node , hidden_node , output_node, lr_rate):
        '''
        各レイヤーのノード数、重み、学習率、活性化関数の設定
        '''

        self.input_node  = input_node
        self.hidden_node = hidden_node
        self.output_node  = output_node

        ##各レイヤー間の重みの設定
        ##各重みを(output , input)のshapeで定義しているのは、error更新の際に重みの行列を転置して、各ノードの誤差計算をするため
        self.between_input_hidden = np.random.normal(.0 , np.sqrt(2/self.input_node) ,(self.hidden_node , self.input_node) )
        self.between_hidden_output = np.random.normal( .0 , np.sqrt(2/self.hidden_node)  ,(self.output_node , self.hidden_node))

        ##学習率の設定
        self.lr_rate = lr_rate

        ##活性化関数の設定
        self.activation_func = lambda x: 1/ (1 + np.exp(-x))



    def train(self, train_data , target):
        '''
        入力、隠れ層、出力層、各レイヤー間の重みの更新
        '''
        ##入力の行列変換
        inputs = np.array(train_data , ndmin=2).T
        targets = np.array(target , ndmin=2).T

        ##隠れ層へのinputの計算
        hidden_inputs = np.dot(self.between_input_hidden , inputs)
        ##隠れ層のactivate function
        hidden_outputs = self.activation_func(hidden_inputs)

        ##出力層へのinputの計算
        final_inputs = np.dot(self.between_hidden_output , hidden_outputs)
        ##出力層のactivate function
        final_outputs = self.activation_func(final_inputs)

        ##誤差計算
        output_errors = targets  - final_outputs

        ##誤差から出力層と隠れ層の重みを更新
        hidden_errors  = np.dot(self.between_hidden_output.T  , output_errors)
        self.between_hidden_output += self.lr_rate*np.dot((output_errors*final_outputs*(1.0 - final_outputs)) , hidden_outputs.T)

        ##誤差から隠れ層と入力層の重みを更新
        self.between_input_hidden += self.lr_rate*np.dot((hidden_errors*hidden_outputs*(1.0 - hidden_outputs)) , inputs.T)

        return self

    def test(self , test_data):
        '''
        各レイヤーからの出力値を計算して、確率値の出力を得る。そして、その最大値のindexを分類とする
        '''

        ##入力の行列変換
        inputs = np.array(test_data  ,ndmin=2).T

        ##隠れ層への入力の計算
        hidden_inputs = np.dot(self.between_input_hidden , inputs)
        ##隠れ層のactivate function
        hidden_outputs = self.activation_func(hidden_inputs)

        ##出力層へのinputの計算
        final_inputs = np.dot(self.between_hidden_output , hidden_outputs)
        ##出力層のactivate function
        final_outputs = self.activation_func(final_inputs)

        return final_outputs


if __name__ =='__main__':
    NN()
