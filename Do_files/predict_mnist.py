import numpy as np
import pandas as pd
from Neural_net_model import NN


def predict_NN():

    mnist_train = np.array(pd.read_csv('/Users/shintaro/Downloads/mnist-in-csv/mnist_train.csv', engine='python'))
    mnist_test = np.array(pd.read_csv('/Users/shintaro/Downloads/mnist-in-csv/mnist_test.csv', engine='python'))

    input_node=784
    hidden_node=100
    output_node=10
    lr_rate=0.3

    model = NN(input_node , hidden_node , output_node , lr_rate)


    epochs = 5

    #データを最大値の255で割り、0~1の範囲にする
    ##train
    for e in range(epochs):
        for data in mnist_train:
            input_data = data[1:]/255
            target = np.zeros(output_node)
            target[data[0]] = 1.0
            model.train(input_data , target)


    ##test
    ok =[]
    for data2 in mnist_test:
        input_data = data2[1:]/255
        target = data2[0]
        output = model.test(input_data)
        pred_label = np.argmax(output)
        if target == pred_label:
            ok.append(1)
        else:
            ok.append(0)

    acc = sum(ok)/len(ok)
    print('Accuracy is {}'.format(acc))

if __name__ =='__main__':
    predict_NN()
