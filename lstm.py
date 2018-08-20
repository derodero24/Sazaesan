
import Deropy.neural as nrl
import numpy as np
import pandas as pd
from keras.layers import LSTM, Activation, Dense
from keras.models import Sequential

data_file = 'サザエさんじゃんけん.tsv'
look_back = 13  # 遡る時間
res_file = 'lstm'


def shuffle_lists(list1, list2):
    '''リストをまとめてシャッフル'''
    seed = np.random.randint(0, 1000)
    np.random.seed(seed)
    np.random.shuffle(list1)
    np.random.seed(seed)
    np.random.shuffle(list2)


def get_data():
    '''データ作成'''
    df = pd.read_csv(data_file, sep='\t',
                     usecols=['rock', 'scissors', 'paper'])
    dataset = df.values.astype(np.float32)

    X_data, y_data = [], []
    for i in range(len(dataset) - look_back - 1):
        x = dataset[i:(i + look_back)]
        X_data.append(x)
        y_data.append(dataset[i + look_back])

    # X_data = np.array(X_data)
    # y_data = np.array(y_data)
    X_data = np.array(X_data[-500:])
    y_data = np.array(y_data[-500:])
    last_data = np.array([dataset[-look_back:]])

    # シャッフル
    shuffle_lists(X_data, y_data)

    return X_data, y_data, last_data


def get_model():
    model = Sequential()
    model.add(LSTM(16, input_shape=(look_back, 3)))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def pred(model, X, Y, label):
    '''正解率 出力'''
    predictX = model.predict(X)
    correct = []
    for real, predict in zip(Y, predictX):
        if real.argmax() == predict.argmax():
            correct.append(1)
        else:
            correct.append(0)
    print(correct[-20:])
    correct = sum(correct) / len(Y)
    print(label + '正解率 : %02.2f ' % correct)


def main():
    # データ取得
    X_data, y_data, last_data = get_data()

    # データ分割
    mid = int(len(X_data) * 0.7)
    train_X, train_y = X_data[:mid], y_data[:mid]
    test_X, test_y = X_data[mid:], y_data[mid:]

    # 学習
    # model = get_model()
    # hist = model.fit(train_X, train_y, epochs=50, batch_size=16,
    #                  validation_data=(test_X, test_y))
    #
    # nrl.save_model(model, res_file)
    # nrl.save_hist(hist, res_file, label='')
    # nrl.plot_hist(res_file, res_file)
    model = nrl.load_model(res_file)

    # 正解率出力
    pred(model, train_X, train_y, 'train')
    pred(model, test_X, test_y, 'test')

    # 来週の手
    next_hand = model.predict(last_data)
    print(next_hand[0])
    hands = ['グー', 'チョキ', 'パー']
    print('来週の手 : ' + hands[next_hand[0].argmax()])


if __name__ == '__main__':
    main()
