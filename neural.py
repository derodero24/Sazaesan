
import Deropy.neural as nrl
import numpy as np
import pandas as pd
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential

data_file = 'サザエさんじゃんけん.tsv'
res_file = 'neural'


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
    X_data = [[0, 0, 0]]
    for row in df.values:
        data = [d + 1 for d in X_data[-1]]
        data[row.argmax()] = 0
        X_data.append(data)

    # numpy.array型に変換
    # X_data = np.array(X_data[:-1])
    # y_data = np.array(df.values)
    X_data = np.array(X_data[-501:-1])
    y_data = np.array(df.values[-500:])

    last_data = np.array(X_data[-1:])

    # 正規化
    X_data = X_data.astype(np.float32)
    last_data = last_data.astype(np.float32)
    X_data /= 255.0
    last_data /= 255.0

    # シャッフル
    shuffle_lists(X_data, y_data)

    return X_data, y_data, last_data


def get_model():
    '''モデルを構築'''
    model = Sequential()
    model.add(Dense(32, input_shape=(3,)))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
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
    X_data, y_data, last_data = get_data()

    # データ分割
    mid = int(len(y_data) * 0.7)
    train_X, train_y = X_data[:mid], y_data[:mid]
    test_X, test_y = X_data[mid:], y_data[mid:]

    # 学習
    # model = get_model()
    # hist = model.fit(train_X, train_y, epochs=200, batch_size=8,
    #                  validation_data=(test_X, test_y))

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
