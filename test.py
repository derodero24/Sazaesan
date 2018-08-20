
# import random
#
# import numpy as np
#
#
# a = [['a', 'c', 'xy'], ['r', 'a', 'n'], ['d', 'o', 'm'], ['r', 'a', 'a'], [1, 0, 2]]
# b = [[0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]]
#
#
# def shuffle_lists(list1, list2):
#     '''リストをまとめてシャッフル'''
#     # seed = random.randint(0, 100)
#     # random.seed(seed)
#     # random.shuffle(list1)
#     # random.seed(seed)
#     # random.shuffle(list2)
#     seed = np.random.randint(0, 1000)
#     np.random.seed(seed)
#     np.random.shuffle(list1)
#     np.random.seed(seed)
#     np.random.shuffle(list2)
#
# # shuffle_lists(a[:-1], b)
#
# seed = np.random.randint(0, 100)
# np.random.seed(seed)
# a[:-1] = np.random.shuffle(a[:-1])
# np.random.seed(seed)
# np.random.shuffle(list2)
#
#
# print(a)
# print(b)


import numpy as np
import pandas as pd
from keras.layers import Activation, Dense
from keras.models import Sequential

data_file = 'サザエさんじゃんけん.tsv'
res_file = 'small_neural'


def get_data():
    '''データ作成'''
    df = pd.read_csv(data_file, sep='\t',
                     usecols=['rock', 'scissors', 'paper'])
    X_data = [[0, 0, 0]]
    for row in df.values:
        data = [d + 1 for d in X_data[-1]]
        data[row.argmax()] = 0
        X_data.append(data)

    X_data = np.array(X_data)
    y_data = np.array(df.values)

    return X_data, y_data


def get_model():
    '''モデルを構築'''
    model = Sequential()
    model.add(Dense(16, input_shape=(3,)))
    model.add(Activation('relu'))
    model.add(Dense(16))
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
    correct = 0
    for real, predict in zip(Y, predictX):
        if real.argmax() == predict.argmax():
            correct += 1
    correct = correct / len(Y)
    print(label + '正解率 : %02.2f ' % correct)


def main():
    X_data, y_data = get_data()

    # 正規化
    X_data = X_data.astype(np.float32)
    X_data /= 255.0

    # データ分割
    mid = int(len(y_data) * 1)
    train_X, train_y = X_data[:mid], y_data[:mid]
    test_X, test_y = X_data[mid:-1], y_data[mid:]

    # 学習
    model = get_model()
    hist = model.fit(train_X, train_y, epochs=100, batch_size=16,
              validation_data=(test_X, test_y))

    # 正解率出力
    # pred(model, train_X, train_y, 'train')
    # pred(model, test_X, test_y, 'test')

    # 来週の手
    next_hand = model.predict(X_data[-1:])
    print(next_hand[0])
    hands = ['グー', 'チョキ', 'パー']
    print(hands[next_hand[0].argmax()])


if __name__ == '__main__':
    main()
