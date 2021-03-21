import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import optimizers, regularizers
from sklearn.metrics import confusion_matrix


def define_model(w=None):
    model = Sequential()
    model.add(Dense(128, input_dim=19, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                    kernel_initializer='random_normal'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    if w != None:
        model.load_weights(w)

    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.00005),
                  metrics=['Recall'])
    return model


def load_data():
    data = pd.read_csv('data.csv')
    train_data, test_data = sklearn.model_selection.train_test_split(data, test_size=0.20, random_state=42)
    train_targets = train_data['Attrition_Flag']
    test_targets = test_data['Attrition_Flag']
    train_data = train_data.drop(columns=['Attrition_Flag'])
    test_data = test_data.drop(columns=['Attrition_Flag'])

    return train_data, train_targets, test_data, test_targets


def train_model():
    train_data, train_targets, test_data, test_targets = load_data()

    model = define_model()

    model.fit(train_data, train_targets, epochs=2000, batch_size=100, verbose=1)

    model.save('model.h5')


def test_model():
    train_data, train_targets, test_data, test_targets = load_data()
    model = define_model(w='model.h5')

    for data, targets, name in ((train_data.to_numpy(), train_targets.to_numpy(), 'train'),
                                (test_data.to_numpy(), test_targets.to_numpy(), 'test')):
        predictions = model.predict(data).round()
        score = sklearn.metrics.recall_score(targets, predictions.round())
        print(name, "recall:", score)
        score = sklearn.metrics.accuracy_score(targets, predictions.round())
        print(name, "accuracy:  ", score)

    np.set_printoptions(precision=2)

    options = [None, 'true']
    for normalize in options:
        conf_mat = confusion_matrix(test_targets, predictions.round(), normalize=normalize)
        print(conf_mat)

    plt.show()


if __name__ == '__main__':
    test_model()
