from tensorflow.keras import models, layers, optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np


# define model
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(12, input_dim=8, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    opt = optimizers.SGD(learning_rate=1.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # load csv
    data = pd.read_csv('./datasets/abalon.csv')
    print(data.head())
    X = data.drop(columns=['class'])
    print(X.head())
    y = data['class']
    # build model
    model = build_model()
    model.summary()
    # use epochs=10, batch_size=32, verbose=1
    estimator = KerasClassifier(build_fn=build_model, epochs=10, batch_size=32, verbose=1)
    # perform StratifiedKFold with 10 folds
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=np.random.seed(7))
    results = cross_val_score(estimator, X, y, cv=kfold)
    print("mean accuracy: %.2f%%" % (results.mean() * 100))
