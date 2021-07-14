from tensorflow.keras import models, layers, optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# define model
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(units=5000, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(units=1, kernel_initializer='normal', activation='sigmoid'))
    opt = optimizers.Adam(learning_rate=1.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = build_model()

