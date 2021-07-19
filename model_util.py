from keras import Sequential
from keras.backend import clear_session
from keras.layers import Dense
from keras.optimizer_v2.gradient_descent import SGD
from pandas import DataFrame
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score, roc_curve
import pandas as pd
import numpy as np



class Objective(object):
    def __init__(self, num_of_epochs):
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.best_model = None
        self.num_of_epochs = num_of_epochs
        self.num_of_class = 0
        self.max_score = 0
        self.loss_function = ""

    def __call__(self, trial):
        # Clear clutter from previous Keras session graphs.
        clear_session()
        if self.best_model is None:
            model = self.get_basic_model()
        else:
            model = self.best_model
        # We compile our model with a sampled learning rate - Note:The optimizer RMSprop get better result
        learning_rate = trial.suggest_float("learning_rate", 1.0, 2.0, log=True)
        batch_size = trial.suggest_int("batch_size", 32, 256, log=True)
        model.compile(loss=self.loss_function, optimizer=SGD(learning_rate=learning_rate), metrics=["accuracy"])
        model.fit(self.x_train, self.y_train, validation_data=(self.x_valid, self.y_valid),
                    shuffle=True, batch_size=batch_size, epochs= self.num_of_epochs, verbose=False)

        # Evaluate the model accuracy on the validation set.
        score = model.evaluate(self.x_valid, self.y_valid, verbose=0)
        if score[1] >= self.max_score:
            self.best_model = model
        return score[1]

    def get_basic_model(self):
        model = Sequential()
        model.add(Dense(12, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(10, activation="sigmoid"))
        if self.num_of_class == 2:
            model.add(Dense(1, activation="sigmoid"))
            self.loss_function = "binary_crossentropy"
        else:
            model.add(Dense(self.num_of_class, activation="softmax"))
            self.loss_function = "sparse_categorical_crossentropy"
        return model

    def set_best_model(self, model):
        self.best_model = model

    def set_num_of_classes(self, num):
        self.num_of_class = num

    def set_params(self, x_train, y_train, x_valid, y_valid,):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

    def get_best_model(self):
        return self.best_model


class ModelEvaluation(object):
    def __init__(self, dataset_name, algorithm_name):
        self.dataset_name = dataset_name
        self.algorithm_name = algorithm_name
        self.df_colunms = ['Dataset Name', 'Algorithm Name', 'Cross Validation index', 'Learning_rate',
                           'Batch_size', 'Accuracy', 'TPR', 'FPR', 'Precision', 'AUC', 'PR-Curve', 'Training Time', 'Inference Time']
        self.results_df = DataFrame(columns = self.df_colunms)

    def get_result_to_final_table(self, fold_num, y_valid, yhat, y_proba, learning_r, batch_size, training_time, inference_time):
        binary_class = len(np.unique(y_valid))==2
        if binary_class:
            TN, FP, FN, TP = confusion_matrix(y_valid, yhat).ravel()
            AUC_temp = roc_auc_score(y_valid, y_proba)
            PR_Curve_temp = average_precision_score(y_valid, y_proba)
        else:
            confusion_matrix_res = confusion_matrix(y_valid, yhat)
            FP = confusion_matrix_res.sum(axis=0) - np.diag(confusion_matrix_res)
            FN = confusion_matrix_res.sum(axis=1) - np.diag(confusion_matrix_res)
            TP = np.diag(confusion_matrix_res)
            TN = confusion_matrix_res.sum() - (FP + FN + TP)
            FP = np.mean(FP.astype(float))
            FN = np.mean(FN.astype(float))
            TP = np.mean(TP.astype(float))
            TN = np.mean(TN.astype(float))
            AUC_temp = roc_auc_score(y_valid, y_proba, multi_class='ovr')
            y_valid_hot = pd.get_dummies(y_valid.astype(str))
            PR_Curve_temp = average_precision_score(y_valid_hot, y_proba)

        accuracy = accuracy_score(y_valid, yhat)
        TPR = round(TP/(TP+FN), 3)
        FPR = round(FP/(FP+TN), 3)
        presision = round(TP/(TP+FP), 3)
        AUC = round(AUC_temp, 3)
        PR_Curve = round(PR_Curve_temp, 3)
        row = pd.Series([self.dataset_name, self.algorithm_name, fold_num, learning_r, batch_size, accuracy, TPR, FPR, presision,
                         AUC, PR_Curve, training_time, inference_time], index = self.df_colunms)
        self.results_df = self.results_df.append(row, ignore_index=True)

    def get_result_df(self):
        return self.results_df



def evaluated_by_confusion_matrix(Y_test, yhat):
    matrix = confusion_matrix(Y_test, yhat)
    tn = matrix[0][0]
    fp = matrix[0][1]
    fn = matrix[1][0]
    tp = matrix[1][1]
    return tp/float(tp + fn), fp/float(tn + fp), tp/float(tp + fp)
