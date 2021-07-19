import os
import time
import optuna
import pandas as pd
import numpy as np
from keras.backend import clear_session, mean, std
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizer_v2.gradient_descent import SGD
from keras.optimizer_v2.rmsprop import RMSprop
from pandas import DataFrame
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import model_util as mu
np.seterr(divide='ignore', invalid='ignore')

def show_model_evaluation(list, dataset_name):
    print(f'==== {dataset_name} ===')
    print('Accuracy: %.3f (%.3f)' % (np.mean(list[0]), np.std(list[0])))
    print('F1: %.3f (%.3f)' % (np.mean(list[1]), np.std(list[1])))
    print('Sensitivity: %.3f (%.3f)' % (np.mean(list[2]), np.std(list[2])))
    print('Specificity: %.3f (%.3f)' % (np.mean(list[3]), np.std(list[3])))
    #print('AUROC: %.3f (%.3f)' % (np.mean(list[4]), np.std(list[4])))

# This function preform the 3-Fold cross validation and uses Optuna to Optimize the model Hyperparameter - Learning rate.
def inner_k_fold(Xtrain, Ytrain):
    best_score = 0
    best_lr = 0
    trial_duration_list = []
    inner_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=np.random.seed(7))
    for train_index, test_index in inner_kfold.split(Xtrain, Ytrain):
        x_train, x_valid = Xtrain.iloc[train_index, :], Xtrain.iloc[test_index, :]
        y_train, y_valid = Ytrain.iloc[train_index], Ytrain.iloc[test_index]
        # Hyper parameter`s optimization
        study = optuna.create_study(direction="maximize")
        # ToDo - change to 50 trials
        objective = mu.Objective(x_train, y_train, x_valid, y_valid)
        study.optimize(objective, n_trials=5)
        trial = study.best_trial
        trial_duration_list.append(trial.duration)
        if trial.value >= best_score:
            best_score = trial.value
            best_lr = study.best_params['learning_rate']
            best_batch_size = study.best_params['batch_size']
    return best_lr, best_score, objective.get_best_model(), best_batch_size, np.mean(trial_duration_list)

# This function perform the 10-Fold cross validation and according to F1-Scoring choose the best model
def train_parameters(dataset_name, algorithm_name,Xtrain, Ytrain):
    f1_max_score = 0
    final_model = None
    final_lr = 0
    current_fold = 1
    num_of_fold = 2
    evaluation_df = DataFrame()
    model_evaluation = mu.ModelEvaluation(dataset_name, algorithm_name)
    outer_kfold = StratifiedKFold(n_splits=num_of_fold, shuffle=True, random_state=np.random.seed(7))
    for train_index, validate_index in outer_kfold.split(Xtrain, Ytrain):
        x_train, x_valid = Xtrain.iloc[train_index, :], Xtrain.iloc[validate_index, :]
        y_train, y_valid = Ytrain.iloc[train_index], Ytrain.iloc[validate_index]
        best_lr, best_score, best_model, best_batch_size, training_time = inner_k_fold(x_train, y_train)
        print("best_lr" + str(best_lr))
        print("best_score" + str(best_score))
        start = time.time()
        yhat = np.argmax(best_model.predict(x_valid), axis=-1)
        y_valid = y_valid.values
        inference_time = time.time() - start
        f1_model_score = f1_score(y_valid, yhat, average='weighted')
        if f1_model_score > f1_max_score:
            f1_max_score = f1_model_score
            final_model = best_model
            final_lr = best_lr
        model_evaluation.get_result_to_final_table(current_fold, y_valid, yhat, {"learning_rate": best_lr, "batch_size": best_batch_size}, training_time, inference_time)
        resualt = model_evaluation.get_result_df()
        current_fold = current_fold+1
        evaluation_df.append(resualt)
    return final_model, final_lr, evaluation_df

def train_model(X_test_model, Y_test_model, model, num_of_round):
    accuracy_list = []
    f1_scores_list = []
    sensitivity_list = []
    specificity_list = []
    ROC_area_list = []
    final_model = None
    f1_max_score = 0
    for index in range(num_of_round):
        X_train, X_test, Y_train, Y_test = train_test_split(X_test_model, Y_test_model, test_size=0.2, random_state=42)
        model.fit(X_train, Y_train)
        yhat = np.argmax(model.predict(X_test), axis=1)
        # evaloation values
        Y_test = Y_test.values
        accuracy_list.append(accuracy_score(Y_test, yhat))
        f1_model_score = f1_score(Y_test, yhat, average='weighted')
        f1_scores_list.append(f1_model_score)
        sensitivity, specificity, presision = mu.evaluated_by_confusion_matrix(Y_test, yhat)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        #ROC_area_list.append(roc_auc_score(Y_test, yhat, multi_class='ovo',average="weighted"))
        if f1_model_score > f1_max_score:
            f1_max_score = f1_model_score
            final_model = model
    eval_params = [accuracy_list, f1_scores_list, sensitivity_list, specificity_list, ROC_area_list]
    return final_model, eval_params

def main_process(algorithm_name, dataset_name, data):
    last_column_name = data.columns[len(data.columns)-1]
    X, y = data.drop(columns=[last_column_name]), data[last_column_name]
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    final_model, final_lr, evaluation_df = train_parameters(dataset_name, algorithm_name, X_train, Y_train)
    best_model, evaluation_params = train_model(X_test, Y_test, final_model, 3)
    return best_model, evaluation_params, evaluation_df

def discretization(data):
    # convert categorial variables into numeric values
    label_encoder = LabelEncoder()
    discret_vec = data.columns
    for category in discret_vec:
        data[category] = label_encoder.fit_transform(data[category])
    # print("data was successfully discretized")
    return data

if __name__ == "__main__":
    # Load data and preaper it
    total_evaluation_df = DataFrame()
    algorithm_name = 'Basic supervized'
    files_names = os.listdir('./datasets/')
    for file_name in files_names:
        data = pd.read_csv('./datasets/'+file_name)
        data = discretization(data)
        best_model, evaluation_params, evaluation_df = main_process(algorithm_name, file_name , data)
        show_model_evaluation(evaluation_params, file_name)


