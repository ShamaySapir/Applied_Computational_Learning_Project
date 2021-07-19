import time
import optuna
import numpy as np
from pandas import DataFrame
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import model_util as mu



class SupervisedModel():
    def __init__(self):
        self.model_name = 'Supervized'
        self.objective = mu.Objective(num_of_epochs=3)

    def main_process(self, dataset_name, data):
        last_column_name = data.columns[len(data.columns) - 1]
        X, y = data.drop(columns=[last_column_name]), data[last_column_name]
        self.objective.set_num_of_classes(len(np.unique(y)))
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=10)
        final_model, final_lr, evaluation_df = self.train_parameters(dataset_name, self.model_name, X_train, Y_train)
        best_model, evaluation_params = self.evaluate_model(X_test, Y_test, final_model, 3)
        return best_model, evaluation_params, evaluation_df

    # This function preform the 3-Fold cross validation and uses Optuna to Optimize the model Hyperparameter - Learning rate.
    def inner_k_fold(self, Xtrain, Ytrain):
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
            self.objective.set_params(x_train, y_train, x_valid, y_valid)
            study.optimize(self.objective, n_trials=50)
            trial = study.best_trial
            trial_duration_list.append(trial.duration)
            if trial.value >= best_score:
                best_score = trial.value
                best_lr = study.best_params['learning_rate']
                best_batch_size = study.best_params['batch_size']
        return best_lr, best_score, self.objective.get_best_model(), best_batch_size, np.mean(trial_duration_list)

    # This function perform the 10-Fold cross validation and according to F1-Scoring choose the best model
    def train_parameters(self, dataset_name, algorithm_name, Xtrain, Ytrain):
        f1_max_score = 0
        final_model = None
        final_lr = 0
        current_fold = 1
        evaluation_df = DataFrame()
        model_evaluation = mu.ModelEvaluation(dataset_name, algorithm_name)
        outer_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=np.random.seed(7))
        for train_index, validate_index in outer_kfold.split(Xtrain, Ytrain):
            x_train, x_valid = Xtrain.iloc[train_index, :], Xtrain.iloc[validate_index, :]
            y_train, y_valid = Ytrain.iloc[train_index], Ytrain.iloc[validate_index]
            best_lr, best_score, best_model, best_batch_size, training_time = self.inner_k_fold(x_train, y_train)
            print("best_lr" + str(best_lr))
            print("best_score" + str(best_score))
            start = time.time()
            y_proba = best_model.predict_proba(x_valid)
            yhat = np.argmax(best_model.predict(x_valid), axis=1)
            inference_time = time.time() - start
            y_valid = y_valid.values
            f1_model_score = f1_score(y_valid, yhat, average='weighted')
            if f1_model_score > f1_max_score:
                f1_max_score = f1_model_score
                final_model = best_model
                final_lr = best_lr
            model_evaluation.get_result_to_final_table(current_fold, y_valid, yhat, y_proba,
                                                       {"learning_rate": best_lr, "batch_size": best_batch_size},
                                                       training_time, inference_time)
            current_fold = current_fold + 1
            evaluation_df = model_evaluation.get_result_df()
        return final_model, final_lr, evaluation_df

    def evaluate_model(self, X_test_model, Y_test_model, model, num_of_round):
        accuracy_list = []
        f1_scores_list = []
        sensitivity_list = []
        specificity_list = []
        ROC_area_list = []
        final_model = None
        f1_max_score = 0
        for index in range(num_of_round):
            X_train, X_test, Y_train, Y_test = train_test_split(X_test_model, Y_test_model, test_size=0.2,  random_state=42)
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
            # ROC_area_list.append(roc_auc_score(Y_test, yhat, multi_class='ovo',average="weighted"))
            if f1_model_score > f1_max_score:
                f1_max_score = f1_model_score
                final_model = model
        eval_params = [accuracy_list, f1_scores_list, sensitivity_list, specificity_list, ROC_area_list]
        return final_model, eval_params

