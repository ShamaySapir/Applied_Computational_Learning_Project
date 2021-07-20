import time
import optuna
import numpy as np
from pandas import DataFrame
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import model_util as mu
from keras.callbacks import History
import pandas as pd


class PLSemiSupervised():
    def __init__(self, num_of_total_epoch, dataset_name):
        self.model_name = 'Pseudo-Label Semi-Supervised'
        self.history = History()
        self.num_of_total_epoch = num_of_total_epoch
        self.dataset_name = dataset_name
        self.Af = 3
        self.T1 = 3
        self.T2 = 8
        self.objective = mu.Objective(num_of_epochs=1)
        self.loss_list = []
        self.curr_epoch = 1
        self.curr_run = 1

    def main_process(self, data):
        final_evaluation_df = DataFrame()
        last_column_name = data.columns[len(data.columns) - 1]
        X, y = data.drop(columns=[last_column_name]), data[last_column_name]
        self.objective.set_num_of_classes(len(np.unique(y)))
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=10)
        X_labeled, X_unlabeled, Y_labeled, Y_unlabeled = train_test_split(X_train, Y_train, test_size=0.6, random_state=10)
        while self.num_of_total_epoch >= self.curr_run:
            model,evaluation_params, evaluation_df, labeled_avg_loss, mixed_avg_loss= self.train_data_mix(X_labeled, X_unlabeled, Y_labeled, Y_unlabeled)
            self.loss_list.append(self.calculate_loss(self.curr_epoch, labeled_avg_loss, mixed_avg_loss))
            self.curr_epoch = self.curr_epoch + 1
            final_evaluation_df = final_evaluation_df.append(evaluation_df)
            self.curr_run = self.curr_run + 1
        return model, evaluation_params, final_evaluation_df, np.mean(self.loss_list)

    def train_data_mix(self, X_labeled, X_unlabeled, Y_labeled, Y_unlabeled):
        evaluation_df = DataFrame()
        # Train the model with only labeled data
        labeled_best_model, labeled_evaluation_params, labeled_avg_loss, labeled_evaluation_df = self.train_model_for_PL(X_labeled, Y_labeled)
        evaluation_df = evaluation_df.append(labeled_evaluation_df)
        # use the model to create Pseudo labels
        y_ul_pred = np.argmax(labeled_best_model.predict(X_unlabeled), axis=1)
        Y_unlabeled = pd.Series(data=y_ul_pred, index=Y_unlabeled.index)
        # Mixed labeled and unlabeled data
        X_mixed_data = X_labeled.append(X_unlabeled)
        Y_mixed_data = Y_labeled.append(Y_unlabeled)
        # Train the model with mixed labels
        mixed_best_model, mixed_evaluation_params, mixed_avg_loss, mixed_evaluation_df = self.train_model_for_PL(X_mixed_data, Y_mixed_data)
        evaluation_df = evaluation_df.append(mixed_evaluation_df)
        return mixed_best_model, mixed_evaluation_params, evaluation_df, labeled_avg_loss, mixed_avg_loss

    def calculate_loss(self, curr_epoch, labeled_avg_loss, mixed_avg_loss):
        af = 0
        if self.T1 > 0:
            af = 0
        elif self.T1 <= curr_epoch <= self.T2:
            af = ((curr_epoch-self.T1)/(self.T2-self.T1))*self.Af
        else:
            af = self.Af
        return labeled_avg_loss + af * mixed_avg_loss

    def train_model_for_PL(self, X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
        final_model, final_lr, evaluation_df = self.train_parameters(X_train, Y_train)
        best_model, evaluation_params, sm_losses = self.evaluate_model(X_test, Y_test, final_model, 3)
        avg_loss = np.mean(sm_losses)
        return best_model, evaluation_params, avg_loss, evaluation_df

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
    def train_parameters(self, Xtrain, Ytrain):
        f1_max_score = 0
        final_model = None
        final_lr = 0
        evaluation_df = DataFrame()
        model_evaluation = mu.ModelEvaluation(self.dataset_name, self.model_name)
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
            y_valid = y_valid.values
            inference_time = time.time() - start
            f1_model_score = f1_score(y_valid, yhat, average='weighted')
            if f1_model_score > f1_max_score:
                f1_max_score = f1_model_score
                final_model = best_model
                final_lr = best_lr
            model_evaluation.get_result_to_final_table(self.curr_epoch, y_valid, yhat, y_proba,
                                                        best_lr, best_batch_size,training_time, inference_time)
            self.curr_epoch = self.curr_epoch + 1
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
            X_train, X_test, Y_train, Y_test = train_test_split(X_test_model, Y_test_model, test_size=0.2, random_state=42)
            model.fit(X_train, Y_train, callbacks=[self.history])
            model_loss = model.history.history['loss']
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
        self.objective.set_best_model(final_model)
        return final_model, eval_params, model_loss

