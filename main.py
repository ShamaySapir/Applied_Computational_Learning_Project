import os
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from Algorithms.Improved_pl_semi_supervised import ImprovedPLSemiSupervised

np.seterr(divide='ignore', invalid='ignore')
from Algorithms.Supervised import SupervisedModel
from Algorithms.PL_semi_supervised import PLSemiSupervised

def show_model_evaluation(list, dataset_name):
    print(f'==== {dataset_name} ===')
    print('Accuracy: %.3f (%.3f)' % (np.mean(list[0]), np.std(list[0])))
    print('F1: %.3f (%.3f)' % (np.mean(list[1]), np.std(list[1])))
    print('Sensitivity: %.3f (%.3f)' % (np.mean(list[2]), np.std(list[2])))
    print('Specificity: %.3f (%.3f)' % (np.mean(list[3]), np.std(list[3])))
    #print('AUROC: %.3f (%.3f)' % (np.mean(list[4]), np.std(list[4])))
    return np.mean(list[0])


def discretization(data):
    # convert categorial variables into numeric values
    label_encoder = LabelEncoder()
    discret_vec = data.columns
    for category in discret_vec:
        data[category] = label_encoder.fit_transform(data[category])
    # print("data was successfully discretized")
    return data


def print_to_csv(df, outdir, f_name):
    fullname = os.path.join(outdir, f_name)
    df.to_csv(fullname)


if __name__ == "__main__":
    outdir = './Results'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # supervised_model.model_name

    # Load data and preaper it
    Supervised_df = DataFrame()
    Pl_semi_df = DataFrame()
    Improved_pl_df = DataFrame()
    files_names = os.listdir('./datasets/')
    for file_name in files_names:
        data = pd.read_csv('./datasets/'+file_name)
        data = discretization(data)
        # #supervised model
        # supervised_model = SupervisedModel()
        # sm_best_model, sm_evaluation_params, sm_evaluation_df = supervised_model.main_process(file_name , data)
        # Supervised_df = Supervised_df.append(sm_evaluation_df)
        # model_mean_accuracy = show_model_evaluation(sm_evaluation_params, file_name)
        # print(Supervised_df)
        #
        # # pseudo label semi supervised model
        # pl_semi_supervised_model = PLSemiSupervised(2, file_name)
        # pl_ssm_best_model, pl_ssm_evaluation_params, pl_ssm_evaluation_df, pl_ssm_avg_loss = pl_semi_supervised_model.main_process(data)
        # pl_ssm_evaluation_df = pl_ssm_evaluation_df.groupby(by=['Dataset Name', 'Algorithm Name', 'Cross Validation index']).mean()
        # Pl_semi_df = Pl_semi_df.append(pl_ssm_evaluation_df)
        # pl_semi_model_mean_accuracy = show_model_evaluation(pl_ssm_evaluation_params, file_name)
        # print(Pl_semi_df)
        #
        # print("The Pseudo-Label Semi-Supervised Avrage Loss is: "+pl_ssm_avg_loss)
        # print("The Pseudo-Label Semi-Supervised Avrage Accuracy is: "+pl_semi_model_mean_accuracy)

        improved_pl_model = ImprovedPLSemiSupervised(10, file_name)
        improved_pl_best_model, improved_pl_evaluation_params, improved_pl_evaluation_df, improved_pl_avg_loss = improved_pl_model.main_process(
            data)
        improved_pl_evaluation_df = improved_pl_evaluation_df.groupby(
            by=['Dataset Name', 'Algorithm Name', 'Cross Validation index']).mean()
        Improved_pl_df = Improved_pl_df.append(improved_pl_evaluation_df)
        improved_pl_model_mean_accuracy = show_model_evaluation(improved_pl_evaluation_params, file_name)
        print(improved_pl_evaluation_df)
        print("The Improved Pseudo-Label Semi-Supervised Avrage Loss is: " + improved_pl_avg_loss)
        print("The Improved Pseudo-Label Semi-Supervised Avrage Accuracy is: " + improved_pl_model_mean_accuracy)

    # print_to_csv(Supervised_df, outdir, "evaluation_supervided.csv")
    # print_to_csv(Pl_semi_df, outdir, "evaluation_pl_semi.csv")
    print_to_csv(Improved_pl_df, outdir, "evaluation_improve_pl_semi.csv")
