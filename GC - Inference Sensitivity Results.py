import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import statistics
import keras
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, TimeDistributed, Flatten
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import confusion_matrix

######################################
######  Data Generating Process ######
######     Function Type I      ######
######################################

"""
The Data Generating Process (DGP) Function creates data with a Vector Autoregressive (VAR) Process up to a dependency of two lags

Parameters
----------

- sample_size (integer) : Determines the amount of simulated data points / observations for each individually artificially created time series
- lags (vector)         : Reflects the wished dependency on the given lags, e. g. lags = [1,4]
- k (integer)           : Number of functions within the VAR Process
- tensor_count (integer): Sets the amount of simulated VAR Processes with k functions. E. g. k = 2 and tensor_count = 100, means that 100 VAR Processes with 2 sich gegenseitig beeinflussenden 
- noise (matrix)        : Created error term matrix of size sample_size x k of the user

Example: sample_size = 1000
         lags = [1,3]
         k = 2
         tensor_count = 10

Meaning: 10 VAR-Processes with 2 functions and 1000 observations each are created which depend on the first and the third lag of the own observations and the other timeseries.
The relationship (Coefficients) vary over the cases and are described within the paper.
"""


# %%
def DataGeneration(case, sample_size,  # integer
                   lags,  # vector e. g. [1,2,3,4]
                   k,
                   tensor_count,  # integer
                   noise):  # error term

    lag_amount = len(lags)

    # create multiple empty coefficient tensors of the amount of tensor_count
    for p in range(lag_amount):
        for tensor in range(tensor_count):
            globals()["lag_" + str(lags[p]) + "_tensor_" + str(tensor + 1)] = []

    # create coefficent matrices and append to the tensor
    for p in range(lag_amount):
        for tensor in range(tensor_count):
            for i in range(sample_size + 2 * lag_amount):
                # create an empty matrix with dimension k x k
                matrix = np.zeros((k, k))

                # append matrix for each simulation step to the tensor
                globals()["lag_" + str(lags[p]) + "_tensor_" + str(tensor + 1)].append(matrix)
                i += 1

    # Granger Case 1 (GC1):

    if case == 1:
        # every tensor for every lag gets a seed equals tensor_count * lag_amount different seeds
        seed_switch = cycle(range((tensor_count) * lag_amount))

        next(seed_switch)

        for p in range(lag_amount):
            for tensor in range(tensor_count):
                np.random.seed(next(
                    seed_switch))  # every tensor for every lag gets a seed equals tensor_count * lag_amount different seeds
                coef_subtensor = np.random.uniform(-0.2, 0.2, (k, k))
                for i in range(p, sample_size + 2 * lag_amount):
                    # coef_array = np.random.uniform(-0.5,0.5,(k,k))
                    # coef_subtensor.append(coef_array)
                    globals()["lag_" + str(lags[p]) + "_tensor_" + str(tensor + 1)][i] = coef_subtensor

    # create an empty array for the simulated data dat
    for tensor in range(tensor_count):
        globals()["dat_tensor_" + str(tensor + 1)] = np.zeros((sample_size + 2 * lag_amount, k))

    # fill the dataframe with the VAR-Model via matrix multiplication

    if lag_amount == 4:
        # for p in range(lag_amount):
        for tensor in range(tensor_count):
            for i in range(p, sample_size + 2 * lag_amount):
                globals()["dat_tensor_" + str(tensor + 1)][i, :] = np.matmul(
                    globals()["lag_" + str(lags[0]) + "_tensor_" + str(tensor + 1)][i],
                    globals()["dat_tensor_" + str(tensor + 1)][i - lags[0], :]) + np.matmul(
                    globals()["lag_" + str(lags[1]) + "_tensor_" + str(tensor + 1)][i],
                    globals()["dat_tensor_" + str(tensor + 1)][i - lags[1], :]) + np.matmul(
                    globals()["lag_" + str(lags[2]) + "_tensor_" + str(tensor + 1)][i],
                    globals()["dat_tensor_" + str(tensor + 1)][i - lags[2], :]) + np.matmul(
                    globals()["lag_" + str(lags[3]) + "_tensor_" + str(tensor + 1)][i],
                    globals()["dat_tensor_" + str(tensor + 1)][i - lags[3], :]) + noise[i]

    if lag_amount == 3:
        # for p in range(lag_amount):
        for tensor in range(tensor_count):
            for i in range(p, sample_size + 2 * lag_amount):
                globals()["dat_tensor_" + str(tensor + 1)][i, :] = np.matmul(
                    globals()["lag_" + str(lags[0]) + "_tensor_" + str(tensor + 1)][i],
                    globals()["dat_tensor_" + str(tensor + 1)][i - lags[0], :]) + np.matmul(
                    globals()["lag_" + str(lags[1]) + "_tensor_" + str(tensor + 1)][i],
                    globals()["dat_tensor_" + str(tensor + 1)][i - lags[1], :]) + np.matmul(
                    globals()["lag_" + str(lags[2]) + "_tensor_" + str(tensor + 1)][i],
                    globals()["dat_tensor_" + str(tensor + 1)][i - lags[2], :]) + noise[i]

    if lag_amount == 2:
        # for p in range(lag_amount):
        for tensor in range(tensor_count):
            for i in range(p, sample_size + 2 * lag_amount):
                globals()["dat_tensor_" + str(tensor + 1)][i, :] = np.matmul(
                    globals()["lag_" + str(lags[0]) + "_tensor_" + str(tensor + 1)][i],
                    globals()["dat_tensor_" + str(tensor + 1)][i - lags[0], :]) + np.matmul(
                    globals()["lag_" + str(lags[1]) + "_tensor_" + str(tensor + 1)][i],
                    globals()["dat_tensor_" + str(tensor + 1)][i - lags[1], :]) + noise[i]

    if lag_amount == 1:
        # for p in range(lag_amount):
        for tensor in range(tensor_count):
            for i in range(p, sample_size + 2 * lag_amount):
                globals()["dat_tensor_" + str(tensor + 1)][i, :] = np.matmul(
                    globals()["lag_" + str(lags[0]) + "_tensor_" + str(tensor + 1)][i],
                    globals()["dat_tensor_" + str(tensor + 1)][i - lags[0], :]) + noise[i]

    else:
        print("Amount of lags has to be at least 1")


#################################
###### Inference Functions ######
#################################


# Analysis of F Statistic and it´s development over the analysed lag span ###

# Plots



def development_F_value(F_table, plot_name):
    plt.figure(figsize=(20, 5))
    plt.plot(F_table)
    for i in range(len(lags)):
        plt.plot([lags[i] - 1, lags[i] - 1], [0, max(F_table.max())], 'k-', lw=2, linestyle="--")
    plt.savefig(f"{plot_name}.pdf", bbox_inches="tight")


# Analysis of F Statistic and it´s correlation with the coefficients ###
def correlation_plot(F_table):
    """
    Plots the F Values of the given F Table to the Coefficients of the current tensor setting

    Parameters
    ----------

    - F_table(Matrix)    : Matrix with F values from analysed Granger Causality Test
    """

    colours = ["g", "r", "b", "grey"]

    lag_Coefs = pd.DataFrame(columns=[f"Tensor {tensor + 1}" for tensor in range(tensor_count)],
                             index=[f"Lag {p + 1}" for p in range(len(lags))])
    lag_Fvalues = F_table.iloc[:len(lags), :]

    for p in range(len(lags)):
        # derive Coefficients from every tensor
        for tensor in range(tensor_count):
            lag_Coefs.iloc[p, tensor] = globals()["lag_" + str(lags[1]) + "_tensor_" + str(tensor + 1)][0][0][1]

    for p in range(len(lags)):
        # correlation = lag_Fvalues.iloc[p,:].corr(abs(lag_Coefs.iloc[p,:]), method = "pearson")
        plt.scatter(lag_Fvalues.iloc[p, :], abs(lag_Coefs.iloc[p, :]), c=colours[p], label=f"Lag {p + 1}")
        plt.legend()
    plt.show()


def resulttables_statsmodels(p_table, F_table):
    for tensor in range(tensor_count):
        print(f"-----------------tensor {tensor + 1}-----------------")
        stats_result = grangercausalitytests(globals()["dat_tensor_" + str(tensor + 1)], maxlag=test_lags_upto)

        for i in range(1, test_lags_upto + 1):
            p_table.loc[f"Number of lags: {i}", f"tensor {tensor + 1}"] = round(stats_result[i][0]["params_ftest"][1],
                                                                                8)
            F_table.loc[f"Number of lags: {i}", f"tensor {tensor + 1}"] = round(stats_result[i][0]["params_ftest"][0],
                                                                                8)


def conf_mat(threshold, yhat):

    inner_loopsize = yhat.shape[0] / (yhat.shape[0] / max(lags))
    outer_loopsize = yhat.shape[1] * (yhat.shape[0] / max(lags))
    true_result_stacked = []

    for outer in range(int(outer_loopsize)):
        for inner in range(1, int(inner_loopsize) + 1):
            if inner == lags[0] or inner == lags[1] or inner == lags[2] or inner == lags[3]:
                value = int(1)
                true_result_stacked.append(value)
            else:
                value = int(0)
                true_result_stacked.append(value)

    y_coi = pd.DataFrame(true_result_stacked)

    # prepare the vector with the estimated data (p-values)
    sub_yhat = yhat.melt()
    yhat_pred = sub_yhat.iloc[:, -1]
    yhat_is_coi = np.empty((sub_yhat.shape[0], 1))

    for i in range(sub_yhat.shape[0]):
        if yhat_pred.loc[i] <= threshold:
            yhat_sub = 1
            yhat_is_coi[i] = yhat_sub
        else:
            yhat_sub = 0
            yhat_is_coi[i] = yhat_sub

    confusion_mat = confusion_matrix(y_coi, yhat_is_coi)
    return confusion_mat



def roc_adj(threshold, yhat):
    """
    Derivation of the appropriate significant-level (alpha) using a adjusted ROC approach, in other words: this helps to find a tradeoff spot.

    Parameters
    ----------

    - threshold (integer): desired significance level
    - y_coi (Vector)     : Vector of size sample_size with bivariate variables 0 and 1. 1 if the lag has an effect 0 else
    - yhat (Vector)      : P value table from desired package

    The following implementation of functions is used to derive the p values of each analysed package within python.
    Test results of other packages are obtained in R and then are imported in python for further processes.
    """

    inner_loopsize = yhat.shape[0] / (yhat.shape[0] / max(lags))
    outer_loopsize = yhat.shape[1] * (yhat.shape[0] / max(lags))
    true_result_stacked = []

    for outer in range(int(outer_loopsize)):
        for inner in range(1, int(inner_loopsize) + 1):
            if inner == lags[0] or inner == lags[1] or inner == lags[2] or inner == lags[3]:
                value = int(1)
                true_result_stacked.append(value)
            else:
                value = int(0)
                true_result_stacked.append(value)

    y_coi = pd.DataFrame(true_result_stacked)

    # prepare the vector with the estimated data (p-values)
    sub_yhat = yhat.melt()
    yhat_pred = sub_yhat.iloc[:, -1]
    yhat_is_coi = np.empty((sub_yhat.shape[0], 1))

    for i in range(sub_yhat.shape[0]):
        if yhat_pred.loc[i] <= threshold:
            yhat_sub = 1
            yhat_is_coi[i] = yhat_sub
        else:
            yhat_sub = 0
            yhat_is_coi[i] = yhat_sub

    # calculate True Positive Rate (TPR)
    col_to_summup_TPR = yhat_is_coi * y_coi
    numerator_TPR = col_to_summup_TPR.sum()
    result_TPR = numerator_TPR / y_coi.sum()
    TruePosRate = round(result_TPR[0], 3)

    # calculate False Positive Rate (FPR)
    col_to_summup_FPR = (1 - y_coi) * yhat_is_coi
    numerator_FPR = col_to_summup_FPR.sum()
    denominator_FPR = 1 - y_coi
    result_FPR = numerator_FPR / denominator_FPR.sum()
    FalsePosRate = round(result_FPR[0], 3)

    return threshold, TruePosRate, FalsePosRate


def createROCcurve(alpha_seq, p_values_test, plot_name):
    """
    Function for ROC Plot

    Parameters
    ----------
    - alpha_seq (DataFrame): Vector with signficance levels
    - true_result_stacked  : desired significance level
    - p_values_test        : Vector of size sample_size with bivariate variables 0 and 1. 1 if the lag has an effect 0 else

    The following implementation of functions is used to derive the p values of each analysed package.
    Test results of other packages are obtained in R and then are imported in python for further processes.
    """

    roc_data = pd.DataFrame(columns=["alpha", "TPR", "FPR"], index=[i for i in range(len(alpha_seq))])

    for alpha in range(len(alpha_seq)):
        roc = roc_adj(threshold=alpha_seq[alpha], yhat=p_values_test)
        roc_data.iloc[alpha]["alpha"] = roc[0]
        roc_data.iloc[alpha]["TPR"] = roc[1]
        roc_data.iloc[alpha]["FPR"] = roc[2]

    plt.figure()
    lw = 2
    plt.plot(
        roc_data["FPR"],
        roc_data["TPR"],
        lw=lw,
        color="black"
    )
    plt.plot([0, max(roc_data["FPR"])], [0, max(roc_data["TPR"])], color="black", lw=lw, linestyle="--")
    plt.plot(roc_data.iloc[1]["FPR"], roc_data.iloc[1]["TPR"], "o", c="black", label="α = 0.01")
    plt.plot(roc_data.iloc[5]["FPR"], roc_data.iloc[5]["TPR"], "x", c="black", label="α = 0.05")
    plt.plot(roc_data.iloc[10]["FPR"], roc_data.iloc[10]["TPR"], "^", c="black", label="α = 0.10")
    plt.legend(loc="lower right")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(plot_name)

    def df_name(df):  # get the name of the dataframe
        name = [x for x in globals() if globals()[x] is df][0]
        return (name)

    plt.savefig(rf"C:/Users/Fabia/OneDrive/Desktop/Studium/Master/Semester III/Seminar Financial Data Analytics/Submission/{plot_name}.pdf", bbox_inches="tight")
    plt.show()
    return roc_data


def createROCcurve_combined(alpha_seq, p_values_test_YX, p_values_test_XY, plot_name):
    """
    Function for ROC Plot

    Parameters
    ----------
    - alpha_seq (DataFrame): Vector with signficance levels
    - true_result_stacked  : desired significance level
    - p_values_test        : Vector of size sample_size with bivariate variables 0 and 1. 1 if the lag has an effect 0 else

    The following implementation of functions is used to derive the p values of each analysed package.
    Test results of other packages are obtained in R and then are imported in python for further processes.
    """

    roc_data_YX = pd.DataFrame(columns=["alpha", "TPR", "FPR"], index=[i for i in range(len(alpha_seq))])
    roc_data_XY = pd.DataFrame(columns=["alpha", "TPR", "FPR"], index=[i for i in range(len(alpha_seq))])

    for alpha in range(len(alpha_seq)):
        roc_YX = roc_adj(threshold=alpha_seq[alpha], yhat=p_values_test_YX)
        roc_data_YX.iloc[alpha]["alpha"] = roc_YX[0]
        roc_data_YX.iloc[alpha]["TPR"] = roc_YX[1]
        roc_data_YX.iloc[alpha]["FPR"] = roc_YX[2]
    roc_data_YX = pd.DataFrame(roc_data_YX)

    for alpha in range(len(alpha_seq)):
        roc_XY = roc_adj(threshold=alpha_seq[alpha], yhat=p_values_test_XY)
        roc_data_XY.iloc[alpha]["alpha"] = roc_XY[0]
        roc_data_XY.iloc[alpha]["TPR"] = roc_XY[1]
        roc_data_XY.iloc[alpha]["FPR"] = roc_XY[2]
    roc_data_XY = pd.DataFrame(roc_data_XY)

    plt.figure()
    lw = 2
    plt.plot(
        roc_data_YX["FPR"],
        roc_data_XY["TPR"],
        lw=lw,
        color="red"
    )
    plt.plot(
        roc_data_XY["FPR"],
        roc_data_YX["TPR"],
        lw=lw,
        color="blue"
    )

    # plt.plot([0, max(roc_data["FPR"])], [0, max(roc_data["TPR"])], color="black", lw=lw, linestyle="--")
    plt.plot(roc_data_XY.iloc[1]["FPR"], roc_data_XY.iloc[1]["TPR"], "x", c="black", label="α = 0.01")
    plt.plot(roc_data_XY.iloc[5]["FPR"], roc_data_XY.iloc[5]["TPR"], "x", c="blue", label="α = 0.05")
    plt.plot(roc_data_XY.iloc[10]["FPR"], roc_data_XY.iloc[10]["TPR"], "x", c="grey", label="α = 0.1")
    plt.plot(roc_data_YX.iloc[1]["FPR"], roc_data_YX.iloc[1]["TPR"], "x", c="black")
    plt.plot(roc_data_YX.iloc[5]["FPR"], roc_data_YX.iloc[5]["TPR"], "x", c="blue")
    plt.plot(roc_data_YX.iloc[10]["FPR"], roc_data_YX.iloc[10]["TPR"], "x", c="grey")
    plt.legend(loc="lower right")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(plot_name)
    # def df_name(df): #get the name of the dataframe
    #    name =[x for x in globals() if globals()[x] is df][0]
    #   return(name)
    # plt.savefig(f"{df_name(p_values_test)}.pdf", bbox_inches = "tight")
    plt.show()
    # return roc_data


def createROCdata(alpha_seq, p_values_test, df_roc):
    for alpha in range(len(alpha_seq)):
        roc = roc_adj(threshold=alpha_seq[alpha], yhat=p_values_test)
        df_roc.iloc[alpha]["alpha"] = roc[0]
        df_roc.iloc[alpha]["TPR"] = roc[1]
        df_roc.iloc[alpha]["FPR"] = roc[2]

    return df_roc


def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    """
    Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.

    Parameters
    ----------
    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """

    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


def import_data_R(Setting, siglags):
    test_names_YX = ["lmtest_p_table", "NlinTS_p_table", "NlinTS_NN_p_table", "GrangerFunc_p_table",
                     "lmtest_F_table", "NlinTS_F_table", "NlinTS_NN_F_table", "GrangerFunc_F_table"]

    for test in test_names_YX:
        # YX
        globals()["GC_" + str(Setting) + "_" + str(test)] = pd.read_csv(
            rf'C:/Users/Fabia/pythoninput/Granger Causality/dataexport/GC Inference Sensitivity {Setting}/Results/{test}.csv')
        globals()["GC_" + str(Setting) + "_" + str(test)] = globals()["GC_" + str(Setting) + "_" + str(test)].set_index(
            "Unnamed: 0")
        globals()["GC_" + str(Setting) + "_" + str(test) + "_siglags"] = globals()["GC_" + str(Setting) + "_" + str(
            test)].iloc[0:siglags, :]

    test_names_XY = ["lmtest_p_table_XY", "NlinTS_p_table_XY", "NlinTS_NN_p_table_XY","GrangerFunc_p_table_XY",
                     "lmtest_F_table_XY", "NlinTS_F_table_XY", "NlinTS_NN_F_table_XY","GrangerFunc_F_table_XY"]

    for test in test_names_XY:
        # XY
        globals()["GC_" + str(Setting) + "_" + str(test)] = pd.read_csv(
            rf'C:/Users/Fabia/pythoninput/Granger Causality/dataexport/GC Inference Sensitivity {Setting}/Results/{test}.csv')
        globals()["GC_" + str(Setting) + "_" + str(test)] = globals()["GC_" + str(Setting) + "_" + str(test)].set_index(
            "Unnamed: 0")
        globals()["GC_" + str(Setting) + "_" + str(test) + "_siglags"] = globals()["GC_" + str(Setting) + "_" + str(
            test)].iloc[0:siglags, :]

    # Join
    for test_YX, test_XY in zip(test_names_YX, test_names_XY):
        globals()["GC_" + str(Setting) + "_" + str(test_YX) + "_joint"] = globals()[
            "GC_" + str(Setting) + "_" + str(test_YX)].append(globals()["GC_" + str(Setting) + "_" + str(test_XY)])
        globals()["GC_" + str(Setting) + "_" + str(test_YX) + "_joint_siglags"] = globals()[
            "GC_" + str(Setting) + "_" + str(test_YX) + "_siglags"].append(
            globals()["GC_" + str(Setting) + "_" + str(test_XY) + "_siglags"])


def import_data_Py(Setting, siglags, both):
    test_names_YX = ["statsmodels_table_p","statsmodels_table_p", "NonlinC_NN_p_table", "NonlinC_GRU_p_table", "NonlinC_LSTM_p_table",
                     "statsmodels_table_F", "NonlinC_NN_F_table", "NonlinC_GRU_F_table", "NonlinC_LSTM_F_table"]


    for test in test_names_YX:
        # YX
        globals()["GC_" + str(Setting) + "_" + str(test)] = pd.read_csv(
            rf'C:/Users/Fabia/pythoninput/Granger Causality/Python Data/GC Inference Sensitivity {Setting}/{test}.csv')
        globals()["GC_" + str(Setting) + "_" + str(test)] = globals()["GC_" + str(Setting) + "_" + str(test)].set_index(
            "Unnamed: 0")
        globals()["GC_" + str(Setting) + "_" + str(test) + "_siglags"] = globals()["GC_" + str(Setting) + "_" + str(
            test)].iloc[0:siglags, :]

    if(both == True):

        test_names_XY = ["statsmodels_table_p_XY","statsmodels_table_p_XY", "NonlinC_NN_p_table_XY", "NonlinC_GRU_p_table_XY","NonlinC_LSTM_p_table_XY",
                         "statsmodels_table_F_XY", "NonlinC_NN_F_table_XY", "NonlinC_GRU_F_table_XY","NonlinC_LSTM_F_table_XY"]

        for test in test_names_XY:
            # XY
            globals()["GC_" + str(Setting) + "_" + str(test)] = pd.read_csv(
                rf'C:/Users/Fabia/pythoninput/Granger Causality/Python Data/GC Inference Sensitivity {Setting}/{test}.csv')
            globals()["GC_" + str(Setting) + "_" + str(test)] = globals()["GC_" + str(Setting) + "_" + str(test)].set_index(
                "Unnamed: 0")
            globals()["GC_" + str(Setting) + "_" + str(test) + "_siglags"] = globals()["GC_" + str(Setting) + "_" + str(
                test)].iloc[0:siglags, :]

        # Join
        for test_YX, test_XY in zip(test_names_YX, test_names_XY):
            globals()["GC_" + str(Setting) + "_" + str(test_YX) + "_joint"] = globals()[
                "GC_" + str(Setting) + "_" + str(test_YX)].append(globals()["GC_" + str(Setting) + "_" + str(test_XY)])
            globals()["GC_" + str(Setting) + "_" + str(test_YX) + "_joint_siglags"] = globals()[
                "GC_" + str(Setting) + "_" + str(test_YX) + "_siglags"].append(
                globals()["GC_" + str(Setting) + "_" + str(test_XY) + "_siglags"])


def ROC_composition(Setting, alpha_seq, both):
    test_names_YX = ["statsmodels_table_p_siglags","statsmodels_table_p_siglags", "NonlinC_NN_p_table_siglags", "NonlinC_GRU_p_table_siglags",
                     "NonlinC_LSTM_p_table_siglags", "lmtest_p_table_siglags", "NlinTS_p_table_siglags",
                     "NlinTS_NN_p_table_siglags","GrangerFunc_p_table_siglags"]

    for test in test_names_YX:
        # YX
        globals()["GC_" + str(Setting) + "_" + str(test) + "_roc_data"] = pd.DataFrame(columns=["alpha", "TPR", "FPR"],
                                                                                       index=[i for i in
                                                                                              range(len(alpha_seq))])
        createROCdata(alpha_seq=alpha_seq, p_values_test=globals()["GC_" + str(Setting) + "_" + str(test)],
                      df_roc=globals()["GC_" + str(Setting) + "_" + str(test) + "_roc_data"])

    if both == True:
        test_names_XY = ["statsmodels_table_p_XY_siglags","statsmodels_table_p_XY_siglags", "NonlinC_NN_p_table_XY_siglags",
                         "NonlinC_GRU_p_table_XY_siglags", "NonlinC_LSTM_p_table_XY_siglags", "lmtest_p_table_XY_siglags",
                         "NlinTS_p_table_XY_siglags", "NlinTS_NN_p_table_XY_siglags","GrangerFunc_p_table_XY_siglags"]

        for test in test_names_XY:
            # XY
            globals()["GC_" + str(Setting) + "_" + str(test) + "_roc_data_XY"] = pd.DataFrame(
                columns=["alpha", "TPR", "FPR"], index=[i for i in range(len(alpha_seq))])
            createROCdata(alpha_seq=alpha_seq, p_values_test=globals()["GC_" + str(Setting) + "_" + str(test)],
                          df_roc=globals()["GC_" + str(Setting) + "_" + str(test) + "_roc_data_XY"])

        # Joint
        test_names_joint = ["statsmodels_table_p_joint_siglags","statsmodels_table_p_joint_siglags", "NonlinC_NN_p_table_joint_siglags",
                            "NonlinC_GRU_p_table_joint_siglags", "NonlinC_LSTM_p_table_joint_siglags",
                            "lmtest_p_table_joint_siglags", "NlinTS_p_table_joint_siglags",
                            "NlinTS_NN_p_table_joint_siglags","GrangerFunc_p_table_joint_siglags"]

        for test in test_names_joint:
            # XY
            globals()["GC_" + str(Setting) + "_" + str(test) + "_roc_data_joint"] = pd.DataFrame(
                columns=["alpha", "TPR", "FPR"], index=[i for i in range(len(alpha_seq))])
            createROCdata(alpha_seq=alpha_seq, p_values_test=globals()["GC_" + str(Setting) + "_" + str(test)],
                          df_roc=globals()["GC_" + str(Setting) + "_" + str(test) + "_roc_data_joint"])


def ROC_composition_pyonly(Setting, alpha_seq):
    test_names_YX = ["statsmodels_table_p_siglags", "NonlinC_NN_p_table_siglags", "NonlinC_GRU_p_table_siglags",
                     "NonlinC_LSTM_p_table_siglags"]

    for test in test_names_YX:
        # YX
        globals()["GC_" + str(Setting) + "_" + str(test) + "_roc_data"] = pd.DataFrame(columns=["alpha", "TPR", "FPR"],
                                                                                       index=[i for i in
                                                                                              range(len(alpha_seq))])
        createROCdata(alpha_seq=alpha_seq, p_values_test=globals()["GC_" + str(Setting) + "_" + str(test)],
                      df_roc=globals()["GC_" + str(Setting) + "_" + str(test) + "_roc_data"])

    test_names_XY = ["statsmodels_table_p_XY_siglags", "NonlinC_NN_p_table_XY_siglags",
                     "NonlinC_GRU_p_table_XY_siglags", "NonlinC_LSTM_p_table_XY_siglags"]

    for test in test_names_XY:
        # XY
        globals()["GC_" + str(Setting) + "_" + str(test) + "_roc_data_XY"] = pd.DataFrame(
            columns=["alpha", "TPR", "FPR"], index=[i for i in range(len(alpha_seq))])
        createROCdata(alpha_seq=alpha_seq, p_values_test=globals()["GC_" + str(Setting) + "_" + str(test)],
                      df_roc=globals()["GC_" + str(Setting) + "_" + str(test) + "_roc_data_XY"])

    # Joint
    test_names_joint = ["statsmodels_table_p_joint_siglags", "NonlinC_NN_p_table_joint_siglags",
                        "NonlinC_GRU_p_table_joint_siglags", "NonlinC_LSTM_p_table_joint_siglags"]

    for test in test_names_joint:
        # XY
        globals()["GC_" + str(Setting) + "_" + str(test) + "_roc_data_joint"] = pd.DataFrame(
            columns=["alpha", "TPR", "FPR"], index=[i for i in range(len(alpha_seq))])
        createROCdata(alpha_seq=alpha_seq, p_values_test=globals()["GC_" + str(Setting) + "_" + str(test)],
                      df_roc=globals()["GC_" + str(Setting) + "_" + str(test) + "_roc_data_joint"])



def ROC_composition_Ronly(Setting, alpha_seq):
    test_names_YX = ["lmtest_p_table_siglags", "NlinTS_p_table_siglags",
                     "NlinTS_NN_p_table_siglags","GrangerFunc_p_table_siglags"]

    for test in test_names_YX:
        # YX
        globals()["GC_" + str(Setting) + "_" + str(test) + "_roc_data"] = pd.DataFrame(columns=["alpha", "TPR", "FPR"],
                                                                                       index=[i for i in
                                                                                              range(len(alpha_seq))])
        createROCdata(alpha_seq=alpha_seq, p_values_test=globals()["GC_" + str(Setting) + "_" + str(test)],
                      df_roc=globals()["GC_" + str(Setting) + "_" + str(test) + "_roc_data"])

    test_names_XY = ["lmtest_p_table_siglags", "NlinTS_p_table_siglags",
                     "NlinTS_NN_p_table_siglags","GrangerFunc_p_table_siglags"]

    for test in test_names_XY:
        # XY
        globals()["GC_" + str(Setting) + "_" + str(test) + "_roc_data_XY"] = pd.DataFrame(
            columns=["alpha", "TPR", "FPR"], index=[i for i in range(len(alpha_seq))])
        createROCdata(alpha_seq=alpha_seq, p_values_test=globals()["GC_" + str(Setting) + "_" + str(test)],
                      df_roc=globals()["GC_" + str(Setting) + "_" + str(test) + "_roc_data_XY"])

    # Joint
    test_names_joint = ["lmtest_p_table_siglags", "NlinTS_p_table_siglags",
                        "NlinTS_NN_p_table_siglags","GrangerFunc_p_table_siglags"]

    for test in test_names_joint:
        # XY
        globals()["GC_" + str(Setting) + "_" + str(test) + "_roc_data_joint"] = pd.DataFrame(
            columns=["alpha", "TPR", "FPR"], index=[i for i in range(len(alpha_seq))])
        createROCdata(alpha_seq=alpha_seq, p_values_test=globals()["GC_" + str(Setting) + "_" + str(test)],
                      df_roc=globals()["GC_" + str(Setting) + "_" + str(test) + "_roc_data_joint"])


#############################################
######          GC - Inference         ######
######   Error Term - Sensitivity II   ######
#############################################

# GC II - Settings

sample_size = 10000 #1000
tensor_count = 30
lags = [2, 3, 4, 5]
lag_amount = len(lags)

# Determine E
sigma = [0.5, 0.5]  # standard deviation of first and second time series
corr = 0  # correlation

covs = [[sigma[0] ** 2, sigma[0] * sigma[1] * corr],
        [sigma[0] * sigma[1] * corr, sigma[1] ** 2]]

np.random.seed(42)
noise = np.random.multivariate_normal(mean=[0, 0], cov=covs, size=(sample_size + 2 * lag_amount,))

# Generate Data
DataGeneration(case=1, sample_size=sample_size, lags=lags, k=2, tensor_count=tensor_count, noise=noise)

# import results from R ###
import_data_R("II", 5)
import_data_R("II_10k", 5)

# import results from Python ###
import_data_Py("II", 5, both = True)

#create roc data for 10k statsmodels

#for denser plots
alpha_seq = np.linspace(0, 0.1, 11).round(2)

#for plots with same scale
alpha_seq_joint = np.linspace(0, 1, 101).round(2)

#both or only python packages
ROC_composition("II", alpha_seq_joint, both = True)
ROC_composition_Ronly("II_10k", alpha_seq_joint)
# ROC Curves



createROCcurve(alpha_seq_joint, GC_II_lmtest_p_table_joint_siglags, "GC_II_lmtest")
createROCcurve(alpha_seq_joint, GC_II_statsmodels_table_p_joint_siglags, "GC_II_statsmodels")
createROCcurve(alpha_seq_joint, GC_II_NonlinC_NN_p_table_siglags, "GC_II_NonlinC_NN")
createROCcurve(alpha_seq_joint, GC_II_NonlinC_GRU_p_table_siglags, "GC_II_NonlinC_GRU")
createROCcurve(alpha_seq_joint, GC_II_NonlinC_LSTM_p_table_siglags, "GC_II_NonlinC_LSTM")
createROCcurve(alpha_seq_joint, GC_II_GrangerFunc_p_table_joint_siglags, "GC_II_GrangerFunc")
createROCcurve(alpha_seq_joint, GC_II_NlinTS_NN_p_table_joint_siglags, "GC_II_NlinTS_NN")
createROCcurve(alpha_seq_joint, GC_II_NlinTS_p_table_joint_siglags, "GC_II_NlinTS_GC")

GC_II_10k_statsmodels_table_p
GC_II_10k_lmtest_p_table

createROCcurve(alpha_seq_joint, GC_II_statsmodels_table_p_joint_siglags, "GC_II_OLS_approach_1000")
createROCcurve(alpha_seq_joint, GC_III_statsmodels_table_p_joint_siglags, "GC_III_OLS_approach_1000")
createROCcurve(alpha_seq_joint, GC_IV_statsmodels_table_p_joint_siglags, "GC_IV_OLS_approach_1000")

#combined ROC curves for OLS approaches
plt.figure()
plt.plot(
    GC_II_statsmodels_table_p_joint_siglags_roc_data_joint["FPR"],
    GC_II_statsmodels_table_p_joint_siglags_roc_data_joint["TPR"],
    lw=2,
    c="gray",
    label = "1k samples"
)
plt.plot(GC_II_statsmodels_table_p_joint_siglags_roc_data_joint.iloc[1]["FPR"], GC_II_statsmodels_table_p_joint_siglags_roc_data_joint.iloc[1]["TPR"], "o", c="black")
plt.plot(GC_II_statsmodels_table_p_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_II_statsmodels_table_p_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")
plt.plot(GC_II_statsmodels_table_p_joint_siglags_roc_data_joint.iloc[10]["FPR"],GC_II_statsmodels_table_p_joint_siglags_roc_data_joint.iloc[10]["TPR"], "^", c="black")
plt.plot(
    GC_II_10k_statsmodels_table_p_joint_siglags_roc_data_joint["FPR"],
    GC_II_10k_statsmodels_table_p_joint_siglags_roc_data_joint["TPR"],
    lw=2,
    c="gray",
    label = "10k samples",
    linestyle = "--"
)
plt.plot(GC_II_10k_statsmodels_table_p_joint_siglags_roc_data_joint.iloc[1]["FPR"], GC_II_10k_statsmodels_table_p_joint_siglags_roc_data_joint.iloc[1]["TPR"], "o", c="black", label = "α = 0.01")
plt.plot(GC_II_10k_statsmodels_table_p_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_II_10k_statsmodels_table_p_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black", label = "α = 0.05")
plt.plot(GC_II_10k_statsmodels_table_p_joint_siglags_roc_data_joint.iloc[10]["FPR"], GC_II_10k_statsmodels_table_p_joint_siglags_roc_data_joint.iloc[10]["TPR"], "^", c="black",label = "α = 0.10")
plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("grangercausalitytests")
plt.legend()
plt.savefig(
    r"C:\Users\Fabia\OneDrive\Desktop\Studium\Master\Semester III\Seminar Financial Data Analytics\Submission\GC_II_grangercausalitytests.pdf",
    bbox_inches="tight")

#combined ROC curves for OLS approaches

#lmtest
plt.figure()
plt.plot(
    GC_II_lmtest_p_table_joint_siglags_roc_data_joint["FPR"],
    GC_II_lmtest_p_table_joint_siglags_roc_data_joint["TPR"],
    lw=2,
    c="gray",
    label = "1k samples"
)
plt.plot(GC_II_lmtest_p_table_joint_siglags_roc_data_joint.iloc[1]["FPR"], GC_II_lmtest_p_table_joint_siglags_roc_data_joint.iloc[1]["TPR"], "o", c="black")
plt.plot(GC_II_lmtest_p_table_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_II_lmtest_p_table_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")
plt.plot(GC_II_lmtest_p_table_joint_siglags_roc_data_joint.iloc[10]["FPR"], GC_II_lmtest_p_table_joint_siglags_roc_data_joint.iloc[10]["TPR"], "^", c="black")
plt.plot(
    GC_II_10k_lmtest_p_table_siglags_roc_data_joint["FPR"],
    GC_II_10k_lmtest_p_table_siglags_roc_data_joint["TPR"],
    lw=2,
    c="gray",
    label = "10k samples",
    linestyle = "--"
)
plt.plot(GC_II_10k_lmtest_p_table_siglags_roc_data_joint.iloc[1]["FPR"], GC_II_10k_lmtest_p_table_siglags_roc_data_joint.iloc[1]["TPR"], "o", c="black", label = "α = 0.01")
plt.plot(GC_II_10k_lmtest_p_table_siglags_roc_data_joint.iloc[5]["FPR"], GC_II_10k_lmtest_p_table_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black", label = "α = 0.05")
plt.plot(GC_II_10k_lmtest_p_table_siglags_roc_data_joint.iloc[10]["FPR"], GC_II_10k_lmtest_p_table_siglags_roc_data_joint.iloc[10]["TPR"], "^", c="black",label = "α = 0.10")
plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("lmtest")
plt.legend()
plt.savefig(
    r"C:\Users\Fabia\OneDrive\Desktop\Studium\Master\Semester III\Seminar Financial Data Analytics\Submission\GC_II_lmtest.pdf",
    bbox_inches="tight")


#GrangerFunc
plt.figure()
plt.plot(
    GC_II_GrangerFunc_p_table_joint_siglags_roc_data_joint["FPR"],
    GC_II_GrangerFunc_p_table_joint_siglags_roc_data_joint["TPR"],
    lw=2,
    c="gray",
    label = "1k samples"
)
plt.plot(GC_II_GrangerFunc_p_table_joint_siglags_roc_data_joint.iloc[1]["FPR"], GC_II_GrangerFunc_p_table_joint_siglags_roc_data_joint.iloc[1]["TPR"], "o", c="black")
plt.plot(GC_II_GrangerFunc_p_table_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_II_GrangerFunc_p_table_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")
plt.plot(GC_II_GrangerFunc_p_table_joint_siglags_roc_data_joint.iloc[10]["FPR"], GC_II_GrangerFunc_p_table_joint_siglags_roc_data_joint.iloc[10]["TPR"], "^", c="black")
plt.plot(
    GC_II_10k_GrangerFunc_p_table_siglags_roc_data_joint["FPR"],
    GC_II_10k_GrangerFunc_p_table_siglags_roc_data_joint["TPR"],
    lw=2,
    c="gray",
    label = "10k samples",
    linestyle = "--"
)
plt.plot(GC_II_10k_GrangerFunc_p_table_siglags_roc_data_joint.iloc[1]["FPR"], GC_II_10k_GrangerFunc_p_table_siglags_roc_data_joint.iloc[1]["TPR"], "o", c="black", label = "α = 0.01")
plt.plot(GC_II_10k_GrangerFunc_p_table_siglags_roc_data_joint.iloc[5]["FPR"], GC_II_10k_GrangerFunc_p_table_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black", label = "α = 0.05")
plt.plot(GC_II_10k_GrangerFunc_p_table_siglags_roc_data_joint.iloc[10]["FPR"], GC_II_10k_GrangerFunc_p_table_siglags_roc_data_joint.iloc[10]["TPR"], "^", c="black",label = "α = 0.10")
plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("GrangerFunc")
plt.legend()
plt.savefig(
    r"C:\Users\Fabia\OneDrive\Desktop\Studium\Master\Semester III\Seminar Financial Data Analytics\Submission\GC_II_GrangerFunc.pdf",
    bbox_inches="tight")



#NlinTS_GC
plt.figure()
plt.plot(
    GC_II_NlinTS_p_table_joint_siglags_roc_data_joint["FPR"],
    GC_II_NlinTS_p_table_joint_siglags_roc_data_joint["TPR"],
    lw=2,
    c="gray",
    label = "1k samples"
)
plt.plot(GC_II_NlinTS_p_table_joint_siglags_roc_data_joint.iloc[1]["FPR"], GC_II_NlinTS_p_table_joint_siglags_roc_data_joint.iloc[1]["TPR"], "o", c="black")
plt.plot(GC_II_NlinTS_p_table_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_II_NlinTS_p_table_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")
plt.plot(GC_II_NlinTS_p_table_joint_siglags_roc_data_joint.iloc[10]["FPR"], GC_II_NlinTS_p_table_joint_siglags_roc_data_joint.iloc[10]["TPR"], "^", c="black")
plt.plot(
    GC_II_10k_NlinTS_p_table_siglags_roc_data_joint["FPR"],
    GC_II_10k_NlinTS_p_table_siglags_roc_data_joint["TPR"],
    lw=2,
    c="gray",
    label = "10k samples",
    linestyle = "--"
)
plt.plot(GC_II_10k_NlinTS_p_table_siglags_roc_data_joint.iloc[1]["FPR"], GC_II_10k_NlinTS_p_table_siglags_roc_data_joint.iloc[1]["TPR"], "o", c="black", label = "α = 0.01")
plt.plot(GC_II_10k_NlinTS_p_table_siglags_roc_data_joint.iloc[5]["FPR"], GC_II_10k_NlinTS_p_table_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black", label = "α = 0.05")
plt.plot(GC_II_10k_NlinTS_p_table_siglags_roc_data_joint.iloc[10]["FPR"], GC_II_10k_NlinTS_p_table_siglags_roc_data_joint.iloc[10]["TPR"], "^", c="black",label = "α = 0.10")
plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("causality.test")
plt.legend()
plt.savefig(
    r"C:\Users\Fabia\OneDrive\Desktop\Studium\Master\Semester III\Seminar Financial Data Analytics\Submission\GC_II_causality.test.pdf",
    bbox_inches="tight")

#NlinTS_NN
plt.figure()
plt.plot(
    GC_II_NlinTS_NN_p_table_joint_siglags_roc_data_joint["FPR"],
    GC_II_NlinTS_NN_p_table_joint_siglags_roc_data_joint["TPR"],
    lw=2,
    c="gray",
    label = "1k samples"
)
plt.plot(GC_II_NlinTS_NN_p_table_joint_siglags_roc_data_joint.iloc[1]["FPR"], GC_II_NlinTS_NN_p_table_joint_siglags_roc_data_joint.iloc[1]["TPR"], "o", c="black")
plt.plot(GC_II_NlinTS_NN_p_table_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_II_NlinTS_NN_p_table_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")
plt.plot(GC_II_NlinTS_NN_p_table_joint_siglags_roc_data_joint.iloc[10]["FPR"],GC_II_NlinTS_NN_p_table_joint_siglags_roc_data_joint.iloc[10]["TPR"], "^", c="black")
plt.plot(
    GC_II_10k_NlinTS_NN_p_table_siglags_roc_data_joint["FPR"],
    GC_II_10k_NlinTS_NN_p_table_siglags_roc_data_joint["TPR"],
    lw=2,
    c="gray",
    label = "10k samples",
    linestyle = "--"
)
plt.plot(GC_II_10k_NlinTS_NN_p_table_siglags_roc_data_joint.iloc[1]["FPR"], GC_II_10k_NlinTS_NN_p_table_siglags_roc_data_joint.iloc[1]["TPR"], "o", c="black", label = "α = 0.01")
plt.plot(GC_II_10k_NlinTS_NN_p_table_siglags_roc_data_joint.iloc[5]["FPR"], GC_II_10k_NlinTS_NN_p_table_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black", label = "α = 0.05")
plt.plot(GC_II_10k_NlinTS_NN_p_table_siglags_roc_data_joint.iloc[10]["FPR"],GC_II_10k_NlinTS_NN_p_table_siglags_roc_data_joint.iloc[10]["TPR"], "^", c="black",label = "α = 0.10")
plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("nlin_causality.test")
plt.legend()
plt.savefig(
    r"C:\Users\Fabia\OneDrive\Desktop\Studium\Master\Semester III\Seminar Financial Data Analytics\Submission\10k_nlin_causality.test.pdf",
    bbox_inches="tight")



#Coefficient plot

lag_2_coefficients = []
lag_3_coefficients = []
lag_4_coefficients = []
lag_5_coefficients = []

for p in lags:
    for tensor in range(tensor_count):
        globals()["lag_" + str(p) + "_coefficients"].append(globals()["lag_" + str(p) + "_tensor_" + str(tensor+1)][10][0][1])

lag_coefficients = pd.DataFrame(data = zip(lag_2_coefficients, lag_3_coefficients, lag_4_coefficients, lag_5_coefficients))

lag_2 = [2] * 30
lag_3 = [3] * 30
lag_4 = [4] * 30
lag_5 = [5] * 30

lag_mat = pd.DataFrame(data = zip(lag_2, lag_3, lag_4, lag_5))

fig = plt.figure(figsize = (20,4))
ax = plt.axes(projection = "3d")
ax.scatter3D(lag_mat, lag_coefficients,GC_II_NonlinC_NN_p_table.iloc[1:5,:].T, c = "red", label = "nonlincausality_NN")
ax.scatter3D(lag_mat, lag_coefficients,GC_II_NlinTS_NN_p_table.iloc[1:5,:].T, c = "blue", label = "nlin_causality.test")
#ax.scatter3D(lag_mat, lag_coefficients,GC_II_NonlinC_GRU_p_table.iloc[1:5,:].T, c = "blue", label = "NonlinC_GRU")
#ax.scatter3D(lag_mat, lag_coefficients,GC_II_NonlinC_LSTM_p_table.iloc[1:5,:].T, c = "green", label = "NonlinC_LSTM")
ax.scatter3D(lag_mat, lag_coefficients,GC_II_statsmodels_table_p.iloc[1:5,:].T, c = "black", label = "OLS regressions")
ax.set_ylabel("Coefficient")
ax.set_xlabel("Lagparameter")
ax.set_zlabel("P Value")
#ax.set_zlim(0,0.1)
plt.legend(loc = "upper right")
plt.savefig(
    r"C:\Users\Fabia\OneDrive\Desktop\Studium\Master\Semester III\Seminar Financial Data Analytics\Submission\coef3d_NonlinC.pdf",
    bbox_inches="tight")


fig = plt.figure(figsize = (20,4))
ax = plt.axes(projection = "3d")
ax.scatter3D(lag_mat, lag_coefficients,GC_II_NonlinC_NN_p_table.iloc[1:5,:].T, c = "red", label = "nonlincausality_NN")
ax.scatter3D(lag_mat, lag_coefficients,GC_II_NlinTS_NN_p_table.iloc[1:5,:].T, c = "blue", label = "nlin_causality")
#ax.scatter3D(lag_mat, lag_coefficients,GC_II_NonlinC_GRU_p_table.iloc[1:5,:].T, c = "blue", label = "NonlinC_GRU")
#ax.scatter3D(lag_mat, lag_coefficients,GC_II_NonlinC_LSTM_p_table.iloc[1:5,:].T, c = "green", label = "NonlinC_LSTM")
ax.scatter3D(lag_mat, lag_coefficients,GC_II_statsmodels_table_p.iloc[1:5,:].T, c = "black", label = "OLS regression")
ax.set_ylabel("Coefficient")
ax.set_xlabel("Lagparameter")
ax.set_zlabel("P Value")
ax.set_zlim(0,0.1)
plt.legend(loc = "upper right")
plt.savefig(
    r"C:\Users\Fabia\OneDrive\Desktop\Studium\Master\Semester III\Seminar Financial Data Analytics\Submission\coef3d_NonlinC_zoom.pdf",
    bbox_inches="tight")

from matplotlib.ticker import FormatStrFormatter

fig = plt.figure(figsize = (20,4))
ax = plt.axes(projection = "3d")
ax.scatter3D(lag_mat, lag_coefficients,GC_II_NonlinC_GRU_p_table.iloc[1:5,:].T, c = "blue", label = "NonlinC_GRU")
ax.scatter3D(lag_mat, lag_coefficients,GC_II_NonlinC_LSTM_p_table.iloc[1:5,:].T, c = "green", label = "NonlinC_LSTM")
ax.scatter3D(lag_mat, lag_coefficients,GC_II_statsmodels_table_p.iloc[1:5,:].T, c = "black", label = "OLS Regression")
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.set_ylabel("Coefficient")
ax.set_xlabel("Lagparameter")
ax.set_zlabel("P Value")
ax.set_zlim(0,0.1)
plt.legend(loc = "upper right")
plt.savefig(
    r"C:\Users\Fabia\OneDrive\Desktop\Studium\Master\Semester III\Seminar Financial Data Analytics\Submission\coef3d_RNN.pdf",
    bbox_inches="tight")


fig = plt.figure(figsize = (20,4))
ax = plt.axes(projection = "3d")
ax.scatter3D(lag_mat, lag_coefficients,GC_II_GrangerFunc_p_table.iloc[1:5,:].T, c = "black", label = "OLS Regression")
ax.scatter3D(lag_mat, lag_coefficients,GC_II_NlinTS_p_table.iloc[1:5,:].T, c = "red", label = "NlinTS")
ax.set_ylabel("Coefficient")
ax.set_xlabel("Lagparameter")
ax.set_zlabel("P Value")
ax.set_zlim(0,0.05)
plt.legend(loc = "upper right")
plt.savefig(
    r"C:\Users\Fabia\OneDrive\Desktop\Studium\Master\Semester III\Seminar Financial Data Analytics\Submission\coef3d_VLTimeSeries.pdf",
    bbox_inches="tight")


#statsmodels

statsmodels_p_average = GC_II_statsmodels_table_p.iloc[1:5,:].mean(axis = 1)
NonlinC_NN_p_average = GC_II_NonlinC_NN_p_table.iloc[1:5,:].mean(axis = 1)
NonlinC_GRU_p_average = GC_II_NonlinC_GRU_p_table.iloc[1:5,:].mean(axis = 1)
NonlinC_LSTM_p_average = GC_II_NonlinC_LSTM_p_table.iloc[1:5,:].mean(axis = 1)
GrangerFunc_p_average = GC_II_GrangerFunc_p_table.iloc[1:5,:].mean(axis = 1)
NlinTS_p_average = GC_II_NlinTS_p_table.iloc[1:5,:].mean(axis = 1)

lag_average = []

for p in lags:
    lag_average.append(sum(globals()["lag_" + str(p) + "_coefficients"])/len(globals()["lag_" + str(p) + "_coefficients"]))


ax.scatter3D(lags, lag_average,statsmodels_p_average, c = "blue", label = "statsmodels")




fig = plt.figure(figsize = (20,4))
ax = plt.axes(projection = "3d")
ax.scatter3D(lags, lag_average,statsmodels_p_average, c = "blue", label = "statsmodels")
ax.plot3D(lags, lag_average,statsmodels_p_average, c = "blue")
ax.scatter3D(lags, lag_average,NonlinC_NN_p_average, c = "red", label = "NonlinC_NN")
ax.plot3D(lags, lag_average,NonlinC_NN_p_average, c = "red")
ax.scatter3D(lags, lag_average,NonlinC_GRU_p_average, c = "yellow",label = "NonlinC_GRU")
ax.plot3D(lags, lag_average,NonlinC_GRU_p_average, c = "yellow")
ax.scatter3D(lags, lag_average,NonlinC_LSTM_p_average, c = "orange", label = "NonlinC_LSTM")
ax.plot3D(lags, lag_average,NonlinC_LSTM_p_average, c = "orange")
ax.scatter3D(lags, lag_average,NlinTS_p_average, c = "grey", label = "NlinTS")
ax.plot3D(lags, lag_average,NlinTS_p_average, c = "grey")
ax.scatter3D(lags, lag_average,GrangerFunc_p_average, c = "black", label = "GrangerFunc")
ax.plot3D(lags, lag_average,GrangerFunc_p_average, c = "black")
ax.set_ylabel("Coefficient")
ax.set_xlabel("Lagparameter")
ax.set_zlabel("P Value")
plt.legend(loc = "upper right")
plt.savefig(
    r"C:\Users\Fabia\OneDrive\Desktop\Studium\Master\Semester III\Seminar Financial Data Analytics\Submission\condensed_coef3d.pdf",
    bbox_inches="tight")



# rotate the axes and update
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)

# alllags
createROCcurve(alpha_seq, GC_II_lmtest_p_table)
createROCcurve(alpha_seq, GC_II_NlinTS_p_table)
createROCcurve(alpha_seq, GC_II_NlinTS_NN_p_table)
createROCcurve(alpha_seq, GC_II_NonlinC_NN_p_table)
createROCcurve(alpha_seq, GC_II_NonlinC_GRU_p_table)

roc_data = pd.DataFrame(columns=["alpha", "TPR", "FPR"], index=[i for i in range(len(alpha_seq))])

for alpha in range(len(alpha_seq)):
    roc = roc_adj(threshold=alpha_seq[alpha], yhat=GC_II_lmtest_p_table_siglags)
    roc_data.iloc[alpha]["alpha"] = roc[0]
    roc_data.iloc[alpha]["TPR"] = roc[1]
    roc_data.iloc[alpha]["FPR"] = roc[2]

plt.figure()
plt.plot(
    GC_II_NlinTS_NN_p_table_joint_siglags_roc_data_joint["FPR"],
    GC_II_NlinTS_NN_p_table_joint_siglags_roc_data_joint["TPR"],
    lw=lw,
    c="gray",
    label = "NlinTS"
)
plt.plot(GC_II_NlinTS_NN_p_table_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_II_NlinTS_NN_p_table_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")
plt.plot(
    GC_III_NlinTS_NN_p_table_joint_siglags_roc_data_joint["FPR"],
    GC_III_NlinTS_NN_p_table_joint_siglags_roc_data_joint["TPR"],
    lw=lw,
    c="gray",
    linestyle = "-."
)
plt.plot(GC_III_NlinTS_NN_p_table_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_III_NlinTS_NN_p_table_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")
plt.plot(
    GC_II_statsmodels_table_p_joint_siglags_roc_data_joint["FPR"],
    GC_II_statsmodels_table_p_joint_siglags_roc_data_joint["TPR"],
    lw=2,
    c="blue",
    label="statsmodels"
)
plt.plot(GC_II_statsmodels_table_p_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_II_statsmodels_table_p_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")
plt.plot(
    GC_III_statsmodels_table_p_joint_siglags_roc_data_joint["FPR"],
    GC_III_statsmodels_table_p_joint_siglags_roc_data_joint["TPR"],
    lw=lw,
    c="blue",
    linestyle="-."
)
plt.plot(GC_III_statsmodels_table_p_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_III_statsmodels_table_p_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")
plt.plot(
    GC_II_NonlinC_NN_p_table_joint_siglags_roc_data_joint["FPR"],
    GC_II_NonlinC_NN_p_table_joint_siglags_roc_data_joint["TPR"],
    lw=lw,
    c="red",
    label="NonlinC_NN"
)
plt.plot(GC_II_NonlinC_NN_p_table_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_II_NonlinC_NN_p_table_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")

plt.plot(
    GC_III_NonlinC_NN_p_table_joint_siglags_roc_data_joint["FPR"],
    GC_III_NonlinC_NN_p_table_joint_siglags_roc_data_joint["TPR"],
    lw=lw,
    c="red",
    linestyle="-."
)
plt.plot(GC_III_NonlinC_NN_p_table_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_III_NonlinC_NN_p_table_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")

plt.plot(
    GC_II_NonlinC_GRU_p_table_joint_siglags_roc_data_joint["FPR"],
    GC_II_NonlinC_GRU_p_table_joint_siglags_roc_data_joint["TPR"],
    lw=lw,
    c="orange",
    label="NonlinC_GRU"
)
plt.plot(GC_II_NonlinC_GRU_p_table_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_II_NonlinC_GRU_p_table_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")

plt.plot(
    GC_III_NonlinC_GRU_p_table_joint_siglags_roc_data_joint["FPR"],
    GC_III_NonlinC_GRU_p_table_joint_siglags_roc_data_joint["TPR"],
    lw=lw,
    c="orange",
    linestyle="-."
)
plt.plot(GC_III_NonlinC_GRU_p_table_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_III_NonlinC_GRU_p_table_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")
plt.plot(
    GC_II_NonlinC_LSTM_p_table_joint_siglags_roc_data_joint["FPR"],
    GC_II_NonlinC_LSTM_p_table_joint_siglags_roc_data_joint["TPR"],
    lw=lw,
    c="green",
    label="NonlinC_LSTM"
)
plt.plot(GC_II_NonlinC_LSTM_p_table_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_II_NonlinC_LSTM_p_table_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")

plt.plot(
    GC_III_NonlinC_LSTM_p_table_joint_siglags_roc_data_joint["FPR"],
    GC_III_NonlinC_LSTM_p_table_joint_siglags_roc_data_joint["TPR"],
    lw=lw,
    c="green",
    linestyle="-."
)
plt.plot(GC_III_NonlinC_LSTM_p_table_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_III_NonlinC_LSTM_p_table_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black", label = "α = 0.05")
plt.legend()
plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("Correlation sensitivity in noise structure")
plt.savefig(
    r"C:\Users\Fabia\OneDrive\Desktop\Studium\Master\Semester III\Seminar Financial Data Analytics\Submission\Error-Cor-Impact_Py.pdf",
    bbox_inches="tight")
plt.legend(loc="lower right")
plt.show()

#############################################
######          GC - Inference         ######
###### Error Term - Sensitivity III    ######
#############################################

# GC III - Settings

# Set Hyperparameters

sample_size = 1000
tensor_count = 30
lags = [2, 3, 4, 5]
lag_amount = len(lags)
test_lags_upt = max(lags) + 3

# Determine E
sigma = [0.2, 0.1]  # standard deviation of first and second time series
corr = 0.8  # correlation

covs = [[sigma[0] ** 2, sigma[0] * sigma[1] * corr],
        [sigma[0] * sigma[1] * corr, sigma[1] ** 2]]

np.random.seed(42)
noise = np.random.multivariate_normal(mean=[0.1, 0.1], cov=covs, size=(sample_size + 2 * lag_amount,))

alpha_seq = np.linspace(0, 0.1, 11).round(2)
alpha_seq_joint = np.linspace(0, 1, 101).round(2)

# Show Correlation
plt.scatter(x=noise[:, 1], y=noise[:, 0], c="gray")
plt.xlabel("Noise - x")
plt.ylabel("Noise - y")
plt.savefig(
    r"C:\Users\Fabia\OneDrive\Desktop\Studium\Master\Semester III\Seminar Financial Data Analytics\Submission\GC-Noise-2.pdf",
    bbox_inches="tight")

# Show first 50 samples
plt.plot(dat_tensor_1[5:50, 0], label="y", c="red")
plt.plot(dat_tensor_1[5:50, 1], label="x", c="blue")
plt.xlabel("Number of samples")
plt.ylabel("Value")
plt.legend()
plt.savefig(
    r"C:\Users\Fabia\OneDrive\Desktop\Studium\Master\Semester III\Seminar Financial Data Analytics\Submission\GC-2-first50.pdf",
    bbox_inches="tight")

# import results from R ###
import_data_R("III", 5)

# import results from Python ###
import_data_Py("III", 5, both = False)

# create ROC tables
ROC_composition("III", alpha_seq_joint)

plt.scatter(y = GC_III_statsmodels_table_p.iloc[1,:], x = [i for i in range(GC_III_statsmodels_table_p.shape[1])])

plt.figure(figsize = (20,4))
plt.plot(GC_III_NonlinC_NN_p_table.iloc[1,:])
plt.axhline(y=0.1, color='r', linestyle='--', label = "alpha = 0.1")
plt.axhline(y=0.05, color='b', linestyle='--', label = "alpha = 0.05")
plt.axhline(y=0.01, color='g', linestyle='--', label = "alpha = 0.01")
plt.legend()

# Result analysis ###

# ROC Data

alpha_seq = np.linspace(0, 0.1, 11).round(2)
alpha_seq_joint = np.linspace(0, 1, 101).round(2)

# ROC Curves

# siglags
createROCcurve(alpha_seq_joint, GC_III_lmtest_p_table_siglags, "GC_III_lmtest_roc")
createROCcurve(alpha_seq_joint, GC_III_NlinTS_p_table_siglags, "GC_III_NlinTS_GC_roc")
createROCcurve(alpha_seq_joint, GC_III_NlinTS_NN_p_table_siglags, "GC_III_NlinTS_NN_roc")
createROCcurve(alpha_seq_joint, GC_III_NonlinC_NN_p_table_siglags, "GC_III_NonlinC_NN_roc")
createROCcurve(alpha_seq_joint, GC_III_NonlinC_GRU_p_table_siglags, "GC_III_GRU_NN_roc")
createROCcurve(alpha_seq_joint, GC_III_NonlinC_LSTM_p_table_siglags, "GC_III_LSTM_NN_roc")


# alllags
createROCcurve(alpha_seq, GC_III_lmtest_p_table)
createROCcurve(alpha_seq, GC_III_NlinTS_p_table)
createROCcurve(alpha_seq, GC_III_NlinTS_NN_p_table)
createROCcurve(alpha_seq, GC_III_NonlinC_NN_p_table)
createROCcurve(alpha_seq, GC_III_NonlinC_GRU_p_table)

# comparison

GC_II_lmtest_roc_data_siglags = pd.DataFrame(columns=["alpha", "TPR", "FPR"],
                                             index=[i for i in range(len(alpha_seq_joint))])
createROCdata(alpha_seq=alpha_seq_joint, p_values_test=GC_II_lmtest_p_table_siglags,
              df_roc=GC_II_lmtest_roc_data_siglags)

GC_II_NlinTS_roc_data_siglags = pd.DataFrame(columns=["alpha", "TPR", "FPR"],
                                             index=[i for i in range(len(alpha_seq_joint))])
createROCdata(alpha_seq=alpha_seq_joint, p_values_test=GC_II_NlinTS_p_table_siglags,
              df_roc=GC_II_NlinTS_roc_data_siglags)

GC_II_statsmodels_roc_data_siglags = pd.DataFrame(columns=["alpha", "TPR", "FPR"],
                                                  index=[i for i in range(len(alpha_seq_joint))])
createROCdata(alpha_seq=alpha_seq_joint, p_values_test=GC_II_statsmodels_p_table_siglags,
              df_roc=GC_II_statsmodels_roc_data_siglags)

GC_II_NonlinC_NN_roc_data_siglags = pd.DataFrame(columns=["alpha", "TPR", "FPR"],
                                                 index=[i for i in range(len(alpha_seq_joint))])
createROCdata(alpha_seq=alpha_seq_joint, p_values_test=GC_II_NonlinC_NN_p_table_siglags,
              df_roc=GC_II_NonlinC_NN_roc_data_siglags)

GC_II_NonlinC_GRU_roc_data_siglags = pd.DataFrame(columns=["alpha", "TPR", "FPR"],
                                                  index=[i for i in range(len(alpha_seq_joint))])
createROCdata(alpha_seq=alpha_seq_joint, p_values_test=GC_II_NonlinC_GRU_p_table_siglags,
              df_roc=GC_II_NonlinC_GRU_roc_data_siglags)

GC_III_lmtest_roc_data_siglags = pd.DataFrame(columns=["alpha", "TPR", "FPR"],
                                              index=[i for i in range(len(alpha_seq_joint))])
createROCdata(alpha_seq=alpha_seq_joint, p_values_test=GC_III_lmtest_p_table_siglags,
              df_roc=GC_III_lmtest_roc_data_siglags)

GC_III_NlinTS_roc_data_siglags = pd.DataFrame(columns=["alpha", "TPR", "FPR"],
                                              index=[i for i in range(len(alpha_seq_joint))])
createROCdata(alpha_seq=alpha_seq_joint, p_values_test=GC_III_NlinTS_p_table_siglags,
              df_roc=GC_III_NlinTS_roc_data_siglags)

GC_III_statsmodels_roc_data_siglags = pd.DataFrame(columns=["alpha", "TPR", "FPR"],
                                                   index=[i for i in range(len(alpha_seq_joint))])
createROCdata(alpha_seq=alpha_seq_joint, p_values_test=GC_III_statsmodels_p_table_siglags,
              df_roc=GC_III_statsmodels_roc_data_siglags)

GC_III_NonlinC_NN_roc_data_siglags = pd.DataFrame(columns=["alpha", "TPR", "FPR"],
                                                  index=[i for i in range(len(alpha_seq_joint))])
createROCdata(alpha_seq=alpha_seq_joint, p_values_test=GC_III_NonlinC_NN_p_table_siglags,
              df_roc=GC_III_NonlinC_NN_roc_data_siglags)

GC_III_NonlinC_GRU_roc_data_siglags = pd.DataFrame(columns=["alpha", "TPR", "FPR"],
                                                   index=[i for i in range(len(alpha_seq_joint))])
createROCdata(alpha_seq=alpha_seq_joint, p_values_test=GC_III_NonlinC_GRU_p_table_siglags,
              df_roc=GC_III_NonlinC_GRU_roc_data_siglags)

plt.figure()
lw = 2
plt.plot(
    GC_II_lmtest_roc_data_siglags["FPR"],
    GC_II_lmtest_roc_data_siglags["TPR"],
    lw=lw,
    color="red",
)
plt.plot(
    GC_II_NlinTS_roc_data_siglags["FPR"],
    GC_II_NlinTS_roc_data_siglags["TPR"],
    lw=lw,
    color="blue",
)
plt.plot(
    GC_III_lmtest_roc_data_siglags["FPR"],
    GC_III_lmtest_roc_data_siglags["TPR"],
    lw=lw,
    color="red",
    linestyle="-."
)
plt.plot(
    GC_III_NlinTS_roc_data_siglags["FPR"],
    GC_III_NlinTS_roc_data_siglags["TPR"],
    lw=lw,
    color="blue",
    linestyle="-."
)

plt.figure()
plt.plot(
    GC_II_statsmodels_roc_data_siglags["FPR"],
    GC_II_statsmodels_roc_data_siglags["TPR"],
    lw=lw,
    color="darkgreen",
)
plt.plot(
    GC_II_NonlinC_NN_roc_data_siglags["FPR"],
    GC_II_NonlinC_NN_roc_data_siglags["TPR"],
    lw=lw,
    color="orange",
)
plt.plot(
    GC_II_NonlinC_GRU_roc_data_siglags["FPR"],
    GC_II_NonlinC_GRU_roc_data_siglags["TPR"],
    lw=lw,
    color="grey",
)
plt.plot(
    GC_III_statsmodels_roc_data_siglags["FPR"],
    GC_III_statsmodels_roc_data_siglags["TPR"],
    lw=lw,
    color="darkgreen",
    linestyle="-."
)
plt.plot(
    GC_III_NonlinC_NN_roc_data_siglags["FPR"],
    GC_III_NonlinC_NN_roc_data_siglags["TPR"],
    lw=lw,
    color="orange",
    linestyle="-."
)
plt.plot(
    GC_III_NonlinC_GRU_roc_data_siglags["FPR"],
    GC_III_NonlinC_GRU_roc_data_siglags["TPR"],
    lw=lw,
    color="grey",
    linestyle="-."
)

tensor_count = 30
plt.scatter(y = GC_III_statsmodels_table_p.iloc[0,:], x = [i for i in range(tensor_count)], marker = ".", label = "OLS Regression", c = "red")
plt.scatter(y = GC_III_NonlinC_NN_p_table.iloc[0,:], x = [i for i in range(tensor_count)], marker = "v", label = "NonlinC_NN", c = "red")
plt.scatter(y = GC_III_NonlinC_GRU_p_table.iloc[0,:], x = [i for i in range(tensor_count)], marker = "x", label = "NonlinC_GRU", c = "red")
plt.scatter(y = GC_III_NonlinC_LSTM_p_table.iloc[0,:], x = [i for i in range(tensor_count)], marker = "s", label = "NonlinC_LSTM", c = "red")
plt.scatter(y = GC_III_NlinTS_NN_p_table.iloc[0,:], x = [i for i in range(tensor_count)], marker = "D", label = "NlinTS", c = "red")
plt.scatter(y = GC_II_statsmodels_table_p.iloc[0,:], x = [i for i in range(tensor_count)], marker = ".", label = "OLS Regression", c = "blue")
plt.scatter(y = GC_II_NonlinC_NN_p_table.iloc[0,:], x = [i for i in range(tensor_count)], marker = "v", label = "NonlinC_NN", c = "blue")
plt.scatter(y = GC_II_NonlinC_GRU_p_table.iloc[0,:], x = [i for i in range(tensor_count)], marker = "x", label = "NonlinC_GRU", c = "blue")
plt.scatter(y = GC_II_NonlinC_LSTM_p_table.iloc[0,:], x = [i for i in range(tensor_count)], marker = "s", label = "NonlinC_LSTM", c = "blue")
plt.scatter(y = GC_II_NlinTS_NN_p_table.iloc[0,:], x = [i for i in range(tensor_count)], marker = "D", label = "NlinTS", c = "blue")
plt.plot([0,30], [0.1,0.1], c = "black", linestyle = "--", label = "10%-significance level")
plt.plot([0,30], [0.05,0.05], c = "black", linestyle = "-.", label = "5%-significance level")
#plt.ylim([0.5,1])
plt.ylim([0,0.15])
#plt.legend(loc = "lower right")
plt.xlabel("Tensorindex")
plt.ylabel("P-values of lag 1")
plt.savefig(rf"C:/Users/Fabia/OneDrive/Desktop/Studium/Master/Semester III/Seminar Financial Data Analytics/Submission/pvaluesovertensors.pdf", bbox_inches="tight")


############################


# X on Y
GC_III_statsmodels_p_table_XY = pd.read_csv(
    r'C:\Users\Fabia\pythoninput\Granger Causality\Python Data\GC - Inference Sensitivity III/statsmodels_table_p_XY.csv')
GC_III_statsmodels_p_table_XY = GC_III_statsmodels_p_table_XY.set_index("Unnamed: 0")
GC_III_statsmodels_p_table_XY_siglags = GC_III_statsmodels_p_table_XY.iloc[0:5, :]

# NonlinC_NN
GC_III_NonlinC_NN_p_table_XY = pd.read_csv(
    r'C:\Users\Fabia\pythoninput\Granger Causality\Python Data\GC - Inference Sensitivity III/NonlinC_NN_p_table_XY.csv')
GC_III_NonlinC_NN_p_table_XY = GC_III_NonlinC_NN_p_table_XY.set_index("Unnamed: 0")
GC_III_NonlinC_NN_p_table_siglags_XY = GC_III_NonlinC_NN_p_table_XY.iloc[0:5, :]

plt.plot([0, max(roc_data["FPR"])], [0, max(roc_data["TPR"])], color="black", lw=lw, linestyle="--")
plt.plot(roc_data.iloc[1]["FPR"], roc_data.iloc[1]["TPR"], "x", c="black", label="α = 0.01")
plt.plot(roc_data.iloc[5]["FPR"], roc_data.iloc[5]["TPR"], "x", c="blue", label="α = 0.05")
plt.plot(roc_data.iloc[10]["FPR"], roc_data.iloc[10]["TPR"], "x", c="grey", label="α = 0.1")
plt.legend(loc="lower right")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(plot_name)

from matrix2latex import matrix2latex

matrix2latex(GC_II_NonlinC_LSTM_p_table[:,:9],"table", "tabular", "small")



#############################################
######          GC - Inference         ######
###### Error Term - Sensitivity IV     ######
#############################################

# GC IV Settings
#Set Hyperparameters

sample_size = 1000
tensor_count = 30
lags = [2,3,4,5]
lag_amount = len(lags)

#Determine E
sigma = [0.8, 0.9] #standard deviation of first and second time series
corr = 0         #correlation

covs = [[sigma[0]**2          , sigma[0]*sigma[1]*corr],
        [sigma[0]*sigma[1]*corr,          sigma[1]**2]]

np.random.seed(42)
noise = np.random.multivariate_normal(mean = [0.1,0.1], cov = covs, size=(sample_size + 2 * lag_amount,))

#Generate Data
DataGeneration(case = 1,sample_size = sample_size, lags = lags, k = 2, tensor_count = tensor_count, noise = noise)

# import results from R ###
import_data_R("IV", 5)

# import results from Python ###
import_data_Py("IV", 5)

# create ROC tables
ROC_composition("IV", alpha_seq_joint)


createROCcurve(alpha_seq_joint, GC_IV_lmtest_p_table_joint_siglags, "GC_IV_OLS_approaches")
createROCcurve(alpha_seq_joint, GC_IV_NonlinC_NN_p_table_siglags, "GC_IV_NonlinC_NN")
createROCcurve(alpha_seq_joint, GC_IV_NonlinC_GRU_p_table_siglags, "GC_IV_NonlinC_GRU")
createROCcurve(alpha_seq_joint, GC_IV_NonlinC_LSTM_p_table_siglags, "GC_IV_NonlinC_LSTM")
createROCcurve(alpha_seq_joint, GC_IV_GrangerFunc_p_table_joint_siglags, "GC_IV_GrangerFunc")
createROCcurve(alpha_seq_joint, GC_IV_NlinTS_NN_p_table_joint_siglags, "GC_IV_NlinTS_NN")


plt.figure()
lw = 2
plt.plot(
    GC_II_NlinTS_NN_p_table_joint_siglags_roc_data_joint["FPR"],
    GC_II_NlinTS_NN_p_table_joint_siglags_roc_data_joint["TPR"],
    lw=lw,
    c="gray",
    label="NlinTS"
)
plt.plot(
    GC_IV_NlinTS_NN_p_table_joint_siglags_roc_data_joint["FPR"],
    GC_IV_NlinTS_NN_p_table_joint_siglags_roc_data_joint["TPR"],
    lw=lw,
    c="gray",
    linestyle = "-."
)
plt.plot(GC_IV_NlinTS_NN_p_table_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_IV_NlinTS_NN_p_table_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")
plt.plot(
    GC_II_statsmodels_table_p_joint_siglags_roc_data_joint["FPR"],
    GC_II_statsmodels_table_p_joint_siglags_roc_data_joint["TPR"],
    lw=lw,
    c="blue",
    label="statsmodels"
)
plt.plot(GC_II_statsmodels_table_p_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_II_statsmodels_table_p_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")
plt.plot(
    GC_IV_statsmodels_table_p_joint_siglags_roc_data_joint["FPR"],
    GC_IV_statsmodels_table_p_joint_siglags_roc_data_joint["TPR"],
    lw=lw,
    c="blue",
    linestyle="-."
)
plt.plot(GC_IV_statsmodels_table_p_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_IV_statsmodels_table_p_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")
plt.plot(
    GC_II_NonlinC_NN_p_table_joint_siglags_roc_data_joint["FPR"],
    GC_II_NonlinC_NN_p_table_joint_siglags_roc_data_joint["TPR"],
    lw=lw,
    c="red",
    label="NonlinC_NN"
)
plt.plot(GC_II_NonlinC_NN_p_table_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_II_NonlinC_NN_p_table_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")
plt.plot(
    GC_IV_NonlinC_NN_p_table_joint_siglags_roc_data_joint["FPR"],
    GC_IV_NonlinC_NN_p_table_joint_siglags_roc_data_joint["TPR"],
    lw=lw,
    c="red",
    linestyle="-."
)
plt.plot(GC_IV_NonlinC_NN_p_table_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_IV_NonlinC_NN_p_table_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")
plt.plot(
    GC_II_NonlinC_GRU_p_table_joint_siglags_roc_data_joint["FPR"],
    GC_II_NonlinC_GRU_p_table_joint_siglags_roc_data_joint["TPR"],
    lw=lw,
    c="orange",
    label="NonlinC_GRU"
)
plt.plot(GC_II_NonlinC_GRU_p_table_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_II_NonlinC_GRU_p_table_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")

plt.plot(
    GC_IV_NonlinC_GRU_p_table_joint_siglags_roc_data_joint["FPR"],
    GC_IV_NonlinC_GRU_p_table_joint_siglags_roc_data_joint["TPR"],
    lw=lw,
    c="orange",
    linestyle="-."
)
plt.plot(GC_IV_NonlinC_GRU_p_table_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_IV_NonlinC_GRU_p_table_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")
plt.plot(
    GC_II_NonlinC_LSTM_p_table_joint_siglags_roc_data_joint["FPR"],
    GC_II_NonlinC_LSTM_p_table_joint_siglags_roc_data_joint["TPR"],
    lw=lw,
    c="green",
    label="NonlinC_LSTM"
)
plt.plot(GC_II_NonlinC_LSTM_p_table_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_II_NonlinC_LSTM_p_table_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")
plt.plot(
    GC_IV_NonlinC_LSTM_p_table_joint_siglags_roc_data_joint["FPR"],
    GC_IV_NonlinC_LSTM_p_table_joint_siglags_roc_data_joint["TPR"],
    lw=lw,
    c="green",
    linestyle="-."
)
plt.plot(GC_IV_NonlinC_LSTM_p_table_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_IV_NonlinC_LSTM_p_table_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black")
plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
plt.plot(GC_II_NlinTS_NN_p_table_joint_siglags_roc_data_joint.iloc[5]["FPR"], GC_II_NlinTS_NN_p_table_joint_siglags_roc_data_joint.iloc[5]["TPR"], "x", c="black", label = "α = 0.05")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("Volatility sensitivity in noise structure")
plt.legend(loc="lower right")
plt.savefig(
    r"C:\Users\Fabia\OneDrive\Desktop\Studium\Master\Semester III\Seminar Financial Data Analytics\Submission\Error-Var-Impact_Py.pdf",
    bbox_inches="tight")
plt.show()


#plt.figure(figsize = (20,4))
plt.scatter(y = GC_IV_statsmodels_table_p.iloc[3,:], x = [i for i in range(tensor_count)], marker = ".", label = "statsmodels", c = "black")
plt.scatter(y = GC_IV_NonlinC_NN_p_table.iloc[3,:], x = [i for i in range(tensor_count)], marker = "v", label = "NonlinC_NN", c = "black")
plt.scatter(y = GC_IV_NonlinC_GRU_p_table.iloc[3,:], x = [i for i in range(tensor_count)], marker = "x", label = "NonlinC_GRU", c = "black")
plt.scatter(y = GC_IV_NonlinC_LSTM_p_table.iloc[3,:], x = [i for i in range(tensor_count)], marker = "s", label = "NonlinC_LSTM", c = "black")
plt.scatter(y = GC_IV_GrangerFunc_p_table.iloc[3,:], x = [i for i in range(tensor_count)], marker = "x", label = "GrangerFunc", c = "black")
plt.scatter(y = GC_IV_NlinTS_NN_p_table.iloc[3,:], x = [i for i in range(tensor_count)], marker = "D", label = "NlinTS", c = "black")
plt.plot([0,30], [0.05,0.05], c = "red", linestyle = "--", label = "5%-significance level")
plt.ylim([0,0.075])
plt.legend(loc = "lower right")
plt.xlabel("Tensorindex")
plt.ylabel("P-values")
plt.savefig(rf"C:/Users/Fabia/OneDrive/Desktop/Studium/Master/Semester III/Seminar Financial Data Analytics/Submission/pvaluesovertensors.pdf", bbox_inches="tight")

#plt.scatter(y = GC_IV_NlinTS_NN_p_table.iloc[4,:], x = [i for i in range(tensor_count)])

plt.plot(GC_IV_GrangerFunc_F_table.iloc[:,0])

for tensor in range(tensor_count):
    plt.figure(figsize = (20,4))
    plt.plot(GC_IV_NonlinC_NN_F_table.iloc[:7,tensor])
    plt.plot(GC_III_NonlinC_NN_F_table.iloc[:7,tensor])


#############################################
######          GC - Inference         ######
######   Error Term - Sensitivity V    ######
#############################################

# GC II - Settings

sample_size = 1000
tensor_count = 30
lags = [2, 3, 4, 5]
lag_amount = len(lags)

# Determine E
sigma = [0.5, 0.5]  # standard deviation of first and second time series
corr = 0  # correlation

covs = [[sigma[0] ** 2, sigma[0] * sigma[1] * corr],
        [sigma[0] * sigma[1] * corr, sigma[1] ** 2]]

np.random.seed(42)
noise = np.random.multivariate_normal(mean=[0, 0], cov=covs, size=(sample_size + 2 * lag_amount,))

# Generate Data
DataGeneration(case=1, sample_size=sample_size, lags=lags, k=2, tensor_count=tensor_count, noise=noise)

# import results from R ###
import_data_R("V", 5)

# import results from Python ###
import_data_Py("V", 5)

#create ROC data
ROC_composition("V",alpha_seq_joint)

conf_mat(0.05, GC_V_statsmodels_table_p)

createROCcurve(alpha_seq_joint, GC_V_lmtest_p_table_joint_siglags, "GC_V_statsmodels_lmtest_grangercausalitytest")
createROCcurve(alpha_seq_joint, GC_V_GrangerFunc_p_table_joint_siglags, "GC_V_GrangerFunc")
createROCcurve(alpha_seq_joint, GC_V_NlinTS_p_table_joint_siglags, "GC_V_NlinTS")
createROCcurve(alpha_seq_joint, GC_V_NlinTS_p_table_joint_siglags, "GC_V_NlinTS")
createROCcurve(alpha_seq_joint, GC_V_NlinTS_NN_p_table_joint_siglags, "GC_V_NlinTS_NN")
createROCcurve(alpha_seq_joint, GC_V_NonlinC_NN_p_table_siglags, "GC_V_NonlinC_NN")
createROCcurve(alpha_seq_joint, GC_V_NonlinC_GRU_p_table_siglags, "GC_V_NonlinC_GRU")
createROCcurve(alpha_seq_joint, GC_V_NonlinC_LSTM_p_table_siglags, "GC_V_NonlinC_LSTM")


#############################################
######          GC - Inference         ######
######   Error Term - Sensitivity VI   ######
#############################################

# GC II - Settings

sample_size = 1000
tensor_count = 30
lags = [2, 3, 4, 5]
lag_amount = len(lags)

# Determine E
sigma = [0.5, 0.5]  # standard deviation of first and second time series
corr = 0  # correlation

covs = [[sigma[0] ** 2, sigma[0] * sigma[1] * corr],
        [sigma[0] * sigma[1] * corr, sigma[1] ** 2]]

np.random.seed(42)
noise = np.random.multivariate_normal(mean=[0, 0], cov=covs, size=(sample_size + 2 * lag_amount,))

# Generate Data
DataGeneration(case=1, sample_size=sample_size, lags=lags, k=2, tensor_count=tensor_count, noise=noise)

# import results from R ###
import_data_R("VI", 5)

# import results from Python ###
import_data_Py("VI", 5)

#create ROC data
ROC_composition("VI",alpha_seq_joint)

createROCcurve(alpha_seq_joint, GC_VI_lmtest_p_table_joint_siglags, "GC_VI_lmtest")
createROCcurve(alpha_seq_joint, GC_VI_statsmodels_table_p_joint_siglags, "GC_VI_statsmodels")
createROCcurve(alpha_seq_joint, GC_VI_GrangerFunc_p_table_joint_siglags, "GC_VI_GrangerFunc")
createROCcurve(alpha_seq_joint, GC_VI_NlinTS_p_table_joint_siglags, "GC_VI_NlinTS_GC")
createROCcurve(alpha_seq_joint, GC_VI_NlinTS_NN_p_table_joint_siglags, "GC_VI_NlinTS_NN")
createROCcurve(alpha_seq_joint, GC_VI_NonlinC_NN_p_table_joint_siglags, "GC_VI_NonlinC_NN")
createROCcurve(alpha_seq_joint, GC_VI_NonlinC_GRU_p_table_joint_siglags, "GC_VI_NonlinC_GRU")
createROCcurve(alpha_seq_joint, GC_VI_NonlinC_LSTM_p_table_joint_siglags, "GC_VI_NonlinC_LSTM")



plt.figure()
lw = 2
plt.plot(
    GC_VI_lmtest_p_table_siglags_roc_data["FPR"],
    GC_VI_lmtest_p_table_siglags_roc_data["TPR"],
    lw=lw,
    c="thistle",
    label="lmtest"
)
plt.plot(GC_VI_lmtest_p_table_siglags_roc_data.iloc[1]["FPR"], GC_VI_lmtest_p_table_siglags_roc_data.iloc[1]["TPR"], "x", c="black", label="α = 0.01")
plt.plot(GC_VI_lmtest_p_table_siglags_roc_data.iloc[5]["FPR"], GC_VI_lmtest_p_table_siglags_roc_data.iloc[5]["TPR"], "x", c="blue", label="α = 0.05")
plt.plot(GC_VI_lmtest_p_table_siglags_roc_data.iloc[10]["FPR"], GC_VI_lmtest_p_table_siglags_roc_data.iloc[10]["TPR"], "x", c="grey", label="α = 0.10")

plt.plot(
    GC_VI_NlinTS_p_table_siglags_roc_data["FPR"],
    GC_VI_NlinTS_p_table_siglags_roc_data["TPR"],
    lw=lw,
    c="black",
    label="NlinTS_NN"
)
plt.plot(GC_VI_NlinTS_p_table_siglags_roc_data.iloc[1]["FPR"], GC_VI_NlinTS_p_table_siglags_roc_data.iloc[1]["TPR"], "x", c="black")
plt.plot(GC_VI_NlinTS_p_table_siglags_roc_data.iloc[5]["FPR"], GC_VI_NlinTS_p_table_siglags_roc_data.iloc[5]["TPR"], "x", c="blue")
plt.plot(GC_VI_NlinTS_p_table_siglags_roc_data.iloc[10]["FPR"], GC_VI_NlinTS_p_table_siglags_roc_data.iloc[10]["TPR"], "x", c="grey")

plt.plot(
    GC_VI_GrangerFunc_p_table_siglags_roc_data["FPR"],
    GC_VI_GrangerFunc_p_table_siglags_roc_data["TPR"],
    lw=lw,
    c="yellow",
    label="GrangerFunc"
)
plt.plot(GC_VI_GrangerFunc_p_table_siglags_roc_data.iloc[1]["FPR"], GC_VI_GrangerFunc_p_table_siglags_roc_data.iloc[1]["TPR"], "x", c="black")
plt.plot(GC_VI_GrangerFunc_p_table_siglags_roc_data.iloc[5]["FPR"], GC_VI_GrangerFunc_p_table_siglags_roc_data.iloc[5]["TPR"], "x", c="blue")
plt.plot(GC_VI_GrangerFunc_p_table_siglags_roc_data.iloc[10]["FPR"], GC_VI_GrangerFunc_p_table_siglags_roc_data.iloc[10]["TPR"], "x", c="grey")

plt.plot(
    GC_VI_NlinTS_NN_p_table_siglags_roc_data["FPR"],
    GC_VI_NlinTS_NN_p_table_siglags_roc_data["TPR"],
    lw=lw,
    c="gray",
    label="NlinTS"
)
plt.plot(GC_VI_NlinTS_NN_p_table_siglags_roc_data.iloc[1]["FPR"], GC_VI_NlinTS_NN_p_table_siglags_roc_data.iloc[1]["TPR"], "x", c="black")
plt.plot(GC_VI_NlinTS_NN_p_table_siglags_roc_data.iloc[5]["FPR"], GC_VI_NlinTS_NN_p_table_siglags_roc_data.iloc[5]["TPR"], "x", c="blue")
plt.plot(GC_VI_NlinTS_NN_p_table_siglags_roc_data.iloc[10]["FPR"], GC_VI_NlinTS_NN_p_table_siglags_roc_data.iloc[10]["TPR"], "x", c="grey")

plt.plot(
    GC_VI_statsmodels_table_p_siglags_roc_data["FPR"],
    GC_VI_statsmodels_table_p_siglags_roc_data["TPR"],
    lw=lw,
    c="blue",
    label="statsmodels"
)
plt.plot(GC_VI_statsmodels_table_p_siglags_roc_data.iloc[1]["FPR"], GC_VI_statsmodels_table_p_siglags_roc_data.iloc[1]["TPR"], "x", c="black")
plt.plot(GC_VI_statsmodels_table_p_siglags_roc_data.iloc[5]["FPR"], GC_VI_statsmodels_table_p_siglags_roc_data.iloc[5]["TPR"], "x", c="blue")
plt.plot(GC_VI_statsmodels_table_p_siglags_roc_data.iloc[10]["FPR"], GC_VI_statsmodels_table_p_siglags_roc_data.iloc[10]["TPR"], "x", c="grey")
plt.plot(
    GC_VI_NonlinC_NN_p_table_siglags_roc_data["FPR"],
    GC_VI_NonlinC_NN_p_table_siglags_roc_data["TPR"],
    lw=lw,
    c="red",
    label="NonlinC_NN"
)
plt.plot(GC_VI_NonlinC_NN_p_table_siglags_roc_data.iloc[1]["FPR"], GC_VI_NonlinC_NN_p_table_siglags_roc_data.iloc[1]["TPR"], "x", c="black")
plt.plot(GC_VI_NonlinC_NN_p_table_siglags_roc_data.iloc[5]["FPR"], GC_VI_NonlinC_NN_p_table_siglags_roc_data.iloc[5]["TPR"], "x", c="blue")
plt.plot(GC_VI_NonlinC_NN_p_table_siglags_roc_data.iloc[10]["FPR"], GC_VI_NonlinC_NN_p_table_siglags_roc_data.iloc[10]["TPR"], "x", c="grey")
plt.plot(
    GC_VI_NonlinC_GRU_p_table_siglags_roc_data["FPR"],
    GC_VI_NonlinC_GRU_p_table_siglags_roc_data["TPR"],
    lw=lw,
    c="orange",
    label="NonlinC_GRU"
)
plt.plot(GC_VI_NonlinC_GRU_p_table_siglags_roc_data.iloc[1]["FPR"], GC_VI_NonlinC_GRU_p_table_siglags_roc_data.iloc[1]["TPR"], "x", c="black")
plt.plot(GC_VI_NonlinC_GRU_p_table_siglags_roc_data.iloc[5]["FPR"], GC_VI_NonlinC_GRU_p_table_siglags_roc_data.iloc[5]["TPR"], "x", c="blue")
plt.plot(GC_VI_NonlinC_GRU_p_table_siglags_roc_data.iloc[10]["FPR"], GC_VI_NonlinC_GRU_p_table_siglags_roc_data.iloc[10]["TPR"], "x", c="grey")
plt.plot(
    GC_VI_NonlinC_LSTM_p_table_siglags_roc_data["FPR"],
    GC_VI_NonlinC_LSTM_p_table_siglags_roc_data["TPR"],
    lw=lw,
    c="green",
    label="NonlinC_LSTM"
)
plt.plot(GC_VI_NonlinC_LSTM_p_table_siglags_roc_data.iloc[1]["FPR"], GC_VI_NonlinC_LSTM_p_table_siglags_roc_data.iloc[1]["TPR"], "x", c="black")
plt.plot(GC_VI_NonlinC_LSTM_p_table_siglags_roc_data.iloc[5]["FPR"], GC_VI_NonlinC_LSTM_p_table_siglags_roc_data.iloc[5]["TPR"], "x", c="blue")
plt.plot(GC_VI_NonlinC_LSTM_p_table_siglags_roc_data.iloc[10]["FPR"], GC_VI_NonlinC_LSTM_p_table_siglags_roc_data.iloc[10]["TPR"], "x", c="grey")
plt.legend()
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("Correlation sensitivity in noise structure")
plt.savefig(
    r"C:\Users\Fabia\OneDrive\Desktop\Studium\Master\Semester III\Seminar Financial Data Analytics\Submission\GC_VI_combined.pdf",
    bbox_inches="tight")
plt.legend(loc="lower right")
plt.show()

#############################################
######          GC - Inference         ######
###### Error Term - Sensitivity VII    ######
#############################################

sample_size = 1000
tensor_count = 30
lags = [2,3,4,5]
lag_amount = len(lags)

#Determine E
sigma = [0.5, 0.7] #standard deviation of first and second time series
corr = 0.2         #correlation

covs = [[sigma[0]**2          , sigma[0]*sigma[1]*corr],
        [sigma[0]*sigma[1]*corr,          sigma[1]**2]]

np.random.seed(42)
noise = np.random.multivariate_normal(mean = [0.1,0.1], cov = covs, size=(sample_size + 2 * lag_amount,))

#Generate Data
DataGeneration(case = 1,sample_size = sample_size, lags = lags, k = 2, tensor_count = tensor_count, noise = noise)

# import results from R ###
import_data_R("VII", 5)

# import results from Python ###
import_data_Py("VII", 5)

# create ROC tables
ROC_composition("VII", alpha_seq_joint)

# export Data
export_data()

createROCcurve(alpha_seq_joint,GC_VII_statsmodels_table_p, "test")

#plt.figure(figsize = (20,4))
plt.scatter(y = GC_IV_statsmodels_table_p.iloc[3,:], x = [i for i in range(tensor_count)], marker = ".", label = "statsmodels", c = "black")
plt.scatter(y = GC_IV_NonlinC_NN_p_table.iloc[3,:], x = [i for i in range(tensor_count)], marker = "v", label = "NonlinC_NN", c = "black")
plt.scatter(y = GC_IV_NonlinC_GRU_p_table.iloc[3,:], x = [i for i in range(tensor_count)], marker = "x", label = "NonlinC_GRU", c = "black")
plt.scatter(y = GC_IV_NonlinC_LSTM_p_table.iloc[3,:], x = [i for i in range(tensor_count)], marker = "s", label = "NonlinC_LSTM", c = "black")
plt.scatter(y = GC_IV_GrangerFunc_p_table.iloc[3,:], x = [i for i in range(tensor_count)], marker = "x", label = "GrangerFunc", c = "black")
plt.scatter(y = GC_IV_NlinTS_NN_p_table.iloc[3,:], x = [i for i in range(tensor_count)], marker = "D", label = "NlinTS", c = "black")
plt.plot([0,30], [0.05,0.05], c = "red", linestyle = "--", label = "5%-significance level")
plt.ylim([0,0.075])
plt.legend(loc = "lower right")
plt.xlabel("Tensorindex")
plt.ylabel("P-values")
plt.savefig(rf"C:/Users/Fabia/OneDrive/Desktop/Studium/Master/Semester III/Seminar Financial Data Analytics/Submission/pvaluesovertensors.pdf", bbox_inches="tight")

#plt.scatter(y = GC_IV_NlinTS_NN_p_table.iloc[4,:], x = [i for i in range(tensor_count)])

plt.plot(GC_IV_GrangerFunc_F_table.iloc[:,0])

for tensor in range(tensor_count):
    plt.figure(figsize = (20,4))
    plt.plot(GC_IV_NonlinC_NN_F_table.iloc[:7,tensor])
    plt.plot(GC_III_NonlinC_NN_F_table.iloc[:7,tensor])


#############################################
######          GC - Inference         ######
###### Error Term - Sensitivity VII    ######
#############################################

sample_size = 1000
tensor_count = 30
lags = [2,3,4,5]
lag_amount = len(lags)

#Determine E
sigma = [0.5, 0.7] #standard deviation of first and second time series
corr = 0.2         #correlation

covs = [[sigma[0]**2          , sigma[0]*sigma[1]*corr],
        [sigma[0]*sigma[1]*corr,          sigma[1]**2]]

np.random.seed(42)
noise = np.random.multivariate_normal(mean = [0.1,0.1], cov = covs, size=(sample_size + 2 * lag_amount,))

#Generate Data
DataGeneration(case = 1,sample_size = sample_size, lags = lags, k = 2, tensor_count = tensor_count, noise = noise)

# import results from R ###
import_data_R("VII", 5)

# import results from Python ###
import_data_Py("VII", 5)

# create ROC tables
ROC_composition("VII", alpha_seq_joint)

# export Data
export_data()

createROCcurve(alpha_seq_joint,GC_VII_statsmodels_table_p, "test")

#plt.figure(figsize = (20,4))
plt.scatter(y = GC_IV_statsmodels_table_p.iloc[3,:], x = [i for i in range(tensor_count)], marker = ".", label = "statsmodels", c = "black")
plt.scatter(y = GC_IV_NonlinC_NN_p_table.iloc[3,:], x = [i for i in range(tensor_count)], marker = "v", label = "NonlinC_NN", c = "black")
plt.scatter(y = GC_IV_NonlinC_GRU_p_table.iloc[3,:], x = [i for i in range(tensor_count)], marker = "x", label = "NonlinC_GRU", c = "black")
plt.scatter(y = GC_IV_NonlinC_LSTM_p_table.iloc[3,:], x = [i for i in range(tensor_count)], marker = "s", label = "NonlinC_LSTM", c = "black")
plt.scatter(y = GC_IV_GrangerFunc_p_table.iloc[3,:], x = [i for i in range(tensor_count)], marker = "x", label = "GrangerFunc", c = "black")
plt.scatter(y = GC_IV_NlinTS_NN_p_table.iloc[3,:], x = [i for i in range(tensor_count)], marker = "D", label = "NlinTS", c = "black")
plt.plot([0,30], [0.05,0.05], c = "red", linestyle = "--", label = "5%-significance level")
plt.ylim([0,0.075])
plt.legend(loc = "lower right")
plt.xlabel("Tensorindex")
plt.ylabel("P-values")
plt.savefig(rf"C:/Users/Fabia/OneDrive/Desktop/Studium/Master/Semester III/Seminar Financial Data Analytics/Submission/pvaluesovertensors.pdf", bbox_inches="tight")

#plt.scatter(y = GC_IV_NlinTS_NN_p_table.iloc[4,:], x = [i for i in range(tensor_count)])

plt.plot(GC_IV_GrangerFunc_F_table.iloc[:,0])

for tensor in range(tensor_count):
    plt.figure(figsize = (20,4))
    plt.plot(GC_IV_NonlinC_NN_F_table.iloc[:7,tensor])
    plt.plot(GC_III_NonlinC_NN_F_table.iloc[:7,tensor])



###############################################
######          GC - Inference           ######
######   Error Term - Sensitivity VIII   ######
###############################################

lags = [3,3,4,4]

# import results from R ###
import_data_R("VIII", 4)
import_data_R("VIII_stat", 4)

# import results from Python ###
import_data_Py("VIII", 4, both = True)
import_data_Py("VIII_stat", 4, both= False)


#create ROC data
ROC_composition_pyonly("VIII",alpha_seq_joint)
ROC_composition("VIII_stat",alpha_seq_joint, both = False)

createROCcurve(alpha_seq_joint, GC_VIII_lmtest_p_table_joint_siglags, "ARX_lmtest")
createROCcurve(alpha_seq_joint, GC_VIII_statsmodels_table_p_joint_siglags, "ARX_statsmodels")
createROCcurve(alpha_seq_joint, GC_VIII_GrangerFunc_p_table_joint_siglags, "ARX_GrangerFunc")
createROCcurve(alpha_seq_joint, GC_VIII_NlinTS_p_table_joint_siglags, "ARX_NlinTS_GC")
createROCcurve(alpha_seq_joint, GC_VIII_NlinTS_NN_p_table_joint_siglags, "ARX_nlin_causality.test")
createROCcurve(alpha_seq_joint, GC_VIII_NonlinC_NN_p_table_siglags, "ARX_NonlinC_NN")
createROCcurve(alpha_seq_joint, GC_VIII_NonlinC_GRU_p_table_siglags, "ARX_NonlinC_GRU")
createROCcurve(alpha_seq_joint, GC_VIII_NonlinC_LSTM_p_table_siglags, "ARX_NonlinC_LSTM")

createROCcurve(alpha_seq_joint, GC_VIII_stat_statsmodels_table_p_siglags, "ARX_stat_statsmodels")
createROCcurve(alpha_seq_joint, GC_VIII_stat_NonlinC_NN_p_table_siglags, "ARX_stat_NonlinC_NN")
createROCcurve(alpha_seq_joint, GC_VIII_stat_NonlinC_GRU_p_table_siglags, "ARX_stat_NonlinC_GRU")
createROCcurve(alpha_seq_joint, GC_VIII_stat_NonlinC_LSTM_p_table_siglags, "ARX_stat_NonlinC_LSTM")
createROCcurve(alpha_seq_joint, GC_VIII_stat_NlinTS_NN_p_table_siglags, "ARX_stat_nlin_causality.test")
createROCcurve(alpha_seq_joint, GC_VIII_stat_GrangerFunc_p_table_siglags, "ARX_stat_GrangerFunc")

createROCcurve(alpha_seq_joint, GC_VIII_stat_lmtest_p_table_siglags, "GC_VIII_stat_lmtest")
createROCcurve(alpha_seq_joint, GC_VIII_stat_statsmodels_table_p_siglags, "GC_VIII_stat_statsmodels")

createROCcurve(alpha_seq_joint, GC_VIII_stat_NlinTS_p_table_siglags, "GC_VIII_stat_NlinTS_GC")




conf_mat(0.05,GC_VIII_statsmodels_table_p_joint_siglags)

inner_loopsize = GC_VIII_NonlinC_LSTM_p_table_siglags.shape[0] / (GC_VIII_NonlinC_LSTM_p_table_siglags.shape[0] / max(lags))
outer_loopsize = GC_VIII_NonlinC_LSTM_p_table_siglags.shape[1] * (GC_VIII_NonlinC_LSTM_p_table_siglags.shape[0] / max(lags))
true_result_stacked = []

for outer in range(int(outer_loopsize)):
    for inner in range(1, int(inner_loopsize) + 1):
        if inner == lags[0] or inner == lags[1] or inner == lags[2] or inner == lags[3]:
            value = int(1)
            true_result_stacked.append(value)
        else:
            value = int(0)
            true_result_stacked.append(value)

y_coi = pd.DataFrame(true_result_stacked)

# prepare the vector with the estimated data (p-values)
sub_yhat = yhat.melt()
yhat_pred = sub_yhat.iloc[:, -1]
yhat_is_coi = np.empty((sub_yhat.shape[0], 1))

for i in range(sub_yhat.shape[0]):
    if yhat_pred.loc[i] <= threshold:
        yhat_sub = 1
        yhat_is_coi[i] = yhat_sub
    else:
        yhat_sub = 0
        yhat_is_coi[i] = yhat_sub

# calculate True Positive Rate (TPR)
col_to_summup_TPR = yhat_is_coi * y_coi
numerator_TPR = col_to_summup_TPR.sum()
result_TPR = numerator_TPR / y_coi.sum()
TruePosRate = round(result_TPR[0], 3)

# calculate False Positive Rate (FPR)
col_to_summup_FPR = (1 - y_coi) * yhat_is_coi
numerator_FPR = col_to_summup_FPR.sum()
denominator_FPR = 1 - y_coi
result_FPR = numerator_FPR / denominator_FPR.sum()
FalsePosRate = round(result_FPR[0], 3)

plt.figure()
lw = 2
plt.plot(
    GC_VIII_stat_statsmodels_table_p_siglags_roc_data["FPR"],
    GC_VIII_stat_statsmodels_table_p_siglags_roc_data["TPR"],
    lw=lw,
    c="blue",
    label="statsmodels"
)
plt.plot(GC_VIII_stat_statsmodels_table_p_siglags_roc_data.iloc[1]["FPR"], GC_VIII_stat_statsmodels_table_p_siglags_roc_data.iloc[1]["TPR"], "x", c="black")
plt.plot(GC_VIII_stat_statsmodels_table_p_siglags_roc_data.iloc[5]["FPR"], GC_VIII_stat_statsmodels_table_p_siglags_roc_data.iloc[5]["TPR"], "x", c="blue")
plt.plot(GC_VIII_stat_statsmodels_table_p_siglags_roc_data.iloc[10]["FPR"], GC_VIII_stat_statsmodels_table_p_siglags_roc_data.iloc[10]["TPR"], "x", c="grey")
plt.plot(
    GC_VIII_statsmodels_table_p_siglags_roc_data["FPR"],
    GC_VIII_statsmodels_table_p_siglags_roc_data["TPR"],
    lw=lw,
    c="red",
    label="statsmodels"
)
plt.plot(GC_VIII_statsmodels_table_p_siglags_roc_data.iloc[1]["FPR"], GC_VIII_statsmodels_table_p_siglags_roc_data.iloc[1]["TPR"], "x", c="black")
plt.plot(GC_VIII_statsmodels_table_p_siglags_roc_data.iloc[5]["FPR"], GC_VIII_statsmodels_table_p_siglags_roc_data.iloc[5]["TPR"], "x", c="blue")
plt.plot(GC_VIII_statsmodels_table_p_siglags_roc_data.iloc[10]["FPR"], GC_VIII_statsmodels_table_p_siglags_roc_data.iloc[10]["TPR"], "x", c="grey")
plt.legend()
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("Correlation sensitivity in noise structure")
plt.savefig(
    r"C:\Users\Fabia\OneDrive\Desktop\Studium\Master\Semester III\Seminar Financial Data Analytics\Submission\GC_VI_combined.pdf",
    bbox_inches="tight")
plt.legend(loc="lower right")
plt.show()

conf_mat(0.01,GC_VIII_stat_statsmodels_table_p_siglags)


# import results from R ###
import_data_R("VIII", 4)
import_data_R("VIII_stat", 4)

# import results from Python ###
import_data_Py("VIII", 4)
import_data_Py("VIII_stat", 4, both= False)



#############################################
######          GC - Inference         ######
###### Error Term - Sensitivity V (COS)######
#############################################

sample_size = 10000
tensor_count = 30
lags = [2,3,4,5]
lag_amount = len(lags)

#Determine E
sigma = [0.5, 0.5] #standard deviation of first and second time series
corr = 0         #correlation

covs = [[sigma[0]**2          , sigma[0]*sigma[1]*corr],
        [sigma[0]*sigma[1]*corr,          sigma[1]**2]]

np.random.seed(42)
noise = np.random.multivariate_normal(mean = [0.1,0.1], cov = covs, size=(sample_size + 2 * lag_amount,))

#Generate Data
DataGeneration(case = 1,sample_size = sample_size, lags = lags, k = 2, tensor_count = tensor_count, noise = noise)

# import results from R ###
import_data_R("V", 5)

# import results from Python ###
import_data_Py("V", 5, both = True)

# create ROC tables
ROC_composition("V", alpha_seq_joint, both = True)



#createROCcurve(alpha_seq_joint, GC_V_lmtest_p_table_joint_siglags, "1_GC_V_lmtest")
createROCcurve(alpha_seq_joint, GC_V_statsmodels_table_p_joint_siglags, "grangercausalitytests")
createROCcurve(alpha_seq_joint, GC_V_GrangerFunc_p_table_joint_siglags, "GrangerFunc")
#createROCcurve(alpha_seq_joint, GC_V_NlinTS_p_table_joint_siglags, "1_GC_V_NlinTS_GC")
createROCcurve(alpha_seq_joint, GC_V_NlinTS_NN_p_table_joint_siglags, "nlin_causality.test")
createROCcurve(alpha_seq_joint, GC_V_NonlinC_NN_p_table_joint_siglags, "nonlincausality_NN")
createROCcurve(alpha_seq_joint, GC_V_NonlinC_GRU_p_table_joint_siglags, "nonlincausality_GRU")
createROCcurve(alpha_seq_joint, GC_V_NonlinC_LSTM_p_table_joint_siglags, "nonlincausality_LSTM")



import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import statistics
import keras
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, TimeDistributed, Flatten
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from statsmodels.tsa.stattools import grangercausalitytests

######################################
######  Data Generating Process ######
######     Function Type I      ######
######################################

"""
The Data Generating Process (DGP) Function creates data with a Vector Autoregressive (VAR) Process up to a dependency of two lags

Parameters
----------

- sample_size (integer) : Determines the amount of simulated data points / observations for each individually artificially created time series
- lags (vector)         : Reflects the wished dependency on the given lags, e. g. lags = [1,4]
- k (integer)           : Number of functions within the VAR Process
- tensor_count (integer): Sets the amount of simulated VAR Processes with k functions. E. g. k = 2 and tensor_count = 100, means that 100 VAR Processes with 2 sich gegenseitig beeinflussenden 
- noise (matrix)        : Created error term matrix of size sample_size x k of the user

Example: sample_size = 1000
         lags = [1,3]
         k = 2
tensor_count = 10

Meaning: 10 VAR-Processes with 2 functions and 1000 observations each are created which depend on the first and the third lag of the own observations and the other timeseries.
The relationship (Coefficients) vary over the cases and are described within the paper.
"""


# %%
def DataGeneration(case, sample_size,  # integer
                   lags,  # vector e. g. [1,2,3,4]
                   k,
                   tensor_count,  # integer
                   noise):  # error term

    lag_amount = len(lags)

    # create multiple empty coefficient tensors of the amount of tensor_count
    for p in range(lag_amount):
        for tensor in range(tensor_count):
            globals()["lag_" + str(lags[p]) + "_tensor_" + str(tensor + 1)] = []

    # create coefficent matrices and append to the tensor
    for p in range(lag_amount):
        for tensor in range(tensor_count):
            for i in range(sample_size + 2 * lag_amount):
                # create an empty matrix with dimension k x k
                matrix = np.zeros((k, k))

                # append matrix for each simulation step to the tensor
                globals()["lag_" + str(lags[p]) + "_tensor_" + str(tensor + 1)].append(matrix)
                i += 1

    # Granger Case 1 (GC1):

    if case == 1:
        # every tensor for every lag gets a seed equals tensor_count * lag_amount different seeds
        seed_switch_tensor = cycle(range((tensor_count) * lag_amount))
        seed_switch_matrix = cycle(range((sample_size) * lag_amount))

        for p in range(lag_amount):
            for tensor in range(tensor_count):
                np.random.seed(next(
                    seed_switch_tensor))  # every tensor for every lag gets a seed equals tensor_count * lag_amount different seeds
                nonlinearity_setting_XonY = np.cos(
                    np.linspace(0, 20, sample_size + 2 * lag_amount)) * np.random.uniform(0, 0.5,
                                                                                          1)  # seeded by tensor switch
                nonlinearity_setting_YonX = np.cos(
                    np.linspace(0, 15, sample_size + 2 * lag_amount)) * np.random.uniform(0, 0.5,
                                                                                          1)  # seeded by tensor switch

                for i in range(p, sample_size + 2 * lag_amount):
                    np.random.seed(next(seed_switch_matrix))
                    coef_matrix = np.zeros((k, k))
                    coef_matrix[0][0] = np.random.uniform(-0.2, 0.2, 1)  # seeded by matrix switch
                    coef_matrix[0][1] = nonlinearity_setting_XonY[i]
                    coef_matrix[1][0] = nonlinearity_setting_YonX[i]
                    coef_matrix[1][1] = np.random.uniform(-0.2, 0.2, 1)  # seeded by matrix switch
                    coef_subtensor = coef_matrix
                    globals()["lag_" + str(lags[p]) + "_tensor_" + str(tensor + 1)][i] = coef_subtensor

    # create an empty array for the simulated data dat
    for tensor in range(tensor_count):
        globals()["dat_tensor_" + str(tensor + 1)] = np.zeros((sample_size + 2 * lag_amount, k))

    # fill the dataframe with the VAR-Model via matrix multiplication

    if lag_amount == 4:
        # for p in range(lag_amount):
        for tensor in range(tensor_count):
            for i in range(p, sample_size + 2 * lag_amount):
                globals()["dat_tensor_" + str(tensor + 1)][i, :] = np.matmul(
                    globals()["lag_" + str(lags[0]) + "_tensor_" + str(tensor + 1)][i],
                    globals()["dat_tensor_" + str(tensor + 1)][i - lags[0], :]) + np.matmul(
                    globals()["lag_" + str(lags[1]) + "_tensor_" + str(tensor + 1)][i],
                    globals()["dat_tensor_" + str(tensor + 1)][i - lags[1], :]) + np.matmul(
                    globals()["lag_" + str(lags[2]) + "_tensor_" + str(tensor + 1)][i],
                    globals()["dat_tensor_" + str(tensor + 1)][i - lags[2], :]) + np.matmul(
                    globals()["lag_" + str(lags[3]) + "_tensor_" + str(tensor + 1)][i],
                    globals()["dat_tensor_" + str(tensor + 1)][i - lags[3], :]) + noise[i]

    if lag_amount == 3:
        # for p in range(lag_amount):
        for tensor in range(tensor_count):
            for i in range(p, sample_size + 2 * lag_amount):
                globals()["dat_tensor_" + str(tensor + 1)][i, :] = np.matmul(
                    globals()["lag_" + str(lags[0]) + "_tensor_" + str(tensor + 1)][i],
                    globals()["dat_tensor_" + str(tensor + 1)][i - lags[0], :]) + np.matmul(
                    globals()["lag_" + str(lags[1]) + "_tensor_" + str(tensor + 1)][i],
                    globals()["dat_tensor_" + str(tensor + 1)][i - lags[1], :]) + np.matmul(
                    globals()["lag_" + str(lags[2]) + "_tensor_" + str(tensor + 1)][i],
                    globals()["dat_tensor_" + str(tensor + 1)][i - lags[2], :]) + noise[i]

    if lag_amount == 2:
        # for p in range(lag_amount):
        for tensor in range(tensor_count):
            for i in range(p, sample_size + 2 * lag_amount):
                globals()["dat_tensor_" + str(tensor + 1)][i, :] = np.matmul(
                    globals()["lag_" + str(lags[0]) + "_tensor_" + str(tensor + 1)][i],
                    globals()["dat_tensor_" + str(tensor + 1)][i - lags[0], :]) + np.matmul(
                    globals()["lag_" + str(lags[1]) + "_tensor_" + str(tensor + 1)][i],
                    globals()["dat_tensor_" + str(tensor + 1)][i - lags[1], :]) + noise[i]

    if lag_amount == 1:
        # for p in range(lag_amount):
        for tensor in range(tensor_count):
            for i in range(p, sample_size + 2 * lag_amount):
                globals()["dat_tensor_" + str(tensor + 1)][i, :] = np.matmul(
                    globals()["lag_" + str(lags[0]) + "_tensor_" + str(tensor + 1)][i],
                    globals()["dat_tensor_" + str(tensor + 1)][i - lags[0], :]) + noise[i]

    else:
        print("Amount of lags has to be at least 1")


sample_size = 10000
tensor_count = 20
lags = [2, 3, 4, 5]
lag_amount = len(lags)

# Determine E
sigma = [0.5, 0.5]  # standard deviation of first and second time series
corr = 0  # correlation

covs = [[sigma[0] ** 2, sigma[0] * sigma[1] * corr],
        [sigma[0] * sigma[1] * corr, sigma[1] ** 2]]

np.random.seed(42)
noise = np.random.multivariate_normal(mean=[0, 0], cov=covs, size=(sample_size + 2 * lag_amount,))

# Generate Data
DataGeneration(case=1, sample_size=sample_size, lags=lags, k=2, tensor_count=tensor_count, noise=noise)


lag_2_coefficients = []
lag_3_coefficients = []
lag_4_coefficients = []
lag_5_coefficients = []


for p in lags:
    for tensor in range(tensor_count):
        globals()["lag_" + str(p) + "_coefficients"].append(globals()["lag_" + str(p) + "_tensor_" + str(tensor+1)][10][0][1])

lag_coefficients = pd.DataFrame(data = zip(lag_2_coefficients, lag_3_coefficients, lag_4_coefficients, lag_5_coefficients))

lag_2 = [2] * 20
lag_3 = [3] * 20
lag_4 = [4] * 20
lag_5 = [5] * 20


lag_mat = pd.DataFrame(data = zip(lag_2, lag_3, lag_4, lag_5))

fig = plt.figure(figsize = (20,4))
ax = plt.axes(projection = "3d")
ax.scatter3D(lag_mat, lag_coefficients,GC_V_NonlinC_LSTM_p_table_XY.iloc[1:5,:].T, c = "orange", label = "NonlinC_LSTM")

fig = plt.figure(figsize = (20,4))
ax = plt.axes(projection = "3d")
ax.scatter3D(lag_mat, lag_coefficients,GC_V_NonlinC_NN_p_table.iloc[1:5,:].T, c = "red", label = "NonlinC_NN")
ax.scatter3D(lag_mat, lag_coefficients,GC_V_NonlinC_GRU_p_table.iloc[1:5,:].T, c = "yellow", label = "NonlinC_GRU")
ax.scatter3D(lag_mat, lag_coefficients,GC_V_NonlinC_LSTM_p_table.iloc[1:5,:].T, c = "orange", label = "NonlinC_LSTM")
ax.scatter3D(lag_mat, lag_coefficients,GC_V_statsmodels_table_p.iloc[1:5,:].T, c = "black", label = "OLS Regression")
ax.set_ylabel("Coefficient")
ax.set_xlabel("Lagparameter")
ax.set_zlabel("P Value")
ax.set_zlim(0,0.05)
plt.legend(loc = "upper right")
plt.savefig(
    r"C:\Users\Fabia\OneDrive\Desktop\Studium\Master\Semester III\Seminar Financial Data Analytics\Submission\coef3d_NonlinC_V.pdf",
    bbox_inches="tight")


#############################################
######          GC - Inference         ######
###### Error Term - Sensitivity VI (Decline)#
#############################################

sample_size = 1000
tensor_count = 30
lags = [2,3,4,5]
lag_amount = len(lags)

#Determine E
sigma = [0.5, 0.7] #standard deviation of first and second time series
corr = 0.2         #correlation

covs = [[sigma[0]**2          , sigma[0]*sigma[1]*corr],
        [sigma[0]*sigma[1]*corr,          sigma[1]**2]]

np.random.seed(42)
noise = np.random.multivariate_normal(mean = [0.1,0.1], cov = covs, size=(sample_size + 2 * lag_amount,))

#Generate Data
DataGeneration(case = 1,sample_size = sample_size, lags = lags, k = 2, tensor_count = tensor_count, noise = noise)

# import results from R ###
import_data_R("VI", 5)

# import results from Python ###
import_data_Py("VI", 5)

# create ROC tables
ROC_composition("VI", alpha_seq_joint)

# export Data
export_data()

plt.figure()
lw = 2
plt.plot(
    GC_VI_GrangerFunc_p_table_siglags_roc_data["FPR"],
    GC_VI_GrangerFunc_p_table_siglags_roc_data["TPR"],
    lw=lw,
    c="black",
    label="GrangerFunc"
)
plt.plot(
    GC_VI_GrangerFunc_p_table_siglags_roc_data["FPR"],
    GC_VI_GrangerFunc_p_table_siglags_roc_data["TPR"],
    lw=lw,
    c="black",
    linestyle="-."
)
plt.plot(
    GC_VI_NlinTS_NN_p_table_siglags_roc_data["FPR"],
    GC_VI_NlinTS_NN_p_table_siglags_roc_data["TPR"],
    lw=lw,
    c="gray",
    label="NlinTS"
)
plt.plot(
    GC_VI_NlinTS_NN_p_table_siglags_roc_data["FPR"],
    GC_VI_NlinTS_NN_p_table_siglags_roc_data["TPR"],
    lw=lw,
    c="gray",
    linestyle = "-."
)
plt.plot(
    GC_VI_statsmodels_table_p_siglags_roc_data["FPR"],
    GC_VI_statsmodels_table_p_siglags_roc_data["TPR"],
    lw=lw,
    c="blue",
    label="statsmodels"
)
plt.plot(
    GC_VI_NonlinC_NN_p_table_siglags_roc_data["FPR"],
    GC_VI_NonlinC_NN_p_table_siglags_roc_data["TPR"],
    lw=lw,
    c="red",
    label="NonlinC_NN"
)
plt.plot(
    GC_VI_NonlinC_GRU_p_table_siglags_roc_data["FPR"],
    GC_VI_NonlinC_GRU_p_table_siglags_roc_data["TPR"],
    lw=lw,
    c="orange",
    label="NonlinC_GRU"
)
plt.plot(
    GC_VI_NonlinC_LSTM_p_table_siglags_roc_data["FPR"],
    GC_VI_NonlinC_LSTM_p_table_siglags_roc_data["TPR"],
    lw=lw,
    c="green",
    label="NonlinC_LSTM"
)
plt.legend()
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
#plt.title("Correlation sensitivity in noise structure")
plt.savefig(
    r"C:\Users\Fabia\OneDrive\Desktop\Studium\Master\Semester III\Seminar Financial Data Analytics\Submission\Nonlin_decreasing.pdf",
    bbox_inches="tight")
plt.legend(loc="lower right")
plt.show()



#############################################
######          GC - Inference         ######
###### Error Term - Sensitivity IX (Decline)#
#############################################

sample_size = 1000
tensor_count = 30
lags = [2,3,4,5]
lag_amount = len(lags)

#Determine E
sigma = [0.5, 0.5] #standard deviation of first and second time series
corr = 0         #correlation

covs = [[sigma[0]**2          , sigma[0]*sigma[1]*corr],
        [sigma[0]*sigma[1]*corr,          sigma[1]**2]]

np.random.seed(42)
noise = np.random.multivariate_normal(mean = [0.1,0.1], cov = covs, size=(sample_size + 2 * lag_amount,))

#Generate Data
DataGeneration(case = 1,sample_size = sample_size, lags = lags, k = 2, tensor_count = tensor_count, noise = noise)

# import results from R ###
import_data_R("IX", 5)

# import results from Python ###
import_data_Py("IX", 5)

# create ROC tables
ROC_composition("IX", alpha_seq_joint)

# export Data
export_data()

plt.figure()
lw = 2
plt.plot(
    GC_IX_GrangerFunc_p_table_siglags_roc_data["FPR"],
    GC_IX_GrangerFunc_p_table_siglags_roc_data["TPR"],
    lw=lw,
    c="black",
    linestyle="-."
)
plt.plot(
    GC_IX_NlinTS_NN_p_table_siglags_roc_data["FPR"],
    GC_IX_NlinTS_NN_p_table_siglags_roc_data["TPR"],
    lw=lw,
    c="gray",
    label="NlinTS"
)
plt.plot(
    GC_IX_NlinTS_GRU_p_table_siglags_roc_data["FPR"],
    GC_IX_NlinTS_GRU_p_table_siglags_roc_data["TPR"],
    lw=lw,
    c="gray",
    linestyle = "-."
)
plt.plot(
    GC_IX_NlinTS_LSTM_p_table_siglags_roc_data["FPR"],
    GC_IX_NlinTS_LSTM_p_table_siglags_roc_data["TPR"],
    lw=lw,
    c="gray",
    linestyle = "-."
)
plt.plot(
    GC_IX_statsmodels_table_p_siglags_roc_data["FPR"],
    GC_IX_statsmodels_table_p_siglags_roc_data["TPR"],
    lw=lw,
    c="blue",
    label="statsmodels"
)
plt.plot(
    GC_IX_NonlinC_NN_p_table_siglags_roc_data["FPR"],
    GC_IX_NonlinC_NN_p_table_siglags_roc_data["TPR"],
    lw=lw,
    c="red",
    label="NonlinC_NN"
)
plt.legend()
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
#plt.title("Correlation sensitivity in noise structure")
plt.savefig(
    r"C:\Users\Fabia\OneDrive\Desktop\Studium\Master\Semester III\Seminar Financial Data Analytics\Submission\Nonlin_increasing.pdf",
    bbox_inches="tight")
plt.legend(loc="lower right")
plt.show()
