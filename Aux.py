# This class has helper functions to evaluate the function
import pandas
from sklearn.metrics import fbeta_score, precision_score, recall_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from HARF import HighAgreementRandomForest


def shuffle_samples(x, y):
    """
    This function creates bootstrapped samples out of the given sample.

    :param x: X values from the initial dataset
    :param y: y values from the initial dataset
    :return: Returns the bootstrapped dataset
    """
    n_samples = x.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=False)
    return x.iloc[idxs], y.iloc[idxs]


def add_noise_index(y_test, noise_percent):
    """
    This function does the same as add noise and additionally return the indexes of the changed values
    :param y_test: Vector with the test target values, which will get added noise
    :param noise_percent: How much noise should be added to the test set in percentage
    :return: Returns the y_noise series with added noise and the indexes at which noise was added
    """
    noise_percent = noise_percent
    y_noise = y_test.copy()
    n_samples = y_noise.shape[0]
    idxs = np.random.choice(n_samples, round(n_samples * noise_percent), replace=False)
    # print("Instances of y changed at the following indexes:\n", y_noise.iloc[idxs])
    y_noise.iloc[idxs] = y_noise.iloc[idxs].replace([True, False], [False, True], inplace=False)
    return y_noise, idxs


def prediction_ntb(predictions):
    """
    This function converts the integer values from the given pd.Series to boolean values
    :return: Returns a pd.Series with all numeric values replaced to boolean values
    """
    predictions_bol = predictions.replace(['1.0', '0.0'], [True, False], inplace=False)

    return predictions_bol


def add_noise(y_noise, noise_percent):
    """
    This function randomly flips the target value for a given percentage of the dataframe by changing
    the target values from True to False and vice-versa.
    :param y_noise: A pandas DataFrame that will be injected with noise
    :param noise_percent: The percentage amount of the dataset that will be injected with noise
    :return: Returns the pandas DataFrame with the added noise
    """
    noise_percent = noise_percent
    y_noise = y_noise.copy()
    n_samples = y_noise.shape[0]
    idxs = np.random.choice(n_samples, round(n_samples * noise_percent), replace=False)
    # print("Instances of y changed at the following indexes:\n", y_noise.iloc[idxs])
    y_noise.iloc[idxs] = y_noise.iloc[idxs].replace([True, False], [False, True], inplace=False)
    return y_noise


def get_noise_predicted(predicted_bol, y_noise, y_test):
    """
    This function creates two vectors as pd.Series. One is the vector for the
    predicted noise, which has a True value when the instance is classified
    as noise and a False value when the instance is classified as not noise.

    This function only transforms the predicted values from the High Agreement
    Random Forest and does not predict anything itself.
    :param predicted_bol: pd.Series with the
    :param y_noise:
    :param y_test:
    :return:
    """
    predicted_noise = pd.Series(index=predicted_bol.index, dtype=bool)
    for x in predicted_bol.index:
        if predicted_bol.loc[x] == y_noise.loc[x]:
            predicted_noise.loc[x] = False

        elif predicted_bol.loc[x] == "NN":
            predicted_noise.loc[x] = False
        else:
            predicted_noise.loc[x] = True
    noise = pd.Series(index=predicted_bol.index, dtype=bool)
    for x in y_test.index:
        if y_test.loc[x] == y_noise.loc[x]:
            noise.loc[x] = False
        else:
            noise.loc[x] = True

    return predicted_noise, noise


def find_noise_harf(x_train, y_train, x_test, y_test, agreement_percentage=0.8, n_trees=500,
                    max_depth=9, max_features='sqrt'):
    """
    This function finds noise on a given target column based on past instances of a training dataset.
    Represents Method M1A in the thesis
    :param max_features:
    :param x_train: pandas DataFrame (DF) containing the correct feature array to train the model
    :param y_train: pandas DF containing the corresponding target values of x_train to train the model
    :param x_test: pandas DF containing the feature array of the dataset of the target columns y_test
    :param y_test: pandas DF containing the list of target columns to test for noise
    :param agreement_percentage: The agreement percentage of the Random Forest
    :param n_trees: The number of trees that the Random Forest will have
    :param max_depth: The maximum depth of each individual tree
    :return:
    Returns the list of instances that the method detected to be noise in predicted_noise
    """
    x_train = x_train.copy()
    x_test = x_test.copy()
    y_train = y_train.copy()
    y_test = y_test.copy()

    assert y_train.columns == y_test.columns
    assert x_train.columns == x_test.columns
    list_predicted_noise = list()

    for j in list(range(len(y_train.columns))):
        y = y_train.iloc[:, j]
        y_test1 = y_test.iloc[:, j]
        # HARF starts
        model = HighAgreementRandomForest(n_trees=n_trees, max_depth=max_depth,
                                          agreement_percentage=agreement_percentage,
                                          max_features=max_features)
        model.fit(x_train, y)
        predictions = pandas.Series(dtype=object)
        predictions = model.predict(x_test)
        predictions_bool = prediction_ntb(predictions)
        predicted_noise = pd.Series(index=predictions_bool.index, dtype=bool)
        # HARF ends
        print("In: {}, model HARF5{} found the following instances to be noise"
              .format(y.name, round(agreement_percentage * 100)))
        print("Start of list:")
        for x in predictions_bool.index:
            if predictions_bool.loc[x] == y_test1.loc[x]:
                predicted_noise.loc[x] = False

            elif predictions_bool.loc[x] == "NN":
                predicted_noise.loc[x] = False
            else:
                predicted_noise.loc[x] = True
                print("Instance with index: {} is tagged as noise".format(x))
        print("- End of list.")

        list_predicted_noise.append(predicted_noise.tolist())
    df_predicted_noise = pd.DataFrame(list_predicted_noise, columns=y_test.index)
    df_predicted_noise = df_predicted_noise.transpose()
    df_predicted_noise.columns = y_test.columns

    return df_predicted_noise


def find_noise_harf_no_train_data(x_data, y_data, n_splits=10, agreement_percentage=0.8,
                                  n_trees=500, max_depth=9, max_features='sqrt'):
    """
    This function finds mislabeled instances in datasets

    :param x_data: Feature vector set
    :param y_data: Target column list
    :param n_splits:
    :param agreement_percentage: The agreement percentage of the Random Forest
    :param max_depth: The maximum depth of each individual tree
    :param n_trees: The number of trees that the Random Forest will have
    :param max_features:

    :return:
    Returns a list with
    """
    assert n_splits > 1
    list_predicted_noise = list()
    x1 = x_data.copy()
    for j in list(range(len(y_data.columns))):

        y1 = y_data.iloc[:, j].copy()
        x, y = shuffle_samples(x1, y1)
        index = list(x.index)
        d = len(y.unique())
        if d == 2:
            y.replace(list(y.unique()), list([True, False]), inplace=True)
        else:
            y.replace(list(y.unique()), list([True]), inplace=True)
        index_list_n = list(split(list(x.index), n_splits))
        model = HighAgreementRandomForest(n_trees=n_trees, max_depth=max_depth,
                                          agreement_percentage=agreement_percentage, max_features=max_features)
        predictions = pandas.Series(dtype=object)
        for i in list(range(n_splits)):
            idx_test = index_list_n[i]
            idx_train = [x for x in index if x not in index_list_n[i]]
            model.fit(x.iloc[idx_train], y.iloc[idx_train])
            predictions = pd.concat([predictions, model.predict(x.iloc[idx_test])])
        predictions_bool = prediction_ntb(predictions)
        predicted_noise = pd.Series(index=predictions_bool.index, dtype=bool)
        print("In: {}, model HARF5{} found the following instances to be noise"
              .format(y.name, round(agreement_percentage * 100)))
        print("Start of list:")
        for x in predictions_bool.index:
            if predictions_bool.loc[x] == y.loc[x]:
                predicted_noise.loc[x] = False

            elif predictions_bool.loc[x] == "NN":
                predicted_noise.loc[x] = False
            else:
                predicted_noise.loc[x] = True
                print("Instance with index: {} is tagged as noise".format(x))
        print("- End of list.")
        predicted_noise.sort_index(inplace=True)
        list_predicted_noise.append(predicted_noise.tolist())

    df_predicted_noise = pd.DataFrame(list_predicted_noise)
    df_predicted_noise = df_predicted_noise.transpose()
    df_predicted_noise.columns = y_data.columns
    return df_predicted_noise


def performance_eval(data, n_eval, train_size=0.8, agreement_percentage=0.8,
                     noise_percent=0.02,
                     tree_max_features=None):
    """
    This function evaluates the High Agreement Random Forest on the given
    dataset. It only works if there is only one target column, and it is in the last position.
    Not to be used for evaluating datasets with many target columns
    :param data: The dataset that will be evaluated for noise. It must include
    the x and y with y and y as a pandas Data
    Frame. The last column must be y
    :param n_eval: The number of times the evaluation will be done.
    :param train_size: The percentage of the dataset that will be allocated
    for the training set, with the test set
    being the opposite
    :param agreement_percentage: This parameter indicates how high the
    percentage should be for the majority voting in the random forest.
    A value of 0.8 indicates that 80% of the decision trees must have
    the same prediction value in order to make a decision on whether
    the label is noise or not
    :param noise_percent: The percentage of the test dataset that will
    be turned into noise
    :param tree_max_features:The maximal amount of features a tree can have.
    :return: This function returns the average F1-score, avg. precision and
    avg. recall while also printing these values on the console
    """
    x = data.copy()
    y = x.iloc[:, -1:].squeeze()
    x.drop(y.name, axis=1, inplace=True)
    y.replace(list(y.unique()), list([True, False]), inplace=True)
    f1_score, precision, recall = 0, 0, 0

    for i in list(range(n_eval)):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size,
                                                            test_size=1 - train_size)
        s = (x_train.dtypes == 'object')
        object_cols = list(s[s].index)
        # col_names = list(x.columns)
        ord_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=100)
        x_train_oe = x_train.copy()
        x_test_oe = x_test.copy()
        x_train_oe[object_cols] = ord_encoder.fit_transform(x_train[object_cols])
        x_test_oe[object_cols] = ord_encoder.transform(x_test[object_cols])
        model = HighAgreementRandomForest(agreement_percentage=agreement_percentage, max_features=tree_max_features)
        model.fit(x_train_oe, y_train)
        predictions = model.predict(x_test_oe)
        y_noise, indexes_noise = add_noise_index(y_test, noise_percent=noise_percent)
        predictions_boolean = prediction_ntb(predictions)
        predictions_noise, noise = get_noise_predicted(predictions_boolean, y_noise, y_test)
        f1_score = f1_score + fbeta_score(noise, predictions_noise, beta=1)
        """
        # Weighted and macro F1 scores not relevant for binary classification.
        weighted_f1 = weighted_f1 + fbeta_score(noise, predictions_noise, beta=1, average='weighted')
        macro_f1 = macro_f1 + fbeta_score(noise, predictions_noise, beta=1, average='macro')
        
        """
        precision = precision + precision_score(noise, predictions_noise)
        recall = recall + recall_score(noise, predictions_noise)

    f1_score = f1_score / n_eval
    # weighted_f1 = weighted_f1 / n_eval
    # macro_f1 = macro_f1 / n_eval
    precision = precision / n_eval
    recall = recall / n_eval

    print("For the {} dataset the HARF-5{} was evaluated {} times at a "
          "noise percentage of {}% with the following scores: "
          .format(y.name, round(agreement_percentage * 100), n_eval, round(noise_percent * 100)))
    print("F1-Score: {}".format(f1_score))
    # print("Weighted F1-Score: {}".format(weighted_f1))
    # print("Macro F1-Score: {}".format(macro_f1))
    print("Precision-Score: {}".format(precision))
    print("Recall-Score: {}".format(recall))
    return f1_score, precision, recall


def performance_eval_x_y(x, y, n_eval, train_size=0.8, agreement_percentage=0.8, noise_percent=0.02,
                         max_features=None, n_trees=500, max_depth=12):
    """
    This function does the same as the performance_eval function but instead of taking
    the entire dataset, it requires that the division between features and target
    values is made prior to entering them as parameters.

    :param max_features: Maximum number of features that will be analyzed when looking for the best split
    :param max_depth: Maximum depth of each decision tree
    :param n_trees: Number of trees in the RF
    :param x: This parameter should include the features X as a pandas DataFrame
    :param y: This parameter should include the target values Y as a pandas DataFrame
    :param n_eval: The number of times the evaluation will be done.
    :param train_size: The percentage of the dataset that will be allocated
    for the training set, with the test set
    being the opposite
    :param agreement_percentage: This parameter indicates how high the
    percentage should be for the majority voting in the random forest.
    A value of 0.8 indicates that 80% of the decision trees must have
    the same prediction value in order to make a decision on whether
    the label is noise or not
    :param noise_percent: The percentage of the test dataset that will
    be turned into noise
    :return: This function returns the average F1-score, avg. precision and
    avg. recall while also printing these values on the console
    """
    x = x.copy()
    y = y.copy()
    y.replace(list(y.unique()), list([True, False]), inplace=True)
    f1_score, precision, recall = 0, 0, 0
    # weighted_f1, macro_f1 = 0, 0
    for i in list(range(n_eval)):
        # print("Evaluation {} from {}".format(i+1, n_eval))
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size,
                                                            test_size=1 - train_size)
        s = (x_train.dtypes == 'object')
        object_cols = list(s[s].index)

        ord_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=100)
        x_train_oe = x_train.copy()
        x_test_oe = x_test.copy()
        # Only relevant for non-binary datasets
        x_train_oe[object_cols] = ord_encoder.fit_transform(x_train[object_cols])
        x_test_oe[object_cols] = ord_encoder.transform(x_test[object_cols])
        model = HighAgreementRandomForest(agreement_percentage=agreement_percentage, max_features=max_features,
                                          n_trees=n_trees, max_depth=max_depth)
        model.fit(x_train_oe, y_train)
        predictions = model.predict(x_test_oe)
        y_noise, indexes_noise = add_noise_index(y_test, noise_percent=noise_percent)
        predictions_boolean = prediction_ntb(predictions)
        predictions_noise, noise = get_noise_predicted(predictions_boolean, y_noise, y_test)
        f1_score = f1_score + fbeta_score(noise, predictions_noise, beta=1)
        # weighted_f1 = weighted_f1 + fbeta_score(noise, predictions_noise, beta=1, average='weighted')
        # macro_f1 = macro_f1 + fbeta_score(noise, predictions_noise, beta=1, average='macro')
        precision = precision + precision_score(noise, predictions_noise)
        recall = recall + recall_score(noise, predictions_noise)

    f1_score = f1_score / n_eval
    # weighted_f1 = weighted_f1 / n_eval
    # macro_f1 = macro_f1 / n_eval
    precision = precision / n_eval
    recall = recall / n_eval

    # print("For the {} dataset the HARF-5{} was evaluated {} times at a "
    #     "noise percentage of {}% with the following scores: "
    #     .format(y.name, round(agreement_percentage * 100), n_eval, round(noise_percent * 100)))

    return f1_score, precision, recall  # , weighted_f1, macro_f1


def evaluate_method1a_n_eval(x_data, y_list, n_eval=10, n_trees=500, train_size=0.8, agreement_percentage=0.8,
                   max_depth=10, noise_percent=0.02, max_features=None):
    """
    This function automates the process of evaluating an entire set of target columns for noise with method M1A. It collects the
    F1, recall and precision values achieved in all the analyzed target columns
    The evaluation is done with the performance_eval_cv_n function, which evaluates the entire target colum for noise
    for when it cannot be assumed that the training dataset is noise free.
    :param train_size: The percentage of the input data that will be used for training the model
    :param max_features: Maximum features to consider when looking for the best split
    :param x_data: pandas DataFrame containing the feature vector set
    :param y_list: pandas DataFrame containing the list of all the target columns to be evaluated
    :param n_eval: Number of evaluations performed per target column
    :param n_trees: Number of decision trees in the RF
    :param agreement_percentage: The agreement percentage of the RF
    :param max_depth: Maximum depth of the trees
    :param noise_percent: The noise percentage that will be artificially added to the dataset
    :return:
    """
    list_values = list()
    for i in list(range(len(y_list.columns))):
        print("Current Label: {}".format(y_list.iloc[:, i].name))
        f1, p, r = performance_eval_x_y(x=x_data,
                                        y=y_list.iloc[:, i],
                                        n_eval=n_eval, agreement_percentage=agreement_percentage,
                                        n_trees=n_trees, max_depth=max_depth, noise_percent=noise_percent,
                                        max_features=max_features, train_size=train_size)
        print("For {}: F1={} p={} r={}".format(y_list.iloc[:, i].name,
                                               f1, p, r))
        list_values.append([y_list.iloc[:, i].name, f1, p, r])

    values_pd = pd.DataFrame(list_values, columns=['Label name', 'F1 Score', 'Precision', 'Recall'])
    return values_pd


def performance_eval_trivial(x, y, n_eval, train_size=0.8, agreement_percentage=0.8, noise_percent=0.02,
                             tree_max_features=None):
    """
    This function is the same as the performance_eval_x_y function, but it works with
    y vectors with only one value.

    :param x: This parameter should include the features X as a pandas DataFrame
    :param y: This parameter should include the target values Y as a pandas DataFrame
    :param n_eval: The number of times the evaluation will be done.
    :param train_size: The percentage of the dataset that will be allocated
    for the training set, with the test set
    being the opposite
    :param agreement_percentage: This parameter indicates how high the
    percentage should be for the majority voting in the random forest.
    A value of 0.8 indicates that 80% of the decision trees must have
    the same prediction value in order to make a decision on whether
    the label is noise or not
    :param noise_percent: The percentage of the test dataset that will
    be turned into noise
    :param tree_max_features:The maximal amount of features a tree can have.
    :return: This function returns the average F1-score, avg. precision and
    avg. recall while also printing these values on the console
    """
    x = x.copy()
    y = y.copy()
    y.replace(list(y.unique()), list([True]), inplace=True)
    f1_score, precision, recall = 0, 0, 0
    # weighted_f1, macro_f1 = 0,0
    for i in list(range(n_eval)):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size,
                                                            test_size=1 - train_size)
        s = (x_train.dtypes == 'object')
        object_cols = list(s[s].index)
        # col_names = list(x.columns)
        ord_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=100)
        x_train_oe = x_train.copy()
        x_test_oe = x_test.copy()
        x_train_oe[object_cols] = ord_encoder.fit_transform(x_train[object_cols])
        x_test_oe[object_cols] = ord_encoder.transform(x_test[object_cols])
        model = HighAgreementRandomForest(agreement_percentage=agreement_percentage, max_features=tree_max_features)
        model.fit(x_train_oe, y_train)
        predictions = model.predict(x_test_oe)
        y_noise, indexes_noise = add_noise_index(y_test, noise_percent=noise_percent)
        predictions_boolean = prediction_ntb(predictions)
        predictions_noise, noise = get_noise_predicted(predictions_boolean, y_noise, y_test)
        f1_score = f1_score + fbeta_score(noise, predictions_noise, beta=1)
        # weighted_f1 = weighted_f1 + fbeta_score(noise, predictions_noise, beta=1, average='weighted')
        # macro_f1 = macro_f1 + fbeta_score(noise, predictions_noise, beta=1, average='macro')
        precision = precision + precision_score(noise, predictions_noise)
        recall = recall + recall_score(noise, predictions_noise)

    f1_score = f1_score / n_eval
    # weighted_f1 = weighted_f1 / n_eval
    # macro_f1 = macro_f1 / n_eval
    precision = precision / n_eval
    recall = recall / n_eval

    print("For the {} dataset the HARF-5{} was evaluated {} times at a "
          "noise percentage of {}% with the following scores: "
          .format(y.name, round(agreement_percentage * 100), n_eval, round(noise_percent * 100)))
    print("F1-Score: {}".format(f1_score))
    # print("Weighted F1-Score: {}".format(weighted_f1))
    # print("Macro F1-Score: {}".format(macro_f1))
    print("Precision-Score: {}".format(precision))
    print("Recall-Score: {}".format(recall))
    return f1_score, precision, recall


def split(a, n):
    """
    This function splits a given list into n parts, which are outputted into another list.
    Each partition is the n-1th element of the new list.
    Source: https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length

    :param n: The number of lists that the main list will be split into
    :param a: the input list
    :return:
    The list containing the other lists as elements
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def performance_eval_cv_n(x1, y1, n=10, n_trees=500, max_depth=10, noise_percentage=0.02):
    """
    This function evaluates the entire dataset for noise, for when it cannot be assumed that the trained dataset
    doesn't contain noise. It adds noise artificially at

    :param max_depth:
    :param n_trees:
    :param noise_percentage: The percentage of noise that will be added to the dataset
    :param x1: The input feature vector as pandas DataFrame
    :param y1: The input target column as pandas DataFrame
    :param n: Number of splits that the dataset will be split into
    :return: The F1, precision and recall scores for the given dataset
    """
    x = x1.copy()
    y = y1.copy()
    x, y = shuffle_samples(x, y)
    index = list(x.index)
    d = len(y.unique())
    if d == 2:
        y.replace(list(y.unique()), list([True, False]), inplace=True)
    else:
        y.replace(list(y.unique()), list([True]), inplace=True)
    y_noise, indexes_noise = add_noise_index(y, noise_percentage)
    index_list_n = list(split(list(x.index), n))
    f1_score, precision, recall = 0, 0, 0
    model = HighAgreementRandomForest(n_trees=n_trees, max_depth=max_depth)
    predictions = pandas.Series(dtype=object)
    for i in list(range(n)):
        idx_test = index_list_n[i]
        idx_train = [x for x in index if x not in index_list_n[i]]
        model.fit(x.iloc[idx_train], y_noise.iloc[idx_train])
        predictions = pd.concat([predictions, model.predict(x.iloc[idx_test])])

    predictions_bool = prediction_ntb(predictions)
    predictions_noise, noise = get_noise_predicted(predictions_bool, y_noise, y)
    f1_score = fbeta_score(noise, predictions_noise, beta=1)
    precision = precision_score(noise, predictions_noise)
    recall = recall_score(noise, predictions_noise)

    return f1_score, precision, recall


def evaluate_method1b_n_eval(x_data, y_list, n_eval=10, m=10, n_trees=500, max_depth=9):
    """
    This function automates the process of evaluating an entire set of target columns for noise. It collects the
    F1, recall and precision values achieved in all the analyzed target columns
    The evaluation is done with the performance_eval_cv_n function, which evaluates the entire target colum for noise
    for when it cannot be assumed that the training dataset is noise free.
    :param max_depth: Maximum depth of the decsion trees
    :param n_trees: Number of trees in the RF
    :param x_data: The feature vector as a pandas DataFrame
    :param y_list: The list of target columns as a pandas DataFrame
    :param n_eval: The number of total evaluations per target column performed
    :param m: The number of total splits that the dataset will be split into
    :return:
    It returns the F1, precision and recall scores for the entire evaluation process as a pandas DataFrame,
    which can be then stored in a .csv file.
    """
    list_values = list()
    for i in list(range(len(y_list.columns))):
        print("Current Label: {}".format(y_list.iloc[:, i].name))
        # F1, Precision and Recall scores
        f1, p, r = 0, 0, 0
        for j in list(range(n_eval)):
            print("Current CV-Evaluation {} from {}".format(j + 1, n_eval))
            f1_, p_, r_ = performance_eval_cv_n(x1=x_data, y1=y_list.iloc[:, i], n=m,
                                                n_trees=n_trees, max_depth=max_depth)
            f1 = f1 + f1_
            p = p + p_
            r = r + r_

        f1 = f1 / n_eval
        p = p / n_eval
        r = r / n_eval
        print("For {}: F1={} p={} r={}".format(y_list.iloc[:, i].name,
                                               f1, p, r))
        list_values.append([y_list.iloc[:, i].name, f1, p, r])
    values_pd = pd.DataFrame(list_values, columns=['Label name', 'F1 Score', 'Precision', 'Recall'])
    return values_pd


def perf_eval_single_n(x_data, y_list, n_eval=10, n_trees=500, agreement_percentage=0.08, max_depth=3):
    """
    This function automates the process of evaluating an entire set of target columns for noise. It collects the
    F1, recall and precision values achieved in all the analyzed target columns
    The evaluation is done with the perf_eval_single_split function, which evaluates the entire target colum for noise
    for when it cannot be assumed that the training dataset is noise free, but without any splits.
    :param n_trees: Number of trees in the RF
    :param max_depth: Maximum depth of the decision trees
    :param agreement_percentage: The agreement percentage of the Random Forest in 0.# format
    :param x_data: The feature vector as a pandas DataFrame
    :param y_list: The list of target columns as a pandas DataFrame
    :param n_eval: The number of total evaluations per target column performed
    :return:
    It returns the F1, precision and recall scores for the entire evaluation process as a pandas DataFrame,
    which can be then stored in a .csv file.
    """
    list_values = list()
    for i in list(range(len(y_list.columns))):
        print("Current Label: {}".format(y_list.iloc[:, i].name))
        # F1, Precision and Recall scores
        f1, p, r = 0, 0, 0
        for j in list(range(n_eval)):
            # print("Current CV-Evaluation {} from {}".format(j + 1, n_eval))
            f1_, p_, r_ = perf_eval_single_split(x1=x_data, y1=y_list.iloc[:, i], n_trees=n_trees,
                                                 agreement_percentage=agreement_percentage, max_depth=max_depth)
            f1 = f1 + f1_
            p = p + p_
            r = r + r_

        f1 = f1 / n_eval
        p = p / n_eval
        r = r / n_eval
        print("For {}: F1={} p={} r={}".format(y_list.iloc[:, i].name,
                                               f1, p, r))
        list_values.append([y_list.iloc[:, i].name, f1, p, r])
    values_pd = pd.DataFrame(list_values, columns=['Label name', 'F1 Score', 'Precision', 'Recall'])
    return values_pd


def perf_eval_single_split(x1, y1, n_trees, agreement_percentage, max_depth):
    """
    This function evaluates method M1B with only no splits. That is, the entire dataset is used to train the RF and
    tested on this RF. The dataset is artificially injected with 2% noise.
    :param x1: pandas DataFrame (DF), Feature vector set
    :param y1: pandas DF, current target column to be analyzed
    :param n_trees: Number of trees in the forest
    :param agreement_percentage: The agreement percentage of the Random Forest in 0.# format
    :param max_depth: Maximum depth of the tree
    :return:
    Returns the F1, precision and recall scores for the analyzed dataset
    """
    x = x1.copy()
    y = y1.copy()
    x, y = shuffle_samples(x, y)
    y.replace(list(y.unique()), list([True, False]), inplace=True)
    y_noise, indexes_noise = add_noise_index(y, 0.02)
    f1_score, precision, recall = 0, 0, 0
    model = HighAgreementRandomForest(n_trees=n_trees, agreement_percentage=agreement_percentage, max_depth=max_depth)
    model.fit(x, y_noise)
    predictions = model.predict(x)
    predictions_bool = prediction_ntb(predictions)
    predictions_noise, noise = get_noise_predicted(predictions_bool, y_noise, y)
    f1_score = fbeta_score(noise, predictions_noise, beta=1)
    precision = precision_score(noise, predictions_noise)
    recall = recall_score(noise, predictions_noise)

    return f1_score, precision, recall


class Aux:
    """
    This class contains functions that do not play a role in the training, fitting and prediction
    process of the HARF and rather compute the noise values for each instance after taking the outputs
    of the HARF model.

    """
