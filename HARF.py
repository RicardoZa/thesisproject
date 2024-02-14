from sklearn.tree import DecisionTreeRegressor
import numpy as np
from collections import Counter
import pandas as pd


def boostrap_samples(x, y):
    """
    This function creates bootstrapped samples out of the given sample.

    :param x: X values from the initial dataset
    :param y: y values from the initial dataset
    :return: Returns the bootstrapped dataset
    """
    n_samples = x.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return x.iloc[idxs], y.iloc[idxs]


class HighAgreementRandomForest:
    """
    This class contains the implementation for the High Agreement Random Forest
    based on the paper of Sluban et al. (2010).
    Concrete implementation based on:
    - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    - https://www.youtube.com/watch?v=kFwe2ZZU7yw&t=18s



    """
    def __init__(self, n_trees=500, max_depth=9, min_samples_split=2,
                 max_features=None, agreement_percentage=0.8):
        """
        Constructor function for the High Agreement Forest class.

        :param n_trees: Number of trees that the Random Forest will have. Default 500
        as in Sluban et al. (2010)
        :param max_depth: Max depth of the trees
        :param min_samples_split:
        :param max_features:
        :param agreement_percentage:

        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.agreement_percentage = agreement_percentage

    def fit(self, x, y):
        """
        This function fits all the decision trees in the forest.
        For the decision trees the model from scikit learn was
        used: sklearn.tree.DecisionTreeRegressor
        :param x: Training features
        :param y: Training target values
        :return: No return value but the self.trees array will include all
        the decision trees after activation
        """
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                         max_features=self.max_features)
            x_sample, y_sample = boostrap_samples(x, y)
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)

    def most_common_label(self, y):
        """
        This function finds the most common predicted label among the decision trees'
        predictions
        :param y: This parameter is the array containing the predictions made by all
        decision trees
        :return: The most common predicted label or a "NN" which stands for not noise
        for when the Random Forest doesn't meet the agreement requirement for the given x
        """
        agreement_percentage = self.agreement_percentage
        counter = Counter(y)
        c = counter.most_common()
        perc_most_common_label = c[0][1] / self.n_trees
        # If agreement of the Random Forest is not greater or equal than the set percentage
        # print("HELPVAL; Counter.most common: {}".format(c))
        # print("HELPVAL; Most common label {}: {}".format(c[0][0], perc_most_common_label))
        if perc_most_common_label >= agreement_percentage:
            return c[0][0]
        else:
            return "NN"

    def predict(self, x_test):
        """
        Makes the final prediction for the given x array.
        :param x_test: pandas Dataframe that will be used to make the prediction value of y
        :return: Returns the predictions as a pandas Series with only the value 1,0 and "NN"
        """
        # This line creates the predictions from all the randomly made trees
        # The predict function from DecisionTreeClassifier is used
        predictions = np.array([tree.predict(x_test) for tree in self.trees])
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0
        # Change the axis of the predictions array to make it compatible with the most_common_label for all Xs
        tree_predictions = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self.most_common_label(prediction) for prediction in tree_predictions])
        predictions_pd = pd.Series(predictions, name='Predictions', index=x_test.index)
        return predictions_pd
