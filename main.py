# The following packages are required to run the code
import pandas as pd
from Aux import performance_eval, add_noise_index, prediction_ntb, get_noise_predicted, performance_eval_x_y, \
    performance_eval_trivial, performance_eval_cv_n, perf_eval_single_n, evaluate_method1a_n_eval, \
    find_noise_harf, find_noise_harf_no_train_data, evaluate_method1b_n_eval
# Change path to current location
data_industry_partner = pd.read_csv("/Users/ricardozacateco/Documents/AM - BARZH/07 Python/CompanySamples01.csv",
                                    sep=";")
data_industry_partner.name = "Company Samples"

# This for loops identifies all the feature columns which only contain one value.
# These will be ignored for most evaluations
index_single = list([])
for x in list(range(1088 - 268)):

    if data_industry_partner.iloc[:, x + 267].nunique() == 1:
        index_single.append(data_industry_partner.columns[x + 267])

# Here the target columns are separated from the feature columns
data_industry_partner_x = data_industry_partner.iloc[:, :267]
data_industry_partner_ys = data_industry_partner.iloc[:, 267:]

# The target columns that contain only one value are separated from the rest
data_industry_partner_ys_no_single = data_industry_partner_ys.drop(columns=index_single, inplace=False)
data_industry_partner_ys_singles = data_industry_partner_ys[index_single]

# This evaluation shows the performance of method M1A on the target columns of the industry dataset excluiding
# The columns that only have single values
values_pd = evaluate_method1a_n_eval(x_data=data_industry_partner_x, y_list=data_industry_partner_ys_no_single,
                                     n_eval=10,
                                     max_depth=9, n_trees=500, agreement_percentage=0.8,
                                     noise_percent=0.02,
                                     max_features=None, train_size=0.8)
values_pd.to_csv("Test_Evaluation1.csv")

"""
# This evaluation shows the methods M1A performance on target columns that only have a single value
# This is strictly to prove that with a training dataset the evaluation is trivial    
for i in list(range(len(data_industry_partner_ys_singles.columns))):
    performance_eval_trivial(data_industry_partner_x,data_industry_partner_ys_singles.iloc[:, i], n_eval=3)
    print(i)
"""

"""
# This evaluation shows the performance of method M1B (no training dataset) with the given number of partitions (m)
values_pd_2 = evaluate_method1b_n_eval(x_data=data_industry_partner_x, y_list=data_industry_partner_ys_no_single,
                                   n_eval=10, m=10, n_trees=500, max_depth=9)
values_pd_2.to_csv('Test_Evaluation_CV20_no_singles.csv')
"""

"""
# This evaluation shows the performance of method M1B (no training dataset) on without splitting the dataset
# That is dataset is only analyzed once for noise with only one partition
values_pd_3 = perf_eval_single_n(x_data=data_industry_partner_x, y_list=data_industry_partner_ys_no_single, n_eval=10,
                                 n_trees=500)
values_pd_3.to_csv('Test_evaluationCV1.csv')
"""
############
############
# The following code simulates how the method could be actually used in a real environment

# This part simulates the method finding errors in the variant specific BOMs and routings with
# a training dataset available
"""
x_train = data_industry_partner_x.iloc[:399, :]
y_train = data_industry_partner_ys_no_single.iloc[:399, :6]

x_test = data_industry_partner_x.iloc[399:, :]
y_test = data_industry_partner_ys_no_single.iloc[399:, :6]
# Method M1A
# predicted_noise has the noise values that the models detected 
predicted_noise = find_noise_harf(x_train=x_train, y_train=y_train,
                                  x_test=x_test, y_test=y_test,
                                  n_trees=500, agreement_percentage=0.8)
predicted_noise.to_csv("Test with training1.csv")
"""
# Method M1B
# This is the case where no training data is available
# This is an exemplary process of how method M1B would check an entire dataset for noise
"""list_predicted_noise = find_noise_harf_no_train_data(data_industry_partner_x, data_industry_partner_ys
                                                     , n_splits=10, n_trees=500, agreement_percentage=0.8)

list_predicted_noise.to_csv("Noise_predictions_test1.csv")
"""
"""
first4 = data_industry_partner_ys_no_single.iloc[:, :4]
list_predicted_noise = find_noise_harf_no_train_data(data_industry_partner_x, first4
                                                     , n_splits=10, n_trees=10, agreement_percentage=0.8)

list_predicted_noise.to_csv("Noise_predictions_test21.csv")
"""
