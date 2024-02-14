# Bachelor Thesis Project - Automatic Validation of Production Planning Documents

This project seeks to automatically find incorrectly specified instances in datasets related to production planning documents.

More concretely, in the resulting bill of materials and routings for personalizable products. The resulting 
datasets of such documents are very large and their validation is usually a very time-consuming task. 
This method aims to facilitate the validation by automatically checking for errors, so that the 
experts only have to look at the specifications deemed as errors.
## Aux
The Aux class contains all the functions to find noise and evaluate the performance of such functions, 
when a ground truth is available.

To find noise in the datasets there is two versions of the method. The first version requires 
a seeing training data to predict class noise on the unseen dataset. Such training data-set is assumed to be error-free. This version is 
referred to as **Method M1A**.

When there is no training dataset available 

### Import 
```bash
import Aux 
```
## Usage
### Method M1A
```bash
import Aux

# returns a pandas DataFrame with boolean values for each value in 
# the target column. True indicating that the class value is considered noise and False that it's 
# not considered noise. 
Aux.find_noise_harf(x_train, y_train, x_test,y_test, 
                agreement_percentage=0.8, n_trees=100, 
                max_depth='sqrt')
```

### Method M1B
```bash
import Aux

# similar to Method M1A it returns a DataFrame with boolean values, to indicate noise.
Aux.find_noise_harf_no_train_data(x_data,y_data, 
                                  n_splits=10,agreement_percentage=0.8, 
                                  n_trees=100, 
```
More use examples can be found in the main.py file. Also, the class Aux contains the docstrings for 
all the functions mentioned above.
# -bachelorthesis
# thesisproject
# thesisproject
