
from sklearn.ensemble  import RandomForestClassifier
import pandas as pd

from scripts_pipeline.classification_models.Classifier_CvGridSearch import CvGridSearchClassifier


class RandomForest(CvGridSearchClassifier):
    """
    Implements methods for Random Forest classification with sklearn.
    It is a Child Class of CvGridSearchClassifier, and it can be instantiated because it implements
    apply_grid_search_cv_classifier method.
    """

    def __init__(self):
        CvGridSearchClassifier.__init__(self)

    def train_classifier(self, model_name, dataset, exp):
        """
        Calls the specific function that builds the parameters for the grid search.
        Implementation of the abstract method from the parent class.
        :param model_name: the model_name to be used in the saving of the model to files.
        """
        self.dataset = dataset
        self.exp = exp
        return self.linear_random_forest_search_n_estimators_max_depth_gamma(model_name)

    def linear_random_forest_search_n_estimators_max_depth_gamma(self, model_name):
        """
        Defines a set of parameters to be used in the classification.
        Invokes the parent class for CV and grid search classification with specified parameters.
        :param model_name: the model_name to be used in the saving of the model to files.
        """
        # defines the parameters values and parameter grid
        if self.exp.use_grid_search:
            param_grid = {'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200],
                          'max_depth': [None, 1, 32]
                          }
        else:
            param_grid = {'n_estimators': [100],
                          'max_depth': [None]
                          }

        param_grid['n_jobs'] = [-1]

        model = RandomForestClassifier()
        # invokes the parent class for CV and grid search classification with specifi
        return self.cv_grid_search_scikit_classifier(model, param_grid, model_name)

    def compute_class_weight(self):
        """
        Computes the classes frequencies and puts it in a dictionary format.
        :return: Dictionary with the classes frequencies.
        """
        labels_dict = pd.DataFrame(self.dataset.y_train, columns=['col1'])
        return labels_dict['col1'].value_counts().to_dict()







