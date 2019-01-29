
from sklearn.linear_model import LogisticRegression
import pandas as pd

from scripts_pipeline.classification_models.Classifier_CvGridSearch import CvGridSearchClassifier


class LogisticRegressionClassifier(CvGridSearchClassifier):
    """
    Implements methods for Logistic Regression classification with sklearn.
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
        return self.linear_logistic_regression_search_C(model_name)

    def linear_logistic_regression_search_C(self, model_name):
        """
        Defines a set of parameters to be used in the classification.
        Invokes the parent class for CV and grid search classification with specified parameters.
        :param model_name: the model_name to be used in the saving of the model to files.
        """
        # defines the parameters values and parameter grid
        if self.exp.use_grid_search:
            param_grid = {'C': [0, 0.01, 0.1, 1]}
        else:
            param_grid = {'C': [1]}

        # defines the model to be used
        if self.exp.consider_class_weight:
            labels_dict = "balanced"
            model = LogisticRegression(class_weight=labels_dict)
        else:
            model = LogisticRegression()

        # invokes the parent class for CV and grid search classification with specified parameters
        return self.cv_grid_search_scikit_classifier(model, param_grid, model_name)








