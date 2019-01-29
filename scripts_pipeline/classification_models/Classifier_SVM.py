
from sklearn.svm import SVC
import pandas as pd

from scripts_pipeline.classification_models.Classifier_CvGridSearch import CvGridSearchClassifier


class SVM(CvGridSearchClassifier):
    """
    Implements methods for SVM classification with sklearn.
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
        return self.linear_SVM_search_C_gamma(model_name)

    def linear_SVM_search_C_gamma(self, model_name):
        """
        Defines a set of parameters to be used in the classification.
        Invokes the parent class for CV and grid search classification with specified parameters.
        :param model_name: the model_name to be used in the saving of the model to files.
        """
        # defines the parameters values and parameter grid
        if self.exp.use_grid_search:
            param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                          'gamma': ['auto',0.001, 0.01, 0.1, 1, 10, 100]}
        else:
            param_grid = {'C': [1],
                          'gamma': ['auto']}

        # defines the model to be used
        if(self.exp.consider_class_weight):
            labels_dict = self.compute_class_weight()
            labels_dict = "balanced"
            model = SVC(class_weight=labels_dict)
        else:
            model = SVC()

        # invokes the parent class for CV and grid search classification with specified parameters
        return self.cv_grid_search_scikit_classifier(model, param_grid, model_name)

    def compute_class_weight(self):
        """
        Computes the classes frequencies and puts it in a dictionary format.
        :return: Dictionary with the classes frequencies.
        """
        labels_dict = pd.DataFrame(self.dataset.y_train, columns=['col1'])
        return labels_dict['col1'].value_counts().to_dict()







