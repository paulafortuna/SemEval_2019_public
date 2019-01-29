
from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from joblib import dump, load
import pandas as pd
import json

from scripts_pipeline.PathsManagement import PathsManagement as Paths


class CvGridSearchClassifier(ABC):
    """
    It is an abstract class that sets the interface for implementing CvGridSearchClassifier procedures.
    This means cross validation (Cv) with GridSearch as provided by scikit learn package.
    Children classes are then forced to implement the abstract method apply_grid_search_cv_classifier
    so that they can be instantiated.
    It also encapsulates for all the classification procedure the results metrics, the saving of files,
    including results report. Finally, also allows the classification of new data.
    """
    def __init__(self):
        self.dataset = None
        self.exp = None
        self.model = []

    @abstractmethod
    def train_classifier(self, dataset, exp):
        """
        Abstract method that allows all the classification procedures to be used in a list and called
        with the same method.
        """
        pass

    def cv_grid_search_scikit_classifier(self, model, parameter_grid, model_name):
        """
        Method that really implements the cross validation and grid search procedure.
        It uses the GridSearchCV from scikit learn and requires some arguments.
        :param model: the model to be tested
        :param parameter_grid: a dictionary (string -> list) with the parameters to be tested
        :param model_name: name of the model to be saved
        """
        # scikit learn model class constructor
        grid_search_model = GridSearchCV(model, parameter_grid, scoring=['f1_macro', 'precision', 'recall'], refit='f1_macro', cv=self.exp.folds_cross_validation,
                                         return_train_score=True)

        # scikit learn model fiting
        y_train = self.dataset.y_train.ravel()

        # shuffle
        x, y = shuffle(self.dataset.x_train, y_train, random_state=42)

        grid_search_model.fit(x, y)

        # save results of the cross validation procedure
        self.save_CV_results(grid_search_model, model_name)

        # store model for posterior use
        self.model = grid_search_model

    def save_model(self, model_name):
        """
        Generic model saving to files. (Does not work for keras. A specific saver is necessary in that case.)
        :param model_name: The name of the model to save.
        """
        model_path = Paths.generate_classification_model_path(self.exp.experiment_name, model_name) + '.joblib'
        dump(self.model, model_path)

    def load_model(self, experiment_name, model_name):
        """
        Load model from files. (Does not work for keras. A specific Loader is necessary in that case.)
        :param experiment_name: identifier of the experiment.
        :param model_name: the name of the model to load.
        """
        model_path = Paths.generate_classification_model_path(experiment_name, model_name) + '.joblib'
        self.model = load(model_path)

    def predict_test_data(self):
        """
        Applies the stored classifier to the test data.
        :param model_name: The name of the model to save.
        """
        # generate prediction
        y_pred = self.model.predict(self.dataset.x_val)

        # generate classification report and print it to screen and file
        results = classification_report(self.dataset.y_val, y_pred)
        results = self.test_results_toDataFrame(results)
        Paths.utils_print_result_to_file(results, "test_evaluation")

    def get_model_name(self):
        """
        Returns the model name as a string
        :return A string corresponding to the name of the model.
        """
        name = Paths.util_names_separator(self.exp.experiment_name,
                                          self.exp.dataset_name,
                                          '_'.join(self.exp.features_to_use),
                                          self.exp.normalize_data,
                                          str(self.exp.consider_class_weight),
                                          str(self.exp.folds_cross_validation),
                                          self.exp.classifier_name)
        return name

    def save_CV_results(self, grid_search_model, model_name):
        """
        Prints cross validation results and save it to file.
        :param grid_search_model: the model to be printed
        :param model_name: the name of the model to be used in the save of the reports
        """
        # prepare results
        results = self.transform_results_csv(grid_search_model, model_name)
        # save to file
        Paths.utils_print_result_to_file(results, "CV_results")

    def transform_results_csv(self, grid_search_model, model_name):
        """
        Transform the cross validation results into CSV.
        :param grid_search_model: The model used.
        :param model_name: The name of the model to save.
        """
        results = self.exp.toDataframe(self.dataset)
        results = results.assign(model_name=model_name)
        results = results.assign(F1=grid_search_model.best_score_)
        #results = results.assign(precision=grid_search_model.cv_results_['mean_test_precision'])
        #results = results.assign(recall=grid_search_model.cv_results_['mean_test_recall'])
        results = results.assign(best_parameter=json.dumps(grid_search_model.best_params_))
        return results

    def test_results_toDataFrame(self, results):
        """
        Prints the results to a csv
        :param results: The results to print.
        """
        print(results)

        results = results.replace('avg', ' ').strip().split()
        results = ['-'] + results

        col_names = results[1:5]
        data_frame_dict = {}
        for i in range(1, 6):
            class_name = results[i*5]
            for j in range(0, 4):
                data_frame_dict[col_names[j] + '.' + class_name] = [results[i*5+1+j]]

        return pd.concat([self.exp.toDataframe(self.dataset), pd.DataFrame.from_dict(data_frame_dict)], axis=1, sort=False)

    def predict_new_data(self, new_data, dataset_name):
        """
        Applies the stored classifier to new data.
        :param model_name: The name of the model to save.
        """
        # generate prediction
        new_data.y_val = self.model.predict(new_data.x_val)
        return new_data






