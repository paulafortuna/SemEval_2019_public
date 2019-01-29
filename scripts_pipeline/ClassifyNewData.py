from scripts_pipeline.Dataset import Dataset
from scripts_pipeline.PathsManagement import Var, PathsManagement as Paths
from scripts_pipeline.feature_extraction.FeaturesList import FeaturesClassifierBuilder
from scripts_pipeline.Experiment import Experiment
from scripts_pipeline.data_preprocessing.DataTransformation import DataTransformation
from scripts_pipeline.classification_models.ClassifiersList import ClassifiersList

import pandas as pd
import numpy as np
from ast import literal_eval


class ClassifyNewContestDataAfterExperiment(FeaturesClassifierBuilder):
    """
    Allows the execution of one classification pipeline based on a Experiment, this time to classify new data.
    Check results folder and Experiment_parameters.csv to know which experiment to run.
    """

    def __init__(self, exp_name):
        """
        Sets the experiment to use for classifying new data.
        Reads from file the parameters of the new experiment
        :param exp_name: the experiment to use for classifying new data.
        """

        # find experiment in list
        parameters = Paths.get_experiment_parameters_from_file(exp_name)

        # The last entrance of the experiment is used
        parameters = parameters.iloc[[-1]]
        parameters.index = [0]

        self.experiment_name = exp_name
        self.dataset_name = parameters['dataset_name'][0]
        self.apply_data_preprocessing = literal_eval(parameters['apply_data_preprocessing'][0])
        self.features_to_use = literal_eval(parameters['features_to_use'][0])
        self.normalize_data = parameters['normalize_data'][0]
        self.classifier_name = parameters['classifiers'][0]
        self.consider_class_weight = parameters['consider_class_weight'][0]
        self.folds_cross_validation = parameters['folds_cross_validation'][0]
        self.use_grid_search = parameters['use_grid_search'][0]
        self.is_extract_features = None
        self.is_include_ids = None
        self.new_data_name = None
        self.new_data = None
        self.classifiers_dict = None
        self.features_parameters = None
        self.features_objects = None

    def start_classification(self, new_data_path, new_data_name, is_extract_features, is_include_ids):
        """
        Starts the classification of the new data with a predefined sequence of procedures.
        """
        self.is_extract_features = is_extract_features
        self.is_include_ids = is_include_ids
        self.new_data_name = new_data_name

        # read new data
        self.new_data = Dataset()
        self.new_data.get_new_data_to_classify(new_data_path, self.is_include_ids)

        # extract features
        self.extract_features()

        # read features
        all_features = self.read_features()

        # normalize data
        all_features = self.transform_data(all_features)

        # classify new data and save it to file
        self.new_data = self.classify_data(all_features)

        # save results to file and include ids if specified
        if self.classifier_name != Var.LSTMFeatures:
            self.new_data.print_prediction_to_file(new_data_name, self.experiment_name, self.is_include_ids)

    def extract_features(self):
        """
        Procedure that uses a dictionary for iterating the feature extraction identifiers passed in experiment.
        For each of this identifiers runs the respective methods that will cause the feature extraction and saving of
        features to files.
        Check Class Var to discover the dictionary (features_parameters) parameters to use.
        """
        # create folder for all features of a dataset
        Paths.assure_features_directory_for_dataset_exists(self.new_data_name)

        self.features_parameters, self.features_objects = FeaturesClassifierBuilder().build_dictionary(
            self.dataset_name)

        if self.is_extract_features:
            for feature in self.features_to_use:
                if feature != Var.LSTMFeatures:
                    self.features_objects[feature].conduct_feature_extraction_new_data(self.new_data,
                                                                                       self.new_data_name,
                                                                                       self.experiment_name)

    def read_features(self):
        """
        Read specified features from files and concatenate them by the given order.
        Check Class Var to discover the dictionary (features_parameters) parameters to use.
        :return: dataset with all the features concatenated and y values already set.
        """
        all_features = Dataset()

        # read all the desired features from file
        for feature in self.features_to_use:
            all_features.read_add_features_new_data(feature, self.new_data_name)

        return all_features

    def transform_data(self, all_features):
        """
        Conducts the specified procedure for data transformation after feature extraction.
        Check Class Var to discover the dictionary (data_transformation) parameters to use.
        :param all_features: dataset to be transformed.
        :return: the dataset after the applied transformation if any.
        """
        # if some normalization was defined apply it!
        if self.normalize_data != Var.none:
            self.transformation = DataTransformation(self.normalize_data, all_features.x_train)
            all_features = self.transformation.apply_transformation_all_features(all_features)

        return all_features

    def classify_data(self, all_features):
        """
        Conducts the specified procedure for data classification.
        :param all_features: Dataset to be classified.
        :return: the dataset after the classification.
        """
        self.classifiers_dict = ClassifiersList().build_dictionary(10,64)
        # uses the dictionaries for iterating the desired classifier
        self.classifier_object = self.classifiers_dict.get(self.classifier_name)
        self.classifier_object.load_model(self.experiment_name, self.classifier_name)
        return self.classifier_object.predict_new_data(all_features, self.new_data_name)

    def include_ids_file(self, dataset):
        """
        Concatenates the ids to the classified data and saves it to file.
        :param dataset: Classified dataset.
        """
        # get classification column
        classification_data = pd.read_csv(self.new_data_path, delimiter='\t', encoding='utf-8')
        classification_data = np.loadtxt(Paths.generate_path_new_data_classification(self.new_data_name, self.exp.experiment_name),
                   delimiter='\t')

        # merge both
        d = {'id': dataset.x_val_id, 'classification_data': classification_data}
        final_data = pd.DataFrame(data=d)

        # save result in file
        np.savetxt(Paths.generate_path_new_data_classification(self.new_data_name, self.exp.experiment_name), final_data, '%d', delimiter='\t')

