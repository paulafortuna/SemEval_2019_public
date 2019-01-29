
### Prepare imports
import pandas as pd
from scripts_pipeline.PathsManagement import Var, PathsManagement as Paths
from scripts_pipeline.Dataset import Dataset
from scripts_pipeline.feature_extraction.FeaturesList import FeaturesClassifierBuilder
from scripts_pipeline.data_preprocessing.DataTransformation import DataTransformation
from scripts_pipeline.classification_models.ClassifiersList import ClassifiersList


class Experiment(FeaturesClassifierBuilder):
    """
    Allows the execution of one classification pipeline with a set of parameters.
    Different objects can be instantiated and different experiments can run at same time.
    Check Class Var to discover the parameters to use.
    """
    def __init__(self,
                 experiment_name,
                 dataset_name,
                 apply_data_preprocessing,
                 features_to_extract,
                 features_to_use,
                 normalize_data,
                 classifier_name,
                 consider_class_weight,
                 folds_cross_validation,
                 use_grid_search,
                 epochs,
                 batch
                 ):

        """
        Sets the experiment parameters and runs the pipeline methods.
        :param experiment_name: the name of the experiment. Should be unique so that identifies models and results files.
        :param dataset_name: the dataset to be used.
        :param apply_data_preprocessing: list of strings with identifiers for the preprocessing methods to be applied
        :param max_num_words: maximum number of words to be considered in the word embedding extraction
        :param glove_model_dim: the number of dimension to be used in the glove model (50, 100 or 300)
        :param features_to_extract: list of strings with identifiers for the features to extract. [] means no features
        :param features_to_use: list of strings with identifiers for the features to use. [] will cause error in this case.
        :param normalize_data: list of strings with identifiers for the normalization procedure, [] means no normalization.
        :param classifier_name: list of strings with identifiers for the classifiers to run. [] means no classification.
        :param consider_class_weight: boolean that indicates if we want to pass the class_weight parameter to the classifiers
        :param folds_cross_validation: number of folds to use
        """
        print("Running experiment " + experiment_name)

        # prepare experiment parameters
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
        self.apply_data_preprocessing = apply_data_preprocessing
        self.features_to_extract = features_to_extract
        self.features_to_use = features_to_use
        self.normalize_data = normalize_data
        self.transformation = None
        self.classifier_name = classifier_name
        self.consider_class_weight = consider_class_weight
        self.folds_cross_validation = folds_cross_validation
        self.use_grid_search = use_grid_search
        self.dataset = None
        self.classifier_object = None
        self.classifiers_dict = None
        self.features_parameters = None
        self.features_objects = None
        self.batch = batch
        self.epochs = epochs

        # if directory does not exist, create folder for all features of a dataset
        Paths.assure_features_directory_for_dataset_exists(self.dataset_name)

    def start_experiment(self):
        """
        Starts an experiment with a predefined sequence of procedures.
        """
        # Read classification original_datasets
        self.dataset = Dataset()
        self.dataset.read_original_data(self.dataset_name)

        # Preprocess data
        self.preprocess_data()

        # extract features
        self.extract_features()

        # read features for training and testing set
        all_features = self.read_features_train_test()

        # normalize data
        all_features = self.transform_data(all_features)

        # build classifier data
        self.build_classifier_data(all_features)

        # save experiment parameters
        self.save_experiment_parameters(all_features)

    def extract_features(self):
        """
        Procedure that uses a dictionary for iterating the feature extraction identifiers passed in experiment.
        For each of this identifiers runs the respective methods that will cause the feature extraction and saving of
        features to files.
        Check Class Var to discover the dictionary (features_parameters) parameters to use.
        """
        # builds the dictionaries of features
        self.features_parameters, self.features_objects = FeaturesClassifierBuilder().build_dictionary(self.dataset_name)

        # uses the two dictionaries for iterating the and running desired feature extraction procedures
        for feature in self.features_to_extract:
            self.features_objects[feature].conduct_feature_extraction_train_test(self.dataset)

    def preprocess_data(self):
        # TODO
        print("preprocess data not implemented")

    def read_features_train_test(self):
        """
        Read specified features from files and concatenate them by the given order.
        Check Class Var to discover the dictionary (features_parameters) parameters to use.
        :return: dataset with all the features concatenated and y values already set.
        """
        all_features = Dataset()

        # read all the desired features from file
        for feature in self.features_to_use:
            all_features.read_add_features(feature, self.dataset_name)

        # add the class as the final column
        all_features.labels_index = self.dataset.labels_index

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

    def build_classifier_data(self, all_features):
        """
        Conducts the specified procedure for data classification.
        Check Class Var to discover the dictionary (classifiers_dict) parameters to use.
        :param all_features: Dataset to be classified.
        :return: the dataset after the applied transformation if any.
        """
        # builds the dictionary of classifiers
        self.classifiers_dict = ClassifiersList().build_dictionary(self.epochs,self.batch)

        # uses the dictionaries for iterating the desired classifier
        self.classifier_object = self.classifiers_dict.get(self.classifier_name)
        self.classifier_object.train_classifier(self.classifier_name, all_features, self)

        self.classifier_object.save_model(self.classifier_name)
        self.classifier_object.load_model(self.experiment_name, self.classifier_name)
        self.classifier_object.predict_test_data()

    def toDataframe(self, dataset):
        """
        Converts an experiment parameters to a dataframe, in order to save it.
        :param dataset: the dataset used in the experiment
        :return: the dataframe generated from the experiment features.
        """
        d = {'experiment_name': [self.experiment_name],
             'dataset_name': [self.dataset_name],
             'apply_data_preprocessing': [self.apply_data_preprocessing],
             'features_to_use': [self.features_to_use],
             'normalize_data': [self.normalize_data],
             'classifiers':[self.classifier_name],
             'consider_class_weight': [self.consider_class_weight],
             'folds_cross_validation': [self.folds_cross_validation],
             'consider_class_weight': [self.consider_class_weight],
             'use_grid_search': [self.use_grid_search],
             'num_rows_train': [dataset.x_train.shape[0]],
             'num_rows_test': [dataset.x_val.shape[0]],
             'num_cols_train': [dataset.x_train.shape[1]],
             'num_cols_test': [dataset.x_val.shape[1]]
             }
        return pd.DataFrame(data=d)

    def save_experiment_parameters(self, dataset):
        """
        Saves the experiment parameters to a file.
        :param dataset: the dataset used in the experiment
        """
        experiment_parameters = self.toDataframe(dataset)
        Paths.utils_print_result_to_file(experiment_parameters, "Experiment_parameters")









