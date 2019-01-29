from abc import abstractmethod, ABC

from scripts_pipeline.PathsManagement import PathsManagement as Paths


class Parameters:
    """
    Works as a Parent Class that for every set of FeatureExtraction procedure contains the parameters.
    """
    def __init__(self, feature_name, dataset_name):
        self.feature_name = feature_name
        self.dataset_name = dataset_name


class FeatureExtraction(ABC):
    """
    It is an abstract class that sets the interface for implementing FeatureExtraction procedures.
    Children classes are then forced to implement the abstract method conduct_feature_extraction
    so that they can be instantiated.
    It also encapsulates for all the feature extraction procedure the saving of files.
    """
    def __init__(self, parameters):
        """
        Initiator of the FeatureExtraction abstract class.
        :param parameters: object of the Parameters class.
        """
        self.parameters = parameters

    def save_features_to_file(self, features_dataset):
        """
        Save features to file.
        :param features_dataset: the Dataset object with the features for being saved.
        """
        features_dataset.save_features(self.parameters.feature_name, self.parameters.dataset_name)

    def save_features_to_file_from_new_data(self, features_dataset, new_data_name):
        """
        Save features to file.
        :param features_dataset: the Dataset object with the features for being saved.
        """
        features_dataset.save_features_x_val(self.parameters.feature_name, new_data_name)

    @abstractmethod
    def conduct_feature_extraction_train_test(self, original_dataset):
        """
        Abstract method that allows all the feature extraction procedures to be used in a list and called
        with the same method.
        :param original_dataset: the original dataset needed for extracting features
        """
        pass

    @abstractmethod
    def conduct_feature_extraction_new_data(self, new_data, new_data_name, experiment_name):
        """
        Abstract method that allows all the feature extraction procedures to be used in a list and called
        with the same method.
        :param original_dataset: the original dataset needed for extracting features
        """
        pass
