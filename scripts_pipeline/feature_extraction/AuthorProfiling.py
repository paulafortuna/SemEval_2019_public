
from scripts_pipeline.feature_extraction.FeatureExtraction import Parameters, FeatureExtraction
from scripts_pipeline.Dataset import Dataset


class AuthorProfilingParameters(Parameters):
    """
    Parameters to be used in the author profiling feature extraction.
    """
    def __init__(self, feature_name, dataset_name):
        Parameters.__init__(self, feature_name, dataset_name)


class AuthorProfiling(FeatureExtraction):
    """
    Implements methods for author profiling feature extraction.
    It is a Child Class of FeatureExtraction, and can be instantiated because it implements
    conduct_feature_extraction method.
    """
    def __init__(self, parameters):
        """
        Calls the parent class FeatureExtraction initiator.
        :param parameters: object of the AuthorProfilingParameters.
        """
        FeatureExtraction.__init__(self, parameters)

    def conduct_feature_extraction_train_test(self, original_dataset):
        """
        Assures the feature extraction occurs in the right order of steps.
        Implementation of the abstract method from the parent class.
        :param original_dataset:the original data to be extracted
        """
        author_ft = self.extract_author_profiling(original_dataset)
        self.save_features_to_file(author_ft)

    def conduct_feature_extraction_new_data(self, new_data, new_data_name):
        """
        Abstract method that allows all the feature extraction procedures to be used in a list and called
        with the same method.
        :param original_dataset: the original dataset needed for extracting features
        """
        # read files where the features were extracted
        author_ft = Dataset()
        author_ft.read_author_profiling_features_path(new_data_name)

        # get only the features and put in the same shape as the other features
        author_ft.x_val = self.put_features_correct_format(author_ft.x_val)

        # save
        self.save_features_to_file_from_new_data(author_ft, new_data_name)

    def extract_author_profiling(self, original_dataset):
        """
        Method that really implements the feature extraction procedure.
        Reads the data produced by AUTOPROTK. Check official article for understanding of features.
        :param original_dataset: texts columns
        :return: the extracted features
        """
        # read files where the features were extracted
        author_ft = Dataset()
        author_ft.set_ids_and_y(original_dataset)
        author_ft.read_author_profiling_features()

        # get only the features and put in the same shape as the other features
        author_ft.x_train = self.put_features_correct_format(author_ft.x_train)
        author_ft.x_val = self.put_features_correct_format(author_ft.x_val)

        # save to files
        if len(original_dataset.x_train) != len(author_ft.x_train):
            print("Error")

        if len(author_ft.x_val) != len(original_dataset.x_val):
            print("Error")

        return author_ft

    def put_features_correct_format(self, x_features):
        """
        Auxiliary method that encapsulates the procedure for separating the several features.
        Those are merged in a single column "features" as result of running the AUPROTK.
        :param x_features: features read from file
        :return: the separated features
        """
        res = []
        for x in range(0, len(x_features)):
            temp = x_features[x].split(',')
            temp = [float(i) for i in temp]
            res.append(temp)
        return res






