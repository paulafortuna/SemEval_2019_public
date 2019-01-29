import os
import pandas as pd

class Var:
    """Variables.
        Makes available the global variables for the experiment.
        Variables and methods should be accessed through the class and not through objects.
    """

    none = 'none'

    # Features
    sentiment_vader = 'sentiment_vader'
    author_profiling = 'author_profiling'
    glove_w_emb_50 = 'glove50d'
    glove_w_emb_100 = 'glove100d'
    glove_w_emb_300 = 'glove300d'
    hatebase = 'hatebase'
    glove_twitter_25_en = "gloveTwitter25d"
    glove_twitter_200_en = "gloveTwitter200d"
    glove_SBW_300_es = "SBW_300_es"

    # Normalization methods
    min_max = 'MinMax'
    normalize_gaussian = 'NormalizeGaussian'
    normalize_linear_algebra = 'NormalizeLinearAlgebra'

    #classifiers
    linear_svm = 'LinearSVM'
    CVgridSearchLSTM = 'CVgridSearchLSTM'
    LSTMFeatures = 'LSTMFeatures'
    xgBoost = 'xgBoost'
    LogisticRegressionClassifier = 'LogisticRegression'
    RandomForest = 'RandomForest'


class PathsManagement:
    """Class for handle the generation of paths.
    Variables and methods should be accessed through the class and not through objects.
    The directories necessary for running the project are listed here as arguments.
    Some of these will be created if they don't exist. Other have to be creeated manually and filled
    with the necessary data.
    """
    directory_project = os.getcwd()

    # datasets paths
    directory_original_data = os.path.join(directory_project, 'original_datasets')

    # hateval en paths
    directory_hateval_en = os.path.join(directory_original_data, 'public_development_en')
    hateval_en_text_data_train = os.path.join(directory_hateval_en, 'train_en.tsv')
    hateval_en_text_data_test = os.path.join(directory_hateval_en, 'dev_en.tsv')

    # hateval es paths
    directory_hateval_es = os.path.join(directory_original_data, 'public_development_es')
    hateval_es_text_data_train = os.path.join(directory_hateval_es, 'train_es.tsv')
    hateval_es_text_data_test = os.path.join(directory_hateval_es, 'dev_es.tsv')

    # offenseval
    directory_offenseval = os.path.join(directory_original_data, 'OffensEval')
    offenseval_en_text_data_train = os.path.join(directory_offenseval, 'offenseval-training-v1.tsv')
    offenseval_en_text_data_test = os.path.join(directory_offenseval, 'offenseval-trial.txt')

    # test dataset - zeerak
    directory_test_dataset = os.path.join(directory_original_data, 'zeerak')
    test_dataset = os.path.join(directory_test_dataset, 'dataset.csv')

    # 2) feature extraction paths
    # resources paths
    directory_resources = os.path.join(directory_project, 'resources')

    # hatebase file path
    file_hatebase = os.path.join(directory_resources, 'hate_base_terms.csv')

    # glove paths
    directory_glove = os.path.join(directory_resources, 'glove')
    directory_glove_twitter = os.path.join(directory_resources, 'glove_twitter')

    # spanish word embeddings file
    file_spanish_word_emb_SBW_300 = os.path.join(directory_resources, 'SBW-vectors-300-min5.txt')

    # 3) extracted features
    directory_saved_features = os.path.join(directory_project, 'saved_extracted_features')

    # extracted in another project
    directory_AUPROTK_features = os.path.join(directory_saved_features, 'AUPROTK_features')

    # 4) models for features and classification
    directory_features_extractors = os.path.join(directory_project, 'saved_feature_extractors')
    directory_classification_models = os.path.join(directory_project, 'saved_classification_models')
    save_obj_word_embeddings_glove = os.path.join(directory_features_extractors, 'word_embeddings_glove')

    # 5) results
    directory_results = os.path.join(directory_project, 'results')

    def __init__(self, base_dir):
        self.directory_project = base_dir

    @staticmethod
    def assure_features_directory_for_dataset_exists(dataset_name):
        """
        Checks inside of the saved_extracted_features directory if folder for features of the dataset already exist.
        If not, creates one.
        :param dataset_name: the name of the dataset.
        """
        directory_saved_features_dataset = os.path.join(PathsManagement.directory_saved_features, dataset_name)
        if not os.path.exists(directory_saved_features_dataset):
            os.mkdir(directory_saved_features_dataset)
            print("Directory ", directory_saved_features_dataset, " Created ")

    @staticmethod
    def verify_create_automatic_directory(path):
        """
        Creates path if does not exist.
        :param path: the name of the path to be created.
        """
        if not os.path.exists(path):
            os.mkdir(path)
            print("Directory ", path, " Created ")

    @staticmethod
    def verify_request_directory(path):
        """
        Requests user to create a non existent but necessary directory.
        :param path: the name of the path to be created.
        """
        if not os.path.exists(PathsManagement.directory_hateval_en):
            print("You need to create ", path)

    @staticmethod
    def generate_saved_feature_path(feature_name, dataset_name, train_or_test):
        """
        Generates the path of the features (only x dimension of data) to be saved.
        :param feature_name: the feature name
        :param dataset_name: the dataset from which features were extracted
        :param train_or_test: indication if the features are from "train", "test" or any other string can be passed
        :return: the path
        """
        directory_saved_features_dataset = os.path.join(PathsManagement.directory_saved_features, dataset_name)
        file_termination = PathsManagement.util_names_separator(feature_name, dataset_name,train_or_test)
        return os.path.join(directory_saved_features_dataset, file_termination)

    @staticmethod
    def generate_saved_feature_path_train(feature_name, dataset_name):
        """
        Similar to generate_saved_feature_path.
        """
        path = PathsManagement.generate_saved_feature_path(feature_name, dataset_name, "train")
        return path

    @staticmethod
    def generate_saved_feature_path_test(feature_name, dataset_name):
        """
        Similar to generate_saved_feature_path.
        """
        return PathsManagement.generate_saved_feature_path(feature_name, dataset_name, "test")

    @staticmethod
    def generate_glove_model_directory_path(dimension):
        """
        Similar to generate_saved_feature_path.
        """
        file_termination = 'glove.6B.' + str(dimension) + 'd.txt'
        return os.path.join(PathsManagement.directory_glove, file_termination)

    @staticmethod
    def generate_glove_twitter_model_directory_path(dimension):
        """
        Similar to generate_saved_feature_path.
        """
        file_termination = 'glove.twitter.27B.' + str(dimension) + 'd.txt'
        return os.path.join(PathsManagement.directory_glove_twitter, file_termination)

    @staticmethod
    def generate_spanish_word_embeddings_model_directory_path():
        """
        Similar to generate_saved_feature_path.
        """
        return PathsManagement.file_spanish_word_emb_SBW_300

    @staticmethod
    def generate_auprotk_new_data_path(new_data_name):
        """
        Similar to generate_saved_feature_path.
        """
        file_termination = PathsManagement.util_names_separator('auprotk', new_data_name + '.tsv')
        path = os.path.join(PathsManagement.directory_AUPROTK_features, file_termination)
        return path

    @staticmethod
    def generate_feature_extractors_path(feature_extractor_name, dataset_name):
        """
        Similar to generate_saved_feature_path.
        """
        file_termination = PathsManagement.util_names_separator(feature_extractor_name, dataset_name)
        return os.path.join(PathsManagement.directory_features_extractors, file_termination)

    @staticmethod
    def generate_classification_model_path(experiment_name, classification_model_name):
        """
        Similar to generate_saved_feature_path.
        """
        experiment_classifiers_directory = os.path.join(PathsManagement.directory_classification_models, experiment_name)
        PathsManagement.verify_create_automatic_directory(experiment_classifiers_directory)
        file_termination = PathsManagement.util_names_separator(experiment_name, classification_model_name)
        return os.path.join(experiment_classifiers_directory, file_termination)

    @staticmethod
    def generate_save_result_path(experiment_name, classification_model_name, type_result):
        """
        Similar to generate_saved_feature_path.
        """
        file_termination = PathsManagement.util_names_separator(experiment_name, classification_model_name, type_result)
        return os.path.join(PathsManagement.directory_results, file_termination)

    @staticmethod
    def generate_path_new_data_classification(new_data_name, experiment_name):
        file_termination = PathsManagement.util_names_separator(new_data_name, experiment_name + '.tsv')
        return os.path.join(PathsManagement.directory_results, file_termination)

    @staticmethod
    def util_names_separator(*arg):
        """
        Receives a list of string and concatenates it using '_'.
        :param arg: The list of string to be concatenated.
        :return: the concatenated string.
        """
        return '_'.join(arg)

    @staticmethod
    def utils_print_result_to_file(result, file):
        """
        Redirects print to file.
        :param result:
        """
        filename = os.path.join(PathsManagement.directory_results, file + '.csv')
        if os.path.exists(filename):
            # read to dataframe
            file_data_frame = pd.read_csv(filename)

            # concatenate dataframe
            result = file_data_frame.append(result)

            # save dataframe
            result.to_csv(filename, sep=',', index=False, encoding='utf-8')
        else:
            result.to_csv(filename, sep=',', index=False, encoding='utf-8')

    @staticmethod
    def get_experiment_parameters_from_file(experiment_id):
        """
        Redirects print to file.
        :param result:
        """
        filename = os.path.join(PathsManagement.directory_results, 'Experiment_parameters.csv')
        if os.path.exists(filename):
            # read to dataframe
            experiment_parameters = pd.read_csv(filename)

            # find id of the experiment
            return experiment_parameters[experiment_parameters['experiment_name'].str.contains(experiment_id)]

    @staticmethod
    def verify_correct_directories_exist():
        # directories for results of the pipeline
        PathsManagement.verify_create_automatic_directory(PathsManagement.directory_results)
        PathsManagement.verify_create_automatic_directory(PathsManagement.directory_features_extractors)
        PathsManagement.verify_create_automatic_directory(PathsManagement.directory_saved_features)
        PathsManagement.verify_create_automatic_directory(PathsManagement.directory_classification_models)

        # directories needed to be created manually
        PathsManagement.verify_request_directory(PathsManagement.directory_original_data)
        PathsManagement.verify_request_directory(PathsManagement.directory_glove)
        PathsManagement.verify_request_directory(PathsManagement.directory_hateval_en)
        PathsManagement.verify_request_directory(PathsManagement.directory_hateval_es)







