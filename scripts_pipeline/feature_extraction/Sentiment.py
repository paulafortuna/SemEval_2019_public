
from scripts_pipeline.Dataset import Dataset
from scripts_pipeline.feature_extraction.FeatureExtraction import Parameters, FeatureExtraction
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class SentimentVaderParameters(Parameters):
    """
    Parameters to be used in the sentiment feature extraction.
    """
    def __init__(self, feature_name, dataset_name):
        Parameters.__init__(self, feature_name, dataset_name)


class SentimentVader(FeatureExtraction):
    """
    Implements methods for sentiment feature extraction with vader.
    It is a Child Class of FeatureExtraction, and it can be instantiated because it implements
    conduct_feature_extraction method.
    """
    def __init__(self, parameters):
        """
        Calls the parent class FeatureExtraction initiator.
        :param parameters: object of the SentimentVaderParameters.
        """
        FeatureExtraction.__init__(self, parameters)

    def conduct_feature_extraction_train_test(self, original_dataset):
        """
        Assures the feature extraction occurs in the right order of steps.
        Implementation of the abstract method from the parent class.
        :param original_dataset:the original data to be extracted
        """
        features_dataset = Dataset()
        features_dataset.set_ids_and_y(original_dataset)
        features_dataset.x_train = self.extract_sentiment(original_dataset.x_train)
        features_dataset.x_val = self.extract_sentiment(original_dataset.x_val)
        self.save_features_to_file(features_dataset)

    def conduct_feature_extraction_new_data(self, new_data, new_data_name, experiment_name):
        """
        Abstract method that allows all the feature extraction procedures to be used in a list and called
        with the same method.
        :param new_data: the new data for being classified
        """
        features_dataset = Dataset()
        features_dataset.x_val = self.extract_sentiment(new_data.x_val)
        features_dataset.x_val_id = new_data.x_val_id
        self.parameters.dataset_name = new_data_name
        self.save_features_to_file(features_dataset)

    def extract_sentiment(self, x_data):
        """
        Method that really implements the feature extraction procedure.
        It uses the vader sentiment from nltk.sentiment.vader imports SentimentIntensityAnalyzer.
        Extracts with the function polarity_scores the neg, neu, pos and compound dimensions of sentiment.
        :param original_dataset: texts columns
        :return: the extracted features
        """
        sid = SentimentIntensityAnalyzer()
        res = pd.DataFrame({
            'neg' : [sid.polarity_scores(instance)['neg'] for instance in x_data],
            'neu': [sid.polarity_scores(instance)['neu'] for instance in x_data],
            'pos': [sid.polarity_scores(instance)['pos'] for instance in x_data],
            'compound': [sid.polarity_scores(instance)['compound'] for instance in x_data],
        })
        return res