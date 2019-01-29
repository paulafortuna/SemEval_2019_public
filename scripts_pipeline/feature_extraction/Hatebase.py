
from scripts_pipeline.Dataset import Dataset
from scripts_pipeline.feature_extraction.FeatureExtraction import Parameters, FeatureExtraction
from scripts_pipeline.PathsManagement import PathsManagement as Paths
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import re



class HatebaseParameters(Parameters):
    """
    Parameters to be used in the Hatebase feature extraction.
    """
    def __init__(self, feature_name, dataset_name, language):
        Parameters.__init__(self, feature_name, dataset_name)
        self.language = language


class Hatebase(FeatureExtraction):
    """
    Implements methods for Hatebase feature extraction.
    It is a Child Class of FeatureExtraction, and it can be instantiated because it implements
    conduct_feature_extraction method.
    """
    def __init__(self, parameters):
        """
        Calls the parent class FeatureExtraction initiator.
        :param parameters: object of the HatebaseParameters.
        """
        FeatureExtraction.__init__(self, parameters)
        self.hateful_words = []
        self.hate_topic_words = []

    def conduct_feature_extraction_train_test(self, original_dataset):
        """
        Assures the feature extraction occurs in the right order of steps.
        Implementation of the abstract method from the parent class.
        :param original_dataset:the original data to be extracted
        """
        self.hateful_words = self.read_list_hateful_words()
        self.hate_topic_words = self.read_list_hate_topic_words()
        features_dataset = Dataset()
        features_dataset.set_ids_and_y(original_dataset)
        features_dataset.x_train = self.extract_hate_words_and_topic(original_dataset.x_train)
        features_dataset.x_val = self.extract_hate_words_and_topic(original_dataset.x_val)
        self.save_features_to_file(features_dataset)

    def conduct_feature_extraction_new_data(self, new_data, new_data_name, experiment_name):
        """
        Abstract method that allows all the feature extraction procedures to be used in a list and called
        with the same method.
        :param new_data: the new data for being classified
        """
        features_dataset = Dataset()
        features_dataset.x_val = self.extract_hate_words_and_topic(new_data.x_val)
        features_dataset.x_val_id = new_data.x_val_id
        self.parameters.dataset_name = new_data_name
        self.save_features_to_file(features_dataset)

    def extract_hate_words_and_topic(self, x_data):
        """
        Method that really implements the feature extraction procedure.
        It counts the hatebase words in every message.
        Two types of words are used: the hateful_words given by hate base. Another set of list is counted.
        This is the hate_topic_words and corresponds to the words used in hatebase to describe each word without being offensive.
        :param x_data: texts columns
        :return: the extracted features
        """
        res = pd.DataFrame({
            'hateful_words': [self.count_frequencies(instance, self.hateful_words) for instance in x_data],
            'hate_topic_words': [self.count_frequencies(instance, self.hate_topic_words) for instance in x_data]
        })
        return res

    def count_frequencies(self, instance, words):
        """
        Method that really implements the feature extraction procedure.
        and counts a set of words in a message.
        :param instance: string with the text of the message
        :param words: words to chack if they are present in the instance
        :return: the total of counts from words found in the instance
        """
        total = 0
        for word in words:
            if ' ' in word and word.lower() in instance.lower():
                    total += 1
            else:
                if word.lower() in instance.lower().split():
                    total += 1

        return total

    def read_list_hateful_words(self):
        """
        Method that reads from files the list of hateful words.
        :return: the list of words
        """
        hatebase = pd.read_csv(Paths.file_hatebase)
        hatebase = hatebase.loc[hatebase['language'] == self.parameters.language]
        s = hatebase['term']
        p = hatebase.loc[hatebase['plural_of'].notnull()]['plural_of']
        words = s.append(p)

        stop_words = set(stopwords.words("english"))
        filtered_words = list(filter(lambda word: word.lower() not in stop_words, words))

        return list(set(filtered_words))

    def read_list_hate_topic_words(self):
        """
        Method that reads from files the list of hateful related words.
        :return: the list of words
        """
        hatebase = pd.read_csv(Paths.file_hatebase)
        hatebase = hatebase.loc[hatebase['language'] == self.parameters.language]
        words = hatebase['hateful_meaning']
        words = ' '.join(words)
        words = re.sub(r'\b\w{1,1}\b', '', words) # remove words with 1 digit
        words = re.sub(r'\w*\d\w*', '', words).strip() # remove words with digits

        #TODO generalize for other languages
        stop_words = set(stopwords.words("english"))
        words = words.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        word_tokens = tokenizer.tokenize(words)

        filtered_words = list(filter(lambda word: word not in stop_words, word_tokens))
        return list(set(filtered_words))
