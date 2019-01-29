from scripts_pipeline.Dataset import Dataset
from scripts_pipeline.feature_extraction.FeatureExtraction import Parameters, FeatureExtraction
from scripts_pipeline.PathsManagement import PathsManagement as Paths, Var

import pickle

from keras.preprocessing.sequence import pad_sequences
import numpy as np
import gensim, sklearn
from collections import defaultdict
from string import punctuation
from gensim.parsing.preprocessing import STOPWORDS
import re

# from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier

# for solving a problem with macOS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class WordEmbeddingsGloveTwitterParameters(Parameters):
    """
    Class used to extract the word embeddings.
    The methods here are from the code of the paper "Deep Learning for Hate Speech Detection in Tweets"
    """
    def __init__(self, feature_name, dataset_name):
        Parameters.__init__(self, feature_name, dataset_name)
        self.tokenizer_name = "glove"
        self.embedding_dim = self.get_dim()
        self.word2vec_model = []

    def get_dim(self):
        if self.feature_name == Var.glove_twitter_25_en:
            return int(25)
        elif self.feature_name == Var.glove_twitter_200_en:
            return int(200)
        elif self.feature_name == Var.glove_SBW_300_es:
            return int(504)


class WordEmbeddingsGloveTwitter(FeatureExtraction):

    def __init__(self, parameters):
        """
        Calls the parent class FeatureExtraction initiator.
        :param parameters: object of the SentimentVaderParameters.
        """
        FeatureExtraction.__init__(self, parameters)
        self.tokenizer = self.glove_tokenize
        self.flags = re.MULTILINE | re.DOTALL
        self.reverse_vocab = {}
        self.vocab = {}
        self.freq = defaultdict(int)
        self.max_sequence_length = None
        self.W = None

    def hashtag(self, text):
        text = text.group()
        hashtag_body = text[1:]
        if hashtag_body.isupper():
            result = u"<hashtag> {} <allcaps>".format(hashtag_body)
        else:
            result = " ".join(["<hashtag>"] + re.split(u"([A-Z])", hashtag_body, flags=self.flags))
        return result

    def allcaps(self, text):
        text = text.group()
        return text.lower() + " <allcaps>"

    def tokenize(self, text):
        # Different regex parts for smiley faces
        eyes = r"[8:=;]"
        nose = r"['`\-]?"

        # function so code less repetitive
        def re_sub(pattern, repl):
            return re.sub(pattern, repl, text, flags=self.flags)

        text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
        text = re_sub(r"/", " / ")
        text = re_sub(r"@\w+", "<user>")
        text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
        text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
        text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
        text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
        text = re_sub(r"<3", "<heart>")
        text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
        text = re_sub(r"#\S+", self.hashtag)
        text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
        text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

        ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
        # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
        text = re_sub(r"([A-Z]){2,}", self.allcaps)

        return text.lower()

    def glove_tokenize(self, text):
        text = self.tokenize(text)
        text = ''.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        return words

    def select_tweets(self, tweets):
        # selects the tweets as in mean_glove_embedding method
        # Processing

        tweet_return = []
        for tweet in tweets:
            _emb = 0
            words = self.tokenizer(tweet.lower())
            for w in words:
                if w in self.parameters.word2vec_model:  # Check if embeeding there in GLove model
                    _emb += 1
            if _emb:  # Not a blank tweet
                tweet_return.append(tweet)
        print('Tweets selected:', len(tweet_return))
        return tweet_return

    def gen_vocab(self, tweets):
        # Processing
        vocab_index = 1
        for tweet in tweets:
            text = self.tokenizer(tweet.lower())
            text = ' '.join([c for c in text if c not in punctuation])
            words = text.split()
            words = [word for word in words if word not in STOPWORDS]

            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = vocab_index
                    self.reverse_vocab[vocab_index] = word       # generate reverse vocab as well
                    vocab_index += 1
                self.freq[word] += 1
                self.vocab['UNK'] = len(self.vocab)
                self.reverse_vocab[len(self.vocab)] = 'UNK'

    def gen_sequence(self, tweets, y_data, id_data):
        X, y, id = [], [], []
        i = 0
        for tweet in tweets:
            text = self.tokenizer(tweet.lower())
            text = ' '.join([c for c in text if c not in punctuation])
            words = text.split()
            words = [word for word in words if word not in STOPWORDS]
            seq, _emb = [], []
            for word in words:
                seq.append(self.vocab.get(word, self.vocab['UNK']))
            X.append(seq)
            y.append(y_data[i])
            id.append(id_data[i])
            i += 1
        return X, y, id

    def get_embedding_weights(self):
        embedding = np.zeros((len(self.vocab) + 1, self.parameters.embedding_dim))
        n = 0
        for k, v in self.vocab.items():
            try:
                embedding[v] = self.parameters.word2vec_model[k]
            except:
                n += 1
                pass
        print("%d embedding missed" % n)
        return embedding

    def conduct_feature_extraction_train_test(self, original_dataset):
        """
        Assures the feature extraction occurs in the right order of steps.
        Implementation of the abstract method from the parent class.
        :param original_dataset:the original data to be extracted
        """
        path_glove = self.get_path()
        self.parameters.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(path_glove)
        features_dataset = Dataset()
        features_dataset.x_train, features_dataset.y_train, features_dataset.x_train_id  = self.extract_word_embeddings_glove_twitter_train(original_dataset)
        features_dataset.x_val, features_dataset.y_val, features_dataset.x_val_id = self.conduct_feature_extraction_test_data(original_dataset)
        self.save_features_to_file(features_dataset)
        self.save_object()

    def extract_word_embeddings_glove_twitter_train(self, dataset):
        """
        Method that really implements the feature extraction procedure.
        It uses the vader sentiment from nltk.sentiment.vader imports SentimentIntensityAnalyzer.
        Extracts with the function polarity_scores the neg, neu, pos and compound dimensions of sentiment.
        :param original_dataset: texts columns
        :return: the extracted features
        """
        train_tweets = self.select_tweets(dataset.x_train)
        self.gen_vocab(train_tweets)
        X, y, id_data = self.gen_sequence(train_tweets, dataset.y_train, dataset.x_train_id)
        self.max_sequence_length = max(map(lambda x: len(x), X))
        data = pad_sequences(X, maxlen=self.max_sequence_length)
        y = np.array(y)
        id_data = np.array(id_data)
        data, y, id_data = sklearn.utils.shuffle(data, y, id_data)
        self.W = self.get_embedding_weights()
        return data, y, id_data

    def conduct_feature_extraction_test_data(self, dataset):
        test_tweets = self.select_tweets(dataset.x_val)
        x_test, y_test, id_test = self.gen_sequence(test_tweets, dataset.y_val, dataset.x_val_id)
        x_test = pad_sequences(x_test, maxlen=self.max_sequence_length)
        return x_test, y_test, id_test

    def conduct_feature_extraction_new_data(self, new_data, new_data_name, experiment_name):

        feature_extractor = WordEmbeddingsGloveTwitter.load_object(self.parameters.feature_name, self.parameters.dataset_name)
        new_data.y_val = new_data.get_array_ids(new_data.x_val_id.size)
        new_data.x_val, y, new_data.x_val_id = feature_extractor.conduct_feature_extraction_test_data(new_data)
        new_data.y_val = None
        self.parameters.dataset_name = new_data_name
        self.save_features_to_file(new_data)

    def save_object(self):
        """
        Saves feature extraction object. It is needed later for classification with the frozen layer.
        """
        f = open(
            Paths.generate_feature_extractors_path(self.parameters.feature_name, self.parameters.dataset_name) + ".obj",
            'wb')
        pickle.dump(self, f)

    @staticmethod
    def load_object(feature_name, dataset_name):
        """
        Static method allows the call and load of the object from the Class, without having an object instantiated.
        """
        f = open(Paths.generate_feature_extractors_path(feature_name, dataset_name) + ".obj", 'rb')
        obj = pickle.load(f)
        return obj

    def get_path(self):
        if self.parameters.feature_name == Var.glove_twitter_25_en:
            return Paths.generate_glove_twitter_model_directory_path(self.parameters.embedding_dim)
        elif self.parameters.feature_name == Var.glove_twitter_200_en:
            return Paths.generate_glove_twitter_model_directory_path(self.parameters.embedding_dim)
        elif self.parameters.feature_name == Var.glove_SBW_300_es:
            return Paths.generate_spanish_word_embeddings_model_directory_path()



