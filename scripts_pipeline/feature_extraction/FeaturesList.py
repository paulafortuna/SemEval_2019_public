from scripts_pipeline.PathsManagement import Var
from WordEmbeddingsGlove import WordEmbeddingsGloveParameters, WordEmbeddingsGlove
from scripts_pipeline.feature_extraction.Sentiment import SentimentVaderParameters, SentimentVader
from scripts_pipeline.feature_extraction.AuthorProfiling import AuthorProfilingParameters, AuthorProfiling
from scripts_pipeline.feature_extraction.Hatebase import HatebaseParameters, Hatebase
from ClassifierAsFeatures import ClassifierAsFeaturesParameters, ClassifierAsFeatures
from scripts_pipeline.feature_extraction.WordEmbeddingsGloveTwitter import WordEmbeddingsGloveTwitterParameters, WordEmbeddingsGloveTwitter


class FeaturesClassifierBuilder:
    """
    Class where all the possible feature extraction procedures from the experiment are stored.
    It uses a dictionary for the building of parameters and features.
    """

    def __init__(self):
        """
        Initializer of the class.
        """
        # prepare methods dictionaries
        self.features_parameters = {}
        self.features_objects = {}

    def build_dictionary(self, dataset_name):
        """
        Method that builds the dictionary and returns it for usage in the experiments.
        :param dataset_name: Name of the dataset to use in the experiment.
        """
        # creates a dictionary of parameters for feature extraction
        self.features_parameters[Var.sentiment_vader] = SentimentVaderParameters(Var.sentiment_vader, dataset_name)
        #self.features_parameters[Var.glove_w_emb_50] = WordEmbeddingsGloveParameters(20000, 50, 50, dataset_name)
        self.features_parameters[Var.author_profiling] = AuthorProfilingParameters(Var.author_profiling,
                                                                                   dataset_name)
        self.features_parameters[Var.hatebase] = HatebaseParameters(Var.hatebase, dataset_name, "eng")
        self.features_parameters[Var.glove_twitter_25_en] = WordEmbeddingsGloveTwitterParameters(
            Var.glove_twitter_25_en, dataset_name)
        self.features_parameters[Var.glove_twitter_200_en] = WordEmbeddingsGloveTwitterParameters(Var.glove_twitter_200_en, dataset_name)
        self.features_parameters[Var.glove_SBW_300_es] = WordEmbeddingsGloveTwitterParameters(
            Var.glove_SBW_300_es, dataset_name)

        # creates a dictionary of feature extraction
        self.features_objects[Var.sentiment_vader] = SentimentVader(self.features_parameters[Var.sentiment_vader])
        #self.features_objects[Var.glove_w_emb_50] = WordEmbeddingsGlove(self.features_parameters[Var.glove_w_emb_50])
        self.features_objects[Var.author_profiling] = AuthorProfiling(self.features_parameters[Var.author_profiling])
        self.features_objects[Var.hatebase] = Hatebase(self.features_parameters[Var.hatebase])
        self.features_objects[Var.glove_twitter_25_en] = WordEmbeddingsGloveTwitter(
            self.features_parameters[Var.glove_twitter_25_en])
        self.features_objects[Var.glove_twitter_200_en] = WordEmbeddingsGloveTwitter(self.features_parameters[Var.glove_twitter_200_en])
        self.features_objects[Var.glove_SBW_300_es] = WordEmbeddingsGloveTwitter(
            self.features_parameters[Var.glove_SBW_300_es])

        return self.features_parameters, self.features_objects
