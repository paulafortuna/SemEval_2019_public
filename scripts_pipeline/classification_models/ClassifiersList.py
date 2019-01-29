from scripts_pipeline.PathsManagement import Var
from scripts_pipeline.classification_models.Classifier_SVM import SVM
from scripts_pipeline.classification_models.Classifier_xgBoost import xgBoost
from scripts_pipeline.classification_models.Classifier_RandomForest import RandomForest
from scripts_pipeline.classification_models.Classifier_LogisticRegression import LogisticRegressionClassifier
from scripts_pipeline.classification_models.Classifier_LSTMFromEmbedding import ClassifierLSTM
from scripts_pipeline.classification_models.Classifier_LSTMFromEmbeddingsBeforeLastLayer import ClassifierLSTMBeforeLastLayer


class ClassifiersList:
    """
    Class where all the possible classifiers from the experiment are stored.
    It uses a dictionary for the building of classifiers.
    """

    def __init__(self):
        """
        Initializer of the class.
        """
        self.classifiers_dict = {}

    def build_dictionary(self, epochs, batch):
        """
        Method that builds the dictionary and returns it for usage in the experiments.
        :param epochs: Number of epochs to consider in the deep learning classifiers.
        :param batch: Batch size to consider in the deep learning classifiers.
        """
        # creates a dictionary of classifiers
        self.classifiers_dict[Var.linear_svm] = SVM()
        self.classifiers_dict[Var.CVgridSearchLSTM] = ClassifierLSTM(Var.CVgridSearchLSTM,
                                                                     False, 'binary_crossentropy', 'adam', epochs, batch,
                                                                     "glove")
        self.classifiers_dict[Var.LSTMFeatures] = ClassifierLSTMBeforeLastLayer(Var.LSTMFeatures, False,
                                                                                'binary_crossentropy', 'adam', epochs, batch, "glove")
        self.classifiers_dict[Var.xgBoost] = xgBoost()
        self.classifiers_dict[Var.LogisticRegressionClassifier] = LogisticRegressionClassifier()
        self.classifiers_dict[Var.RandomForest] = RandomForest()

        return self.classifiers_dict
