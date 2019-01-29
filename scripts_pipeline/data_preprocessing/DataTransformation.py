import pickle
from scripts_pipeline.PathsManagement import Var, PathsManagement as Paths
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

class DataTransformation:
    """
    Allows the access to a set of functions for data transformation after feature extraction and reading from files.
    The available transformations are in the Var class. One of these should be decided and passed to the Experiment.
    it is also possible to don't conduct any data transformation.
    The available transformations from are sklearn.preprocessing:
    'MinMax' (MinMaxScaler) - rescale data between 0 and 1
    'NormalizeGaussian' (StandardScaler) - Standardize Data to a standard Gaussian distribution with a mean of 0 and a standard deviation of 1
    'NormalizeLinearAlgebra' - Normalize rescaling each observation (row) to have a length of 1 (called a unit norm in linear algebra)
    """

    def __init__(self, type_transformation, x):
        """
        Initializer that defines a dictionary with all the types of transformation available for being called.
        :param type_transformation: a string with the desired data transformation (available in the Var class)
        :param x: x data to transform
        """
        self.data_transformation = {}
        self.data_transformation[Var.min_max] = MinMaxScaler(feature_range=(0, 1))
        self.data_transformation[Var.normalize_gaussian] = StandardScaler()
        self.data_transformation[Var.normalize_linear_algebra] = Normalizer()
        self.scaler = self.data_transformation.get(type_transformation).fit(x)

    def apply_transformation(self, X):
        if X == []:
            return X
        return self.scaler.transform(X)

    def apply_transformation_all_features(self, all_features):
        all_features.x_train = self.apply_transformation(all_features.x_train)
        all_features.x_val = self.apply_transformation(all_features.x_val)
        return all_features

    #save features to file
    def save_transformation_to_file(self):
        # save self.type_transformation
        f = open(Paths.save_obj_word_embeddings_glove + ".obj", 'wb')
        pickle.dump(self, f)

    @staticmethod
    def load_transformation_from_file(self):
        # save self.type_transformation
        f = open(Paths.save_obj_word_embeddings_glove + ".obj", 'rb')
        obj = pickle.load(f)
        return obj





