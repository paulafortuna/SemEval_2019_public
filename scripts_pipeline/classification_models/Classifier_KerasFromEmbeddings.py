from abc import ABC

from sklearn.metrics import classification_report
from keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd

from scripts_pipeline.classification_models.Classifier_CvGridSearch import CvGridSearchClassifier
import numpy as np
from scripts_pipeline.feature_extraction.WordEmbeddingsGloveTwitter import WordEmbeddingsGloveTwitter
from scripts_pipeline.PathsManagement import PathsManagement as Paths
from keras.models import load_model


class ClassifierKerasFromEmbeddings(CvGridSearchClassifier, ABC):
    """
    Implements methods for deep learning classification with keras.
    It is a Child Class of CvGridSearchClassifier, and it can be instantiated because it implements
    apply_grid_search_cv_classifier method.
    """
    def __init__(self, classifier_name, is_learn_embeddings, loss_fun, optimizer, epochs, batch_size, initialize_weights_with):
        """
        Initializer of the class. Uses a dictionary to select the classifier to be used.
        :param dataset: the dataset to be used in the classification.
        :param exp: the experimetn object where some necessary parameters will be stored.
        :param classifier_name: the name of the classifier to be used. Should be selected from Var
        """
        CvGridSearchClassifier.__init__(self)
        self.classifier_name = classifier_name
        self.is_learn_embeddings = is_learn_embeddings
        self.loss_fun = loss_fun
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.initialize_weights_with = initialize_weights_with  # "random"
        self.feature_extractor = None
        self.embedding_dim = None
        self.tokenizer_name = None
        self.W = None
        self.classifier_func = None
        self.model_function = None

    def create_model(self):
        """
        Creates a deep learning model using keras and word embeddings
        Reading of the word embeddings weights from file, application of a classifier function, model function and returns the model.
        :return: returns the built model.
        """
        # embedding part
        self.feature_extractor = WordEmbeddingsGloveTwitter.load_object(self.exp.features_to_use[0], self.exp.dataset_name)
        self.embedding_dim = self.feature_extractor.parameters.embedding_dim
        self.tokenizer_name = self.feature_extractor.parameters.tokenizer_name

        self.W = self.feature_extractor.W

        # make the model
        model = self.model_function(self.dataset.x_train.shape[1])

        return model

    def train_classifier(self, model_name, dataset, exp):
        """
        Calls the specific function that builds the parameters for the grid search.
        Implementation of the abstract method from the parent class.
        :param model_name: the model_name to be used in the saving of the model to files.
        """
        self.dataset = dataset
        self.exp = exp
        self.deep_learning_search_batch_epoch_grid(model_name)

    def deep_learning_search_batch_epoch_grid(self, model_name):
        """
        Defines a set of parameters to be used in the classification.
        Invokes the parent class for CV and grid search classification with specified parameters.
        :param model_name: the model_name to be used in the saving of the model to files.
        """
        # Based on this
        # https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

        # defines the parameters values and parameter grid
        if self.exp.use_grid_search:
            batch_size = [16, 32, 64, 128]
            epochs = [10, 20]
        else:
            batch_size = [self.batch_size]
            epochs = [self.epochs]

        validation_split = [0.1]
        verbose = [0]

        param_grid = dict(batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=verbose)

        if self.exp.consider_class_weight:
            class_weight = self.compute_class_weight()
            param_grid['class_weight'] = [class_weight]

        # defines the model to be used - this method is specific from keras and deep learning
        model = KerasClassifier(build_fn=self.classifier_func)

        # invokes the parent class for CV and grid search classification with specified parameters
        self.cv_grid_search_scikit_classifier(model, param_grid, model_name)

    def compute_class_weight(self):
        """
        Computes the classes frequencies and puts it in a dictionary format.
        :return: Dictionary with the classes frequencies.
        """
        labels_dict = pd.DataFrame(self.dataset.y_train, columns=['col1'])
        return labels_dict['col1'].value_counts().to_dict()

    def train_new_model_best_parameters(self):
        """
        After checking in the grid search we extract again the best model and save it.
        :return: The new model extracted with the best parameters.
        """

        # get parameters of the best classifier
        params = self.model.best_params_
        batch_size = params['batch_size']
        epochs = params['epochs']
        if self.exp.consider_class_weight:
            class_weight = params['class_weight']
        else:
            class_weight = None

        # train the best model
        new_model = self.create_model()
        y_train = np.asarray(self.dataset.y_train)
        new_model.fit(self.dataset.x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      class_weight=class_weight,
                      validation_split=0.1,
                      verbose=0)
        return new_model

    def save_model(self, model_name):
        """
        Saves the new extracted model with the best parameters to files
        """

        new_model = self.train_new_model_best_parameters()
        # save model
        model_path = Paths.generate_classification_model_path(self.exp.experiment_name, model_name + '.h5')
        new_model.save(model_path)

    def load_model(self, experiment_name, model_name):
        """
        Loads the new extracted model with the best parameters from files
        """
        model_path = Paths.generate_classification_model_path(experiment_name, model_name + '.h5')
        self.model = load_model(model_path)

    def predict_test_data(self):
        """
        Predicts the test data using the model with the best parameters.
        """

        X_test = self.dataset.x_val
        y_test = self.dataset.y_val
        y_test = np.array(y_test)

        # generate prediction
        y_prob = self.model.predict(X_test)
        y_pred = y_prob.argmax(axis=-1)
        #y_prob = self.model.predict(X_test)
        #y_prob[y_prob >= 0.5] = 1
        #y_prob[y_prob < 0.5] = 0
        #y_pred = y_prob

        # generate classification report and print it to screen and file
        results = classification_report(y_test.ravel(), y_pred)
        results = self.test_results_toDataFrame(results)
        Paths.utils_print_result_to_file(results, "test_evaluation")

    def predict_new_data(self, new_data, new_data_name):
        """
        Predicts the new data using the model with the best parameters.
        :param new_data: file with the new data
        :param new_data_name: new data name to be used in the save of the file
        """
        # generate prediction
        # generate prediction
        #y_prob = self.model.predict(new_data.x_val)
        #y_prob[y_prob >= 0.5] = 1
        #y_prob[y_prob < 0.5] = 0
        #new_data.y_val = y_prob

        y_prob = self.model.predict(new_data.x_val)
        new_data.y_val = y_prob.argmax(axis=-1)
        return new_data
