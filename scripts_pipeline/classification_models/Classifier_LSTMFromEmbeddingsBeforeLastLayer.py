from keras.models import Model
from scripts_pipeline.Dataset import Dataset
from scripts_pipeline.PathsManagement import PathsManagement as Paths
from keras.models import load_model
from scripts_pipeline.classification_models.Classifier_LSTMFromEmbedding import ClassifierLSTM


class ClassifierLSTMBeforeLastLayer(ClassifierLSTM):
    """
    Descends from ClassifierLSTM class and extracts the final layer before binary classification.
    """
    def __init__(self, classifier_name, is_learn_embeddings, loss_fun, optimizer, epochs, batch_size, initialize_weights_with):
        """
        Initializer of the class.
        :param classifier_name: the name of the classifier
        :param is_learn_embeddings: boolean indicating if the embeddings should be learnt or not.
        :param loss_fun: the loss function to be used
        :param optimizer: the optimizer to be used
        :param epochs: the number of epochs to be used
        :param batch_size: the batch size to be used
        :param initialize_weights_with: if the weights of the model should be used with "glove" or "random"
        """
        ClassifierLSTM.__init__(self, classifier_name, is_learn_embeddings, loss_fun,
                                               optimizer, epochs, batch_size, initialize_weights_with)
        self.features_dataset = None

    def train_classifier(self, model_name, dataset, exp):
        """
        Overrids the parent train_classifier so that the last layer before classification can be extracted.
        Saves the weights as if a feature selection procedure had occurred.
        :param model_name: the name of the classifier
        :param dataset: the dataset to be used in the classification.
        :param exp: the experiment originating this classification
        """

        # grid search
        super().train_classifier(model_name, dataset, exp)

        # train of the best model from the grid search
        model = super().train_new_model_best_parameters()

        # get the final layer before classification by position
        self.model = Model(inputs=model.input, outputs=model.layers[-3].output)
        self.model.compile(loss=self.loss_fun, optimizer=self.optimizer, metrics=['accuracy'])

        # the dataset to be used
        self.features_dataset = Dataset()
        self.features_dataset.set_ids_and_y(self.dataset)

        # saving of the results of this classifier as feature extraction procedure
        self.features_dataset.x_train = self.model.predict(self.dataset.x_train)
        self.predict_test_data()
        self.features_dataset.save_features(model_name, exp.dataset_name)
        print(self.features_dataset.x_train)

    def save_model(self, model_name):
        """
        Save model in files.
        :param model_name: the name of the classifier
        """
        # save model
        model_path = Paths.generate_classification_model_path(self.exp.experiment_name, model_name + '.h5')
        self.model.save(model_path)

    def load_model(self, experiment_name, model_name):
        """
        Load the model from files
        :param experiment_name: the name of the experiment generating this classifier.
        :param model_name: the name of the classifier
        """
        model_path = Paths.generate_classification_model_path(experiment_name, model_name + '.h5')
        self.model = load_model(model_path)

    def predict_test_data(self):
        """
        Applies the stored classifier to the test data.
        :param model_name: The name of the model to save.
        """
        self.features_dataset.x_val = self.model.predict(self.dataset.x_val)
        self.features_dataset.save_features(self.exp.experiment_name, self.exp.dataset_name)

    def predict_new_data(self, new_data, new_data_name):
        """
        Applies the stored classifier to new data.
        :param new_data: The new data to classify.
        :param new_data_name: The name of the data to save.
        """
        new_data.x_val = self.model.predict(new_data.x_val)
        new_data.save_features(self.classifier_name, new_data_name)