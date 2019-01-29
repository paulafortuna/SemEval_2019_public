from scripts_pipeline.classification_models.Classifier_KerasFromEmbeddings import ClassifierKerasFromEmbeddings
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Embedding


class ClassifierLSTM(ClassifierKerasFromEmbeddings):
    """
    Implements LSTM for deep learning classification with keras.
    Extracted from the paper "Deep Learning for Hate Speech Detection in Tweets".
    It is a Child Class of ClassifierKerasFromEmbeddings.
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
        ClassifierKerasFromEmbeddings.__init__(self, classifier_name, is_learn_embeddings, loss_fun, optimizer, epochs, batch_size, initialize_weights_with)
        self.model_function = self.lstm_model
        self.classifier_func = self.create_model

    def lstm_model(self, sequence_length):
        """
        The classifier contruction and model compilation
        :param sequence_length: the sequence length used as input in the model.
        """
        model_variation = 'LSTM'
        print('Model variation is %s' % model_variation)
        model = Sequential()
        model.add(Embedding(len(self.feature_extractor.vocab)+1, self.embedding_dim, input_length=sequence_length, trainable=self.is_learn_embeddings, dtype='float32'))
        model.add(Dropout(0.25))
        model.add(LSTM(50))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss=self.loss_fun, optimizer=self.optimizer, metrics=['accuracy'])

        print(model.summary())

        if self.initialize_weights_with == "glove":
            model.layers[0].set_weights([self.W])
        elif self.initialize_weights_with == "random":
            self.shuffle_weights(model)

        return model