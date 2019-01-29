
from scripts_pipeline.PathsManagement import PathsManagement as Paths

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import pickle

class Dataset:
    """
    Provides in a single structure all the dataset elements necessary for a classification pipeline.
    It contains x and y, for both training and testing.
    Contains also methods for reading and writing these structures from files.
    """

    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.labels_index = None
        self.x_train_id = None
        self.x_val_id = None

    def set_ids_and_y(self, old_dataset):
        """
        Sets the fields id and y using for that the fields of another dataset passed as argument
        :param old_dataset: The dataset to use with the ids and y to be set in the self.
        """
        self.y_train = old_dataset.y_train
        self.y_val = old_dataset.y_val
        self.x_train_id = old_dataset.x_train_id
        self.x_val_id = old_dataset.x_val_id

    def read_original_data(self, dataset_name):
        """
        Reads a dataset from file. The method uses a bit copy paste of code, instead of encapsulation,
        This is because reading a dataset is usually very dependant of the specific case,
        :param dataset_name: Name of the dataset to be loaded. The dataset should contain train and test in separated files.
        """
        if(dataset_name == "hateval_en"):
            # read from file and extract the correct columns
            dataset_hs = pd.read_csv(Paths.hateval_en_text_data_train, delimiter='\t', encoding='utf-8')
            self.x_train = dataset_hs["text"].str.lower()
            self.y_train = dataset_hs[["HS"]].values
            self.x_train_id = dataset_hs[["id"]].values[:,0]

            # read from file and extract the correct columns
            dataset_hs = pd.read_csv(Paths.hateval_en_text_data_test, delimiter='\t', encoding='utf-8')
            self.x_val = dataset_hs["text"].str.lower()
            self.y_val = dataset_hs[["HS"]].values
            self.x_val_id = dataset_hs[["id"]].values[:,0]

            # makes a dictionary of the classes
            self.labels_index = {"0": 0}
            self.labels_index["1"] = 1
        elif (dataset_name == "offenseval"):
            # read from file and extract the correct columns
            def label_offensive(row):
                if row['subtask_a'] == 'OFF':
                    return 1
                elif row['subtask_a'] != 'OFF':
                    return 0
            dataset_hs = pd.read_csv(Paths.offenseval_en_text_data_train, delimiter='\t', encoding='utf-8')
            self.x_train = dataset_hs["tweet"].str.lower()
            self.y_train = dataset_hs.apply(lambda row: label_offensive(row), axis=1)
            self.x_train_id = dataset_hs[["id"]].values[:, 0]

            # read from file and extract the correct columns
            dataset_hs = pd.read_csv(Paths.offenseval_en_text_data_test, delimiter='\t', encoding='utf-8')
            self.x_val = dataset_hs["tweet"].str.lower()
            self.y_val = dataset_hs.apply(lambda row: label_offensive(row), axis=1)
            self.x_val_id = self.get_array_ids(self.x_val.size)

            # makes a dictionary of the classes
            self.labels_index = {"0": 0}
            self.labels_index["1"] = 1
        elif (dataset_name == "hateval_es"):
            # read from file and extract the correct columns
            dataset_hs = pd.read_csv(Paths.hateval_es_text_data_train, delimiter='\t', encoding='utf-8')
            self.x_train = dataset_hs["text"].str.lower()
            self.y_train = dataset_hs[["HS"]].values
            self.x_train_id = dataset_hs[["id"]].values[:,0]

            # read from file and extract the correct columns
            dataset_hs = pd.read_csv(Paths.hateval_es_text_data_test, delimiter='\t', encoding='utf-8')
            self.x_val = dataset_hs["text"].str.lower()
            self.y_val = dataset_hs[["HS"]].values
            self.x_val_id = dataset_hs[["id"]].values[:,0]

            # makes a dictionary of the classes
            self.labels_index = {"0": 0}
            self.labels_index["1"] = 1
        elif (dataset_name == "zeerak"):
            dataset_hs = pd.read_csv(Paths.test_dataset, delimiter=',', encoding='utf-8')
            text = dataset_hs["text"].str.lower()

            y_values = dataset_hs[["Class"]].values
            lb = preprocessing.LabelBinarizer()
            y_values = lb.fit_transform(y_values)
            y_values = y_values[:16907, 1:2] + y_values[:16907, 2:3]

            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(text, y_values, test_size = 0.1, random_state = 42, stratify=y_values)

            self.x_train_id = self.get_array_ids(self.x_train.size)
            self.x_val_id = self.get_array_ids(self.x_val.size)

            # makes a dictionary of the classes
            self.labels_index = {"0": 0}
            self.labels_index["1"] = 1

        elif (dataset_name == "test"):
            dataset_hs = pd.read_csv(Paths.test_dataset, delimiter=',', encoding='utf-8')
            text = dataset_hs["text"].str.lower()

            y_values = dataset_hs[["Class"]].values
            lb = preprocessing.LabelBinarizer()
            y_values = lb.fit_transform(y_values)
            y_values = y_values[:16907, 1:2] + y_values[:16907, 2:3]

            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(text, y_values, test_size = 0.1, random_state = 42, stratify=y_values)

            self.x_train_id = self.get_array_ids(self.x_train.size)
            self.x_val_id = self.get_array_ids(self.x_val.size)

            self.x_train = self.x_train[0:1000]
            self.y_train = self.y_train[0:1000]
            self.x_val = self.x_val[:100]
            self.y_val = self.y_val[:100]
            self.x_train_id = self.x_train_id[0:1000]
            self.x_val_id = self.x_val_id[0:100]

            # makes a dictionary of the classes
            self.labels_index = {"0": 0}
            self.labels_index["1"] = 1
        elif (dataset_name == "hateval_en_my_division"):
            # read from file and extract the correct columns
            dataset_hs_train = pd.read_csv(Paths.hateval_en_text_data_train, delimiter='\t', encoding='utf-8')
            dataset_hs_test = pd.read_csv(Paths.hateval_en_text_data_test, delimiter='\t', encoding='utf-8')

            # glue both
            dataset_hs = pd.concat([dataset_hs_train,dataset_hs_test], axis=0)

            # extract the correct columns

            text = dataset_hs["text"].str.lower()
            y_values = dataset_hs[["HS"]].values[:,0]

            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(text, y_values, test_size = 0.1, random_state = 42, stratify=y_values)

            # makes a dictionary of the classes
            self.labels_index = {"0": 0}
            self.labels_index["1"] = 1

            self.x_train_id = self.get_array_ids(self.x_train.size)
            self.x_val_id = self.get_array_ids(self.x_val.size)

        else:
            print("The specified dataset does not exist.")

    def get_array_ids(self, num):
        """
        Builds array with num integer different ids, starting from 0.
        :param num: the number of ids to be generater.
        """
        return np.arange(num)

    def save_features(self, feature_name, dataset_name):
        """
        Saves the dataset object to files.
        Useful for using mainly in the feature extraction procedure.
        :param feature_name: name of the features that were extracted.
        :param dataset_name: name of the original dataset
        """
        path = Paths.generate_saved_feature_path(feature_name, dataset_name, "")
        f = open(path + ".obj",'wb')
        pickle.dump(self, f)

    def print_prediction_to_file(self, new_data_name, experiment_name, ids):
        """
        Saves the prediction of new data to files.
        :param new_data_name: name of the new data to be used in the filename.
        :param experiment_name: name of the original experiment to be used in the filename.
        :param ids boolean referencing if the ids should be included in the file to save as a column.
        """
        if ids:
            all_data = pd.DataFrame(data=self.y_val, index=self.x_val_id)
        else:
            all_data = pd.DataFrame(data=self.y_val)

        path = Paths.generate_path_new_data_classification(new_data_name, experiment_name)
        all_data.to_csv(path, sep='\t')

    def read_add_features(self, feature_name, dataset_name):
        """
        Reads the dataset from files to a dataset object in the feature reading procedure.
        It can be used when the dataset already contains other features.
        In this case the new features will be concatenated columnwise.
        :param feature_name: name of the features that were extracted.
        :param dataset_name: name of the original dataset
        """
        path = Paths.generate_saved_feature_path(feature_name, dataset_name, "")
        f = open(path + ".obj", 'rb')
        obj = pickle.load(f)

        # if the dataset is empty
        if self.x_train is None:
            self.x_train = obj.x_train
            self.x_val = obj.x_val
            self.y_train = obj.y_train
            self.y_val = obj.y_val
            self.x_train_id = obj.x_train_id
            self.x_val_id = obj.x_val_id
        else:
            # for the existing dataset concatenate in the same structure id and x_train
            all_data_x_train = pd.DataFrame(data=self.x_train, index=self.x_train_id)
            all_data_x_val = pd.DataFrame(data=self.x_val, index=self.x_val_id)
            new_data_x_train = pd.DataFrame(data=obj.x_train, index=obj.x_train_id)
            new_data_x_val = pd.DataFrame(data=obj.x_val, index=obj.x_val_id)

            new_data_x_train = new_data_x_train.add_suffix(feature_name)
            new_data_x_val = new_data_x_val.add_suffix(feature_name)

            all_data_x_train = all_data_x_train.join(new_data_x_train, how='inner')
            all_data_x_val = all_data_x_val.join(new_data_x_val, how='inner')

            self.x_train = all_data_x_train.values
            self.x_val = all_data_x_val.values

    def read_add_features_new_data(self, feature_name, new_data_name):
        """
        Reads the dataset from files to a dataset object in the feature reading procedure of new data.
        It can be used when the dataset already contains other features.
        In this case the new features will be concatenated columnwise.
        :param feature_name: name of the features that were extracted.
        :param dataset_name: name of the original dataset
        """
        path = Paths.generate_saved_feature_path(feature_name, new_data_name, "")
        f = open(path + ".obj", 'rb')
        obj = pickle.load(f)

        # if the dataset is empty
        if self.x_val is None:
            self.x_val = obj.x_val
            self.y_val = obj.y_val
            self.x_val_id = obj.x_val_id
        else:
            # for the existing dataset concatenate in the same structure id and x_train
            all_data_x_val = pd.DataFrame(data=self.x_val, index=self.x_val_id)
            new_data_x_val = pd.DataFrame(data=obj.x_val, index=obj.x_val_id)
            new_data_x_val = new_data_x_val.add_suffix(feature_name)
            all_data_x_val = all_data_x_val.join(new_data_x_val, how='inner')
            self.x_val = all_data_x_val.values

    def read_author_profiling_features(self):
        """
        Reads the x elements (train and test) from fthe author profiling set of features to a dataset object.
        These features are initially extracted for the AUPROTK and require an additional step of conversion.
        """
        # read from file and extract the correct columns
        data_train = pd.read_csv(Paths.generate_auprotk_new_data_path('train'), delimiter='\t', encoding='utf-8')
        self.x_train = data_train["features"].str.lower()

        # read from file and extract the correct columns
        data_test = pd.read_csv(Paths.generate_auprotk_new_data_path('test'), delimiter='\t', encoding='utf-8')
        self.x_val = data_test["features"].str.lower()

    def read_author_profiling_features_path(self, new_data_name):
        """
        Reads the x elements (train and test) from fthe author profiling set of features to a dataset object.
        These features are initially extracted for the AUPROTK and require an additional step of conversion.
        """
        # read from file and extract the correct columns
        path = Paths.generate_auprotk_new_data_path(new_data_name)
        data_train = pd.read_csv(path, delimiter='\t', encoding='utf-8')
        self.x_val = data_train["features"].str.lower()

    def get_new_data_to_classify(self, new_data_path, include_ids):
        # read from file and extract the correct columns
        """
        Reads the new data from files.
        :param new_data_path: path where to read new data.
        :param include_ids: boolean indicating if the data contains a column id. if not a false id is generated.
        """
        data_test = pd.read_csv(new_data_path, delimiter='\t', encoding='utf-8')
        try:
            self.x_val = data_test["text"].str.lower()
        except:
            self.x_val = data_test["tweet"].str.lower()

        if include_ids:
            self.x_val_id = data_test["id"]
        else:
            self.x_val_id = self.get_array_ids(self.x_val.size)
