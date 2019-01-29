
### Prepare imports
from scripts_pipeline.PathsManagement import Var, PathsManagement as Paths
from scripts_pipeline.Experiment import Experiment
from scripts_pipeline.ClassifyNewData import ClassifyNewContestDataAfterExperiment
import itertools


#from scripts_pipeline.feature_extraction.WordEmbeddingsGloveTwitterLSTM import main_LSTM
#Paths('/Users/paulafortuna/PycharmProjects/HatEval')

#main_LSTM()


#########
# define directory for HatEval project: the only one that should be changed
#directory_project = '/Users/paulafortuna/PycharmProjects/HatEval'  # directory for HatEval project: the only one that should be changed
#directory_project = '/home/paulatfortuna/HatEval'  # directory server

#########
# some preliminary tests

##### experiment for extracting features from both train and test #####

#### test

FOLDS = 10
new_data_path = "/home/paulatfortuna/HatEval/original_datasets/public_development_en/test_en.tsv"
def combinations_any_length(l):
    comb = []
    for i in range(len(l)):
        comb += itertools.combinations(l,i+1)
    return comb

def experiment_name(num_experiment, dataset_name, features, classifier, normalize_data, class_weight, use_grid_search, epochs, batch):
    return "experiment" + str(num_experiment) + Paths.util_names_separator(dataset_name, '_'.join(features), classifier,
                                                                    "normalize" + normalize_data,
                                                                    "weight" + str(class_weight),
                                                                    "use_grid_search" + str(use_grid_search),
                                                                    "epochs" + str(epochs),
                                                                    "batch" + str(batch))

##############
# Replication experiment
#############

experiment_id = "extract_LSTMFeatures"


exp = Experiment(experiment_name=experiment_id,
                 dataset_name="hateval_en", # hateval_es hateval_en test hateval_en_my_division zeerak
                 apply_data_preprocessing=[],
                 features_to_extract=[],  # Var.sentiment_vader, Var.hatebase, Var.glove_twitter_25_en[]
                 features_to_use=[Var.glove_twitter_200_en],
                 normalize_data=Var.none,  # Var.min_max, Var.normalize_gaussian , Var.normalize_linear_algebra
                 classifier_name=Var.LSTMFeatures,  # Var.LSTMFeatures Var.linear_svm, Var.CVgridSearchLSTM, Var.xgBoost, Var.LogisticRegressionClassifier, Var.RandomForest
                 consider_class_weight=False,
                 folds_cross_validation=FOLDS,
                 use_grid_search=False,
                 epochs=10,
                 batch=128
                 )
exp.start_experiment()


experiment_id = "extract_other"

exp = Experiment(experiment_name=experiment_id,
                 dataset_name="hateval_en", # hateval_es hateval_en test hateval_en_my_division zeerak
                 apply_data_preprocessing=[],
                 features_to_extract=[Var.sentiment_vader, Var.hatebase],  # Var.sentiment_vader, Var.hatebase, Var.glove_twitter_25_en[]
                 features_to_use=[Var.sentiment_vader, Var.hatebase],
                 normalize_data=Var.none,  # Var.min_max, Var.normalize_gaussian , Var.normalize_linear_algebra
                 classifier_name=Var.xgBoost,  # Var.LSTMFeatures Var.linear_svm, Var.CVgridSearchLSTM, Var.xgBoost, Var.LogisticRegressionClassifier, Var.RandomForest
                 consider_class_weight=True,
                 folds_cross_validation=FOLDS,
                 use_grid_search=False,
                 epochs=10,
                 batch=128
                 )
exp.start_experiment()


def make_experiments():
    num_experiment = 0
    datasets_to_use = ["hateval_en"]
    for dataset_name in datasets_to_use:
        print(dataset_name)
        if num_experiment <20000:
            for features in combinations_any_length([Var.LSTMFeatures, Var.sentiment_vader, Var.hatebase]):
                for classifier in [Var.xgBoost]:
                    for normalize_data in [Var.none]:
                        for class_weight in [False, True]:
                            for use_grid_search in [False]:
                                num_experiment += 1
                                exp_name = experiment_name(num_experiment, dataset_name, features, classifier, normalize_data,
                                                           class_weight, use_grid_search)
                                print(exp_name)
                                exp = Experiment(experiment_name=exp_name,
                                                 dataset_name=dataset_name,
                                                 apply_data_preprocessing=[],
                                                 features_to_extract=[],
                                                 features_to_use=features,
                                                 normalize_data=normalize_data,
                                                 classifier_name=classifier,
                                                 consider_class_weight=class_weight,
                                                 folds_cross_validation=FOLDS,
                                                 use_grid_search=use_grid_search,
                                                 epochs=10,
                                                 batch=128
                                                 )
                                exp.start_experiment()
                                experiment_id = exp_name

                                classify_new_data = ClassifyNewContestDataAfterExperiment(experiment_id)
                                classify_new_data.start_classification(new_data_path, "test_new_data", True, "id")


make_experiments()
