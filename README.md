# SemEval_2019_public

This repository presents the code for the Stop PropagHate team on HatEval and OffensEval tasks of SemEval 2019. 
The detailed results and approach followed can be found in the report:
(to be published soon).

@inproceedings{fortunaSemEval, 
title={{Stop PropagHate at SemEval-2019 Tasks 5 and 6: Are abusive language classification results reproducible?}}, 
author={Fortuna, Paula and Soler-Company, Juan and Nunes Sérgio}, 
booktitle={Proceedings of The 13th International Workshop on Semantic Evaluation (SemEval)}, 
year={2019} 
} 


## Summary

This repository replicates one of the most relevant works on the state-of-the-art literature, using word embeddings and LSTM for hate speech automatic detection https://github.com/pinkeshbadjatiya/twitter-hatespeech, with an update for Python 3. After circumventing some of the problems of the original code, we tested it using the same data achieving worse results than what cited in the original paper. We found poor results when applying it to the HatEval contest. We think this is due mainly to inconsistencies in the data of this contest. Finally, for the OffensEval we believe that the classifier performed well, proving to have a better performance for offense detection than for hate speech.


## The framework

The framework here available provides features and classifiers specially used in this task. Regarding the features, we extract Glove Twitter word embeddings, sentiment  and  frequencies  of  words  from  Hate-base. The last is a set of features developed in ourwork. Regarding the classifiers we used LSTM, SVM and xg-Boost.


## Configuration
In the [configurations section](https://github.com/paulafortuna/SemEval_2019_public/tree/master/configurations) you can access on how to configure this project.


## The experiments

The exact experiments conducted for this paper are available here:
- replication [script](https://github.com/paulafortuna/SemEval_2019_public/blob/master/main_replication.py)
- hatEval A [script](https://github.com/paulafortuna/SemEval_2019_public/blob/master/main_hateval_a.py)
- offensEval A [script](https://github.com/paulafortuna/SemEval_2019_public/blob/master/main_offenseval_a.py)


## Class diagrams

We aimed at developing this project in a modular and extensible way. We used OOP and provide some documentation for it:
-  [class diagrams](https://docs.google.com/presentation/d/1t64DdCrN2avDvKocUBp-2kDHYl5dkT_2M8aRPLiw3u8/edit?usp=sharing)


## Documentation
We also provide an extense documentation helpful for future iterations of this project:
- doxygen

