from typing import Any

from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, RandomForestClassifier)
from sklearn.linear_model import (LogisticRegression,
                                  PassiveAggressiveClassifier, Perceptron,
                                  RidgeClassifier, RidgeClassifierCV,
                                  SGDClassifier)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier


# some hyperparameters were removed since they would either never converge on our data, take too long
# or did not make sense. 
# For example, we do not use l1 loss, since it leads to sparse solutions, and
# it might not work with a combination of the other parameters.
# Similar cases also apply to other hyperparameters.
# For LinearDiscriminantAnalysis we also removed the "eigen" computation, 
# since it would not converge most of the time
# Same with "penalty" different than l2 for LogisticRegression: sometimes, it 
# would fail due to combination of parameters.

CLASSIFIERS_HYPERPARAMETER_LIST: dict[ClassifierMixin, dict[str, Any]] = {
    RandomForestClassifier: {
        "n_estimators": [10, 50, 100, 200, 500],
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 4, 8, 16],
        "max_features": ["sqrt", "log2", None],
    },
    LGBMClassifier: {
        "n_estimators": [100, 200, 300, 400, 500],
        "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.5],
        "max_depth": [-1, 10, 20, 30, 50],
        "min_child_samples": [20, 50, 100, 200, 300],
        "subsample": [0.7, 0.8, 0.9, 1.0],
    },
    ExtraTreesClassifier: {
        "n_estimators": [10, 50, 100, 200, 500],
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 4, 8, 16],
        "max_features": ["sqrt", "log2", None],
    },
    XGBClassifier: {
        "n_estimators": [100, 200, 300, 400, 500],
        "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.5],
        "max_depth": [3, 4, 5, 6, 7],
        "min_child_weight": [1, 2, 3, 4, 5],
        "subsample": [0.7, 0.8, 0.9, 1.0],
    },
    DecisionTreeClassifier: {
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 4, 8, 16],
        "max_features": ["sqrt", "log2"],
        "criterion": ["gini", "entropy", "log_loss"],
    },
    ExtraTreeClassifier: {
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 4, 8, 16],
        "max_features": ["sqrt", "log2"],
        "criterion": ["gini", "entropy", "log_loss"],
    },
    SVC: {
        "C": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [2, 3, 4, 5],
        "gamma": ["scale", "auto"] + [0.1, 1.0, 10.0],
        "shrinking": [True, False],
    },
    NuSVC: {
        "nu": [0.1, 0.3, 0.5, 0.7],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [2, 3, 4, 5],
        "gamma": ["scale", "auto"] + [0.1, 1.0, 10.0],
        "shrinking": [True, False],
        # sometimes, it hangs and does not converge
        "max_iter": [1000000],
    },
    LinearSVC: {
        "C": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "loss": ["hinge", "squared_hinge"],
        "penalty": [ "l2"],
        "dual": ['auto'],
        "multi_class": ["ovr", "crammer_singer"],
    },
    KNeighborsClassifier: {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": [10, 20, 30, 40, 50],
        "p": [1, 2],
    },
    NearestCentroid: {
        "metric": ["euclidean", "manhattan", "cosine"],
        "shrink_threshold": [None] + [0.1, 0.2, 0.3, 0.4, 0.5],
    },
    BernoulliNB: {
        "alpha": [0.0, 0.1, 0.2, 0.3, 0.4],
        "binarize": [0.0, 0.1, 0.2, 0.3, 0.4],
        "fit_prior": [True, False],
    },
    GaussianNB: {
        "priors": [None] + [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]
    },
    AdaBoostClassifier: {
        "n_estimators": [50, 100, 200, 400, 800],
        "learning_rate": [0.01, 0.1, 0.2, 0.5, 1.0],
        "algorithm": ["SAMME", "SAMME.R"],
        "base_estimator": [
            DecisionTreeClassifier(max_depth=1),
            DecisionTreeClassifier(max_depth=2),
        ],
    },
    BaggingClassifier: {
        "n_estimators": [10, 50, 100, 200, 500],
        "max_samples": [1.0, 0.8, 0.6, 0.4, 0.2],
        "max_features": [1.0, 0.8, 0.6, 0.4, 0.2],
        "bootstrap": [True, False],
        "bootstrap_features": [True, False],
    },
    # REMOVED BECAUSE IN SOME INSTANCES IT FAILS TO CONVERGE
    # CalibratedClassifierCV: {
    #     "method": ["sigmoid", "isotonic"],
    # },
    LinearDiscriminantAnalysis: {
        "solver": ["lsqr"],
        "shrinkage": [None, 0.1, 0.2, 0.3, 0.4],
        "priors": [None] + [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]],
    },
    QuadraticDiscriminantAnalysis: {
        "priors": [None] + [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]],
        "reg_param": [0.0, 0.1, 0.2, 0.3, 0.4],
    },
    LabelSpreading: {
        "kernel": ["knn", "rbf"],
        "n_neighbors": [3, 5, 7, 9, 11],
        "alpha": [0.1, 0.2, 0.3, 0.4, 0.5],
        "max_iter": [30],
        "n_jobs": [None, -1],
    },
    LabelPropagation: {
        "kernel": ["knn", "rbf"],
        "n_neighbors": [3, 5, 7, 9, 11],
        "max_iter": [1000],
        "n_jobs": [None, -1],
    },
    LogisticRegression: {
        "penalty": ["l2"],
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "max_iter": [100],
        "multi_class": ["auto", "ovr"],
    },
    RidgeClassifierCV: {
        "alphas": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "store_cv_values": [True, False],
        "fit_intercept": [True, False],
    },
    RidgeClassifier: {
        "alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "fit_intercept": [True, False],
        # "max_iter": [100, 200, 300, 400, 500],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
    },
    SGDClassifier: {
        "loss": ["hinge", "modified_huber", "squared_hinge", "perceptron"],
        "penalty": ["l2",  "elasticnet"],
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
        "eta0": [0.01, 0.1, 1.0, 10.0],
        "learning_rate": ["optimal", "constant", "invscaling", "adaptive"],
        "max_iter": [1000],
    },
    Perceptron: {
        "penalty": [None, "l2",  "elasticnet"],
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
        "fit_intercept": [True, False],
        "max_iter": [1000],
        "shuffle": [True, False],
    },
    PassiveAggressiveClassifier: {
        "C": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "fit_intercept": [True, False],
        "max_iter": [1000],
        "shuffle": [True, False],
        "average": [True, False],
    },
    DummyClassifier: {"strategy": ["stratified"]},
}

# CLASSIFIERS_HYPERPARAMETER_LIST: dict[ClassifierMixin, dict[str, Any]] = {
#     NuSVC: {
#         # large nus does not make it converge
#         "nu": [0.1, 0.3, 0.5, 0.7, 0.9],
#         "kernel": ["linear", "poly", "rbf", "sigmoid"],
#         "degree": [2, 3, 4, 5],
#         "gamma": ["scale", "auto"] + [0.1, 1.0, 10.0],
#         "shrinking": [True, False],
#         "max_iter": [1000000],
#     },
# }
