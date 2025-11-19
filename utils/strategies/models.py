from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Union, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from shared_schemas.model_configs import ModelModel
from utils.strategies.interfaces import ModelBuilder


@dataclass
class LogRegBuilder(ModelBuilder):
    cfg: ModelModel

    def make_estimator(self) -> Any:
        penalty = self.cfg.penalty
        solver = self.cfg.solver

        sk_penalty: Optional[str]
        if penalty == "none":
            sk_penalty = None
        else:
            sk_penalty = penalty

        if sk_penalty is None:
            if solver == "liblinear":
                solver = "lbfgs"
        elif sk_penalty == "elasticnet":
            solver = "saga"
        elif sk_penalty == "l1":
            if solver not in ("liblinear", "saga"):
                solver = "saga"

        l1_ratio = self.cfg.l1_ratio if sk_penalty == "elasticnet" else None

        return LogisticRegression(
            C=self.cfg.C,
            penalty=sk_penalty,
            solver=solver,
            max_iter=self.cfg.max_iter,
            class_weight=self.cfg.class_weight,
            multi_class="auto",
            l1_ratio=l1_ratio,
        )

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class SVMBuilder(ModelBuilder):
    cfg: ModelModel
    def make_estimator(self) -> Any:
        return SVC(
            C=self.cfg.svm_C,
            kernel=self.cfg.svm_kernel,
            degree=self.cfg.svm_degree,
            gamma=self.cfg.svm_gamma,
            coef0=self.cfg.svm_coef0,
            shrinking=self.cfg.svm_shrinking,
            probability=self.cfg.svm_probability,
            tol=self.cfg.svm_tol,
            cache_size=self.cfg.svm_cache_size,
            class_weight=self.cfg.svm_class_weight,
            max_iter=self.cfg.svm_max_iter,
            decision_function_shape=self.cfg.svm_decision_function_shape,
            break_ties=self.cfg.svm_break_ties,
        )
    def build(self) -> Any:
        return self.make_estimator()



@dataclass
class DecisionTreeBuilder(ModelBuilder):
    cfg: ModelModel
    def make_estimator(self) -> Any:
        return DecisionTreeClassifier(
            criterion=self.cfg.tree_criterion,
            splitter=self.cfg.tree_splitter,
            max_depth=self.cfg.tree_max_depth,
            min_samples_split=self.cfg.tree_min_samples_split,
            min_samples_leaf=self.cfg.tree_min_samples_leaf,
            min_weight_fraction_leaf=self.cfg.tree_min_weight_fraction_leaf,
            max_features=self.cfg.tree_max_features,
            max_leaf_nodes=self.cfg.tree_max_leaf_nodes,
            min_impurity_decrease=self.cfg.tree_min_impurity_decrease,
            class_weight=self.cfg.tree_class_weight,
            ccp_alpha=self.cfg.tree_ccp_alpha,
        )
    def build(self) -> Any:
        return self.make_estimator()



@dataclass
class RandomForestBuilder(ModelBuilder):
    cfg: ModelModel
    def make_estimator(self) -> Any:
        return RandomForestClassifier(
            n_estimators=self.cfg.rf_n_estimators,
            criterion=self.cfg.rf_criterion,
            max_depth=self.cfg.rf_max_depth,
            min_samples_split=self.cfg.rf_min_samples_split,
            min_samples_leaf=self.cfg.rf_min_samples_leaf,
            min_weight_fraction_leaf=self.cfg.rf_min_weight_fraction_leaf,
            max_features=self.cfg.rf_max_features,
            max_leaf_nodes=self.cfg.rf_max_leaf_nodes,
            min_impurity_decrease=self.cfg.rf_min_impurity_decrease,
            bootstrap=self.cfg.rf_bootstrap,
            oob_score=self.cfg.rf_oob_score,
            n_jobs=self.cfg.rf_n_jobs,
            class_weight=self.cfg.rf_class_weight,
            ccp_alpha=self.cfg.rf_ccp_alpha,
            warm_start=self.cfg.rf_warm_start,
        )
    def build(self) -> Any:
        return self.make_estimator()

@dataclass
class KNNBuilder(ModelBuilder):
    cfg: ModelModel
    def make_estimator(self) -> Any:
        return KNeighborsClassifier(
            n_neighbors=self.cfg.knn_n_neighbors,
            weights=self.cfg.knn_weights,
            algorithm=self.cfg.knn_algorithm,
            leaf_size=self.cfg.knn_leaf_size,
            p=self.cfg.knn_p,
            metric=self.cfg.knn_metric,
            n_jobs=self.cfg.knn_n_jobs,
        )
    def build(self) -> Any:
        return self.make_estimator()