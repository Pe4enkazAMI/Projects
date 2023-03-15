from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample
        self.train_loss: float = 0

        self.early_stopping_rounds: int = early_stopping_rounds
        self.validation_loss: float = 0
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        n = int(self.subsample*x.shape[0])
        sample = np.random.choice(x.shape[0], n) # выбрали индексы
        x_boot = x[sample] # бутстрапированная выборка
        y_boot = y[sample] # бутстрапированная выборка

        new_base_model = self.base_model_class(**self.base_model_params)
        new_base_model.fit(x_boot, y_boot)
        new_pred = new_base_model.predict(x)
        gamma = self.find_optimal_gamma(y=y,
                                        old_predictions=predictions,
                                        new_predictions=new_pred)

        self.gammas.append(gamma)
        self.models.append(new_base_model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        val_loss = []
        val_acc = []
        count = 0
        best_score = 0
        for _ in range(self.n_estimators):
            si = -self.loss_derivative(y=y_train, z=train_predictions)
            self.fit_new_base_model(x_train, si, train_predictions)

            train_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_train)
            valid_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_valid)

            val_loss.append(self.validation_loss)
            self.validation_loss += self.loss_fn(y_valid, valid_predictions)
            self.train_loss += self.loss_fn(y_train, train_predictions)
            val_acc.append(self.score(x_valid, y_valid))

            scoring = self.score(x_valid, y_valid)
            if self.early_stopping_rounds is not None:
                if scoring > best_score:
                    best_score = max(best_score, scoring)
                else:
                    count += 1
                if count == self.early_stopping_rounds:
                    break

        if self.plot:
            plt.plot(val_acc)
            plt.xlabel("n_estimators")
            plt.ylabel("Validation ROC-AUC")

    def predict_proba(self, x):
        preds_prob = np.zeros((x.shape[0], 2))
        sum_of_predicts = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            sum_of_predicts += gamma * model.predict(x)
        preds_prob[:, 0], preds_prob[:, 1] = self.sigmoid(1 - sum_of_predicts), self.sigmoid(sum_of_predicts)
        return preds_prob

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        matrix = np.zeros(self.models[0].feature_importances_.shape[0])
        for model in self.models:
            matrix += model.feature_importances_
        return matrix / np.sum(matrix)
