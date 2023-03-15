import math

import numpy as np
from collections import Counter
import typing as tp


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов, len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    zipped_target_feature = np.array(sorted(zip(feature_vector, target_vector), key=lambda x: x[0]))
    sorted_target = zipped_target_feature[:, 1]

    unique_feature_nums, count_unique = np.unique(zipped_target_feature[:, 0], return_counts=True)

    thrs = (unique_feature_nums[:-1] + unique_feature_nums[1:])/2

    if len(thrs) == 0:
        return [], [], None, None

    possible_positive = np.arange(1, len(target_vector))

    left_ratio = np.cumsum(sorted_target[:-1])/possible_positive

    impurity_left = 1 - left_ratio**2 - (1 - left_ratio)**2

    right_ratio = np.flip(np.cumsum(np.flip(sorted_target, axis=0)[:-1]) / possible_positive, axis=0)

    impurity_right = 1 - right_ratio**2 - (1 - right_ratio)**2

    Q = -1 * possible_positive / len(target_vector) * impurity_left - np.flip(possible_positive, axis=0) / \
        len(target_vector) * impurity_right

    ginis = Q[(np.cumsum(count_unique) - 1)[:-1]]
    best_index_gini = np.argmax(ginis)
    best_gin = ginis[best_index_gini]
    thrs_best = thrs[best_index_gini]

    return thrs, ginis, thrs_best, best_gin

# БЛЯТЬ, ПИШИТЕ ТАЙПИНГИ!!!!!!!!!!!!


class DecisionTree:
    def __init__(self, feature_types: tp.List[str],
                 max_depth: tp.Union[int, None] = None,
                 min_samples_split: tp.Union[int, None] = None,
                 min_samples_leaf: tp.Union[int, None] = None):

        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self.depth = 0

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]): # Надо проверять, что у нас все одного класса
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            self.depth = max(self.depth, depth)
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None

        if depth < self._max_depth:
            for feature in range(sub_X.shape[1]): # Надо проходить по всем
                feature_type = self._feature_types[feature]
                categories_map = {}

                if feature_type == "real":
                    feature_vector = sub_X[:, feature]
                elif feature_type == "categorical":
                    counts = Counter(sub_X[:, feature])
                    clicks = Counter(sub_X[sub_y == 1, feature])
                    ratio = {}
                    for key, current_count in counts.items():
                        if key in clicks:
                            current_click = clicks[key]
                        else:
                            current_click = 0
                        ratio[key] = current_click / current_count # формула для кликов вот такая а не как была
                    sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                    # выше ошибка была в категории
                    categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                    feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
                    # list забыли
                else:
                    raise ValueError

                if len(np.unique(feature_vector)) == 1: # проверяем в рамках одной фичи
                    continue

                _, _, threshold, gini = find_best_split(feature_vector, sub_y)
                if gini_best is None or gini > gini_best:
                    feature_best = feature
                    gini_best = gini
                    split = feature_vector < threshold
                    if np.sum(split) >= self._min_samples_leaf and \
                            len(split) - np.sum(split) >= self._min_samples_leaf:

                        if feature_type == "real":
                            threshold_best = threshold
                        elif feature_type == "categorical": # у нас все с маленькой буквы
                            threshold_best = list(map(lambda x: x[0],
                                                      filter(lambda x: x[1] < threshold, categories_map.items())))
                        else:
                            raise ValueError

        if feature_best is None or len(sub_X) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0] # most_common возвращает лист тюплов
            self.depth = max(self.depth, depth)
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth+1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth+1)
        # ну признаки тоже надо

    # обход дерева in_order
    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        if self._feature_types[node["feature_split"]] == "real":
            if x[node["feature_split"]] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            if x[node["feature_split"]] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self, deep=True):
        return {"feature_types": self._feature_types}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self