import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.stats import mode


class KNN:
    def __init__(self, X_train, X_test, Y_train, Y_test, k_val):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.k_val = k_val

        self.k_neighbors_list = None

    def predict(self, X):
        predictions, knn_indices = self._predict(X)
        return predictions

    def predict_point(self, X):
        predictions, knn_indices = self._predict(X)
        return knn_indices

    def _predict(self, x):
        y_hat = []
        for test_pt in x.to_numpy():  # Для каждой точки из тестового набора данных
            distances = []
            for i in range(len(self.X_train)):  # Вычисляем расстояние между текущей точкой и каждой точкой тренировочного набора
                distances.append(euclidean_distance(np.array(self.X_train.iloc[i]), test_pt))

            distances_data = pd.DataFrame(data=distances, columns=['distance'], index=self.X_train.index)

            # Находим K ближайших соседей
            k_neighbors_list = distances_data.sort_values(by=['distance'], axis=0)[:self.k_val]

            # Получаем метки (labels) для K ближайших соседей
            labels = k_neighbors_list.index

            # Выбираем наиболее частое значение среди меток соседей
            voting = mode([self.Y_train.loc[x] for x in labels])[0]

            y_hat.append(voting)

        return y_hat, labels

    def accuracy(self):
        y_pred = self.predict(self.X_test)
        accuracy = accuracy_score(self.Y_test, y_pred)
        return accuracy


# Определение функции расчета евклидова расстояния между двумя точками
def euclidean_distance(pt1, pt2) -> int:
    distance = np.sqrt((pt1[0] - pt2[0]) ** 2 + ((pt1[1] - pt2[1]) / 1000) ** 2)
    return distance
