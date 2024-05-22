import numpy as np
from scipy.stats import mode
import pandas as pd


# Определение функции расчета евклидова расстояния между двумя точками
def euclidean_distance(pt1, pt2) -> int:
    distance = np.sqrt(np.sum(pt1-pt2)**2)
    return distance


# Определение функции K ближайших соседей (KNN)
def KNN(X_train, X_test, Y_train, Y_test, k_val):
    y_hat = []
    for test_pt in X_test.to_numpy():  # Для каждой точки из тестового набора данных
        distances = []
        for i in range(len(X_train)):  # Вычисляем расстояние между текущей точкой и каждой точкой тренировочного набора
            distances.append(euclidean_distance(np.array(X_train.iloc[i]), test_pt))

        distances_data = pd.DataFrame(data=distances, columns=['distance'], index=Y_train.index)

        # Находим K ближайших соседей
        k_neighbors_list = distances_data.sort_values(by=['distance'], axis=0)[:k_val]

        # Получаем метки (labels) для K ближайших соседей
        labels = Y_train.loc[k_neighbors_list.index]

        # Выбираем наиболее частое значение среди меток соседей
        voting = mode(labels)[0]

        y_hat.append(voting)

    return y_hat
