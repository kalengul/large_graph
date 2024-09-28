import random
import time
import pandas as pd
import numpy as np
import win32com.client  # Для загрузки из Excel
import os
import continious
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import fsolve


def load_Excel_eigenvalues_real():
    # Загрузите данные из Excel
    file_path = 'matrix.xlsx'
    df = pd.read_excel(file_path, index_col=0)
    # Преобразуйте DataFrame в матрицу смежности
    adjacency_matrix = df.values
    # Получите названия вершин
    vertices = df.index.tolist()
    # print("Названия вершин:", vertices, len(vertices))
    # print("Матрица смежности:\n", adjacency_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(adjacency_matrix)
    # print("Собственные значения:")
    # print(eigenvalues)
    # print("Собственные векторы:")
    # print(eigenvectors)
    # Убираем комплексную составляющую
    eigenvalues_real = eigenvalues.real
    eigenvectors_real = eigenvectors.real
    # Теперь eigenvalues_real и eigenvectors_real содержат только действительные части
    # print("Действительные собственные значения:", eigenvalues_real, len(eigenvalues_real))
    # print("Действительные собственные векторы:\n", eigenvectors_real)
    return vertices, eigenvalues_real, adjacency_matrix


class Production:
    industry = ''
    N_Production = 0
    power = []
    arr_production = []
    arr_parametr = []


# Функция для генерации данных о предприятиях
def generate_enterprises(vertices):
    file_path = 'matrix_production.xlsx'
    df = pd.read_excel(file_path, index_col=0)
    # Преобразуйте DataFrame в матрицу смежности
    enterprises_matrix = df.values
    kol_enterprises = 0
    # Получите названия вершин
    enterprises_vertices = df.index.tolist()
    M_Norm_Parametr = [0] * N_Parametr
    D_Norm_Parametr = [0] * N_Parametr
    i = 0
    while i < N_Parametr:
        M_Norm_Parametr[i] = random.random() * 100
        D_Norm_Parametr[i] = random.random() * 500
        i = i + 1
    enterprises_data = {}
    # print('vertices=',vertices)
    for industry_index, industry in enumerate(vertices):
        # print('industry_index=',industry_index, 'industry=',industry)
        # Случайное количество предприятий
        num_enterprises = random.randint(Min_N_enterprises, Max_N_enterprises)
        kol_enterprises = kol_enterprises + num_enterprises

        # Список для хранения данных о предприятиях
        enterprises = []

        for i in range(num_enterprises):
            # Генерация случайной мощности предприятия (например, от 100 до 10000)
            power = random.random() * K_Power

            # Генерация объема произведенной продукции (например, от 1000 до 50000)
            production_volume = random.randint(1000, 50000)

            # Генерация массива параметров на основе объемов производства
            parameters = []
            for j in range(N_Parametr):
                if enterprises_matrix[industry_index, j % 125] != 0:
                    Mx = M_Norm_Parametr[j] * enterprises_matrix[industry_index, j % 125]
                    Dx = D_Norm_Parametr[j]
                    param_value = np.random.normal(Mx, Dx)  # Генерация по нормальному распределению
                else:
                    param_value = 0
                parameters.append(param_value)

            # Сохранение данных о предприятии
            enterprises.append({
                'industry': industry_index,
                'power': power,
                'production_volume': production_volume,
                'parameters': parameters
            })
        # print(len(enterprises),'enterprises=', enterprises)
        # Сохранение данных об отрасли
        enterprises_data[industry_index] = enterprises

    return enterprises_data, kol_enterprises


def print_enterprise(enterprise_data):
    for industry, enterprises in enterprise_data.items():
        print(f"Отрасль: {industry}")
        for i, enterprise in enumerate(enterprises):
            print(f"  Предприятие {i + 1}:")
            print(f"    Мощность: {enterprise['power']}")
            print(f"    Объемы продукции: {enterprise['production_volume']}")
            print(f"    Параметры: {enterprise['parameters']}")


# Функция для сохранения результатов в Excel
def save_results_to_excel(results_df, filename='results.xlsx'):
    results_df.to_excel(filename, index=False)
    print(f"Результаты сохранены в файл: {filename}")


def clustering(enterprise_data,Eps):
    # Подготовка данных для кластеризации
    data_for_clustering = []
    enterprise_labels = []

    for industry, enterprises in enterprise_data.items():
        for i, enterprise in enumerate(enterprises):
            # Собираем параметры для кластеризации
            data_for_clustering.append(enterprise['parameters'])
            enterprise_labels.append(f"{industry} Предприятие {i + 1}")

    # Преобразуем в массив NumPy
    data_array = np.array(data_for_clustering)

    # Инициализация DataFrame для хранения результатов
    results_df = pd.DataFrame({'Предприятие': enterprise_labels})

    # K-Means
    #num_clusters = Nom_Cluster
    #kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    #cluster_labels_k_means = kmeans.fit_predict(data_array)

    # DBSCAN
    dbscan = DBSCAN(eps=Eps, min_samples=5)
    cluster_labels_dbscan = dbscan.fit_predict(data_array)
    Nom_Cluster = len(set(cluster_labels_dbscan)) - (1 if -1 in cluster_labels_dbscan else 0)
    Nom_outliers = np.sum(cluster_labels_dbscan == -1)

    # Mean Shift
    #mean_shift = MeanShift()
    #cluster_labels_mean_shift = mean_shift.fit_predict(data_array)
    #Nom_Cluster = len(np.unique(cluster_labels_mean_shift))

    # Сохраняем результат в enterprise_data
    index = 0
    differences_count=0
    # Словарь для хранения соответствия между кластерами
    cluster_mapping = {}
    for industry in enterprise_data.keys():
        for enterprise in enterprise_data[industry]:
            # Добавляем информацию о кластере в каждое предприятие
            enterprise['cluster_k_means'] = int(cluster_labels_dbscan[index])
            # Получаем текущий кластер K-means и кластер industry
            k_means_cluster = enterprise['cluster_k_means']
            industry_cluster = enterprise['industry']

            # Заполняем словарь соответствия
            if k_means_cluster!=-1:
                if industry_cluster not in cluster_mapping:
                    cluster_mapping[industry_cluster] = k_means_cluster
                else:
                    # Если уже есть соответствие, проверяем его
                    if cluster_mapping[industry_cluster] != k_means_cluster:
                        differences_count += 1
            index += 1


    # Сохранение результатов в Excel
    # save_results_to_excel(results_df)
    return enterprise_data,Nom_Cluster, Nom_outliers,differences_count


def regression_cluster(enterprise_data, eigenvalues_real):
    # Регрессионная модель для одного кластера
    arr_model = []
    arr_mse = []
    kol_enterprises_cluster = 0
    selected_cluster = 0  # Выберите нужный кластер
    while selected_cluster < Nom_Cluster:
        # Собираем данные для регрессии
        X = []
        y = []

        for industry, enterprises in enterprise_data.items():
            for enterprise in enterprises:
                if enterprise['cluster_k_means'] == selected_cluster:
                    parameters = enterprise['parameters']
                    power = enterprise['power']
                    industry_value = eigenvalues_real[enterprise['industry']]
                    kol_enterprises_cluster = kol_enterprises_cluster + 1
                    X.append(parameters)  # Параметры + мощность
                    y.append(industry_value * power)  # Значение отрасли * мощность

        X = np.array(X)
        y = np.array(y)

        # Обучение регрессионной модели
        model = LinearRegression()
        model.fit(X, y)

        # Оценка модели
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)

        arr_model.append(model)
        arr_mse.append(mse)
        selected_cluster = selected_cluster + 1

    # print(f"Среднеквадратичная ошибка: {mse}")
    return arr_model, arr_mse, kol_enterprises_cluster


def regression_cluster_indusry(enterprise_data, eigenvalues_real):
    # Регрессионная модель для одного кластера
    arr_model = []
    arr_mse = []
    selected_cluster = 0  # Выберите нужный кластер
    while selected_cluster < len(eigenvalues_real):
        # Собираем данные для регрессии
        X = []
        y = []

        for industry, enterprises in enterprise_data.items():
            for enterprise in enterprises:
                if enterprise['industry'] == selected_cluster:
                    parameters = enterprise['parameters']
                    power = enterprise['power']
                    industry_value = eigenvalues_real[enterprise['industry']]

                    X.append(parameters)  # Параметры + мощность
                    y.append(industry_value * power)  # Значение отрасли * мощность

        X = np.array(X)
        y = np.array(y)

        # Обучение регрессионной модели
        model = LinearRegression()
        model.fit(X, y)

        # Оценка модели
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)

        arr_model.append(model)
        arr_mse.append(mse)
        selected_cluster = selected_cluster + 1

    # print(f"Среднеквадратичная ошибка: {mse}")
    return arr_model, arr_mse


# Функция для поиска X по известному Y
def find_x_for_y(model, y_target):
    # Определяем функцию, которая равна разнице между предсказанным Y и целевым Y
    def func(x):
        return model.predict(np.array([[x]]))[0] - y_target

    # Используем fsolve для нахождения корня уравнения func(x) = 0
    x_solution = fsolve(func, [0] * N_Parametr)  # Начальное приближение x0
    return x_solution


if __name__ == "__main__":
    print('Загрузка матрицы и вычисление собственных векторов по отраслям производства')
    vertices, eigenvalues_real, adjacency_matrix = load_Excel_eigenvalues_real()

    NameFile = os.getcwd() + '/file.xlsx'
    Excel = win32com.client.Dispatch("Excel.Application")
    wb = Excel.Workbooks.Open(NameFile)
    sheet = wb.ActiveSheet

    arr_mse = []
    arr_mse_indusry = []
    K_Power = 100
    Min_N_enterprises = 250
    Max_N_enterprises = 500
    N_Parametr = 500
    Nom_Cluster = 20

    Row = 2
    ColConst = 4
    Eps = 8600
    while Eps > 7000:
        for Col in range(100):

            start_time = time.time()
            print(Eps,Nom_Cluster, Col, 'Создание предприятий')
            enterprises, kol_enterprises = generate_enterprises(vertices)
            end_time = time.time()
            sheet.Cells(Row + 1, Col + ColConst).value = end_time - start_time
            print(Nom_Cluster, kol_enterprises, Col, 'Кластеризация производств')
            start_time = time.time()
            enterprise_data, Nom_Cluster, Nom_outliers,differences_count = clustering(enterprises,Eps)
            end_time = time.time()
            sheet.Cells(Row, Col + ColConst).value = Nom_Cluster
            sheet.Cells(Row + 2, Col + ColConst).value = end_time - start_time
            print(Nom_Cluster, Nom_outliers, differences_count, Col, 'Построение уравнения регрессии')
            start_time = time.time()
            arr_regression_model, arr_mse, kol_enterprises_cluster = regression_cluster(enterprise_data,
                                                                                        eigenvalues_real)
            end_time = time.time()
            sheet.Cells(Row + 3, Col + ColConst).value = end_time - start_time
            sheet.Cells(Row + 4, Col + ColConst).value = max(arr_mse)
            sheet.Cells(Row + 5, Col + ColConst).value = min(arr_mse)
            sheet.Cells(Row + 6, Col + ColConst).value = sum(arr_mse) / len(arr_mse)
            sheet.Cells(Row + 7, Col + ColConst).value = kol_enterprises
            sheet.Cells(Row + 8, Col + ColConst).value = kol_enterprises_cluster
            print(Nom_Cluster, Col, 'Построение уравнения регрессии по отраслям')
            start_time = time.time()
            arr_regression_model_indusry, arr_mse_indusry = regression_cluster_indusry(enterprise_data,
                                                                                       eigenvalues_real)
            end_time = time.time()
            sheet.Cells(Row + 9, Col + ColConst).value = end_time - start_time
            sheet.Cells(Row + 10, Col + ColConst).value = max(arr_mse_indusry)
            sheet.Cells(Row + 11, Col + ColConst).value = min(arr_mse_indusry)
            sheet.Cells(Row + 12, Col + ColConst).value = sum(arr_mse_indusry) / len(arr_mse_indusry)
            sheet.Cells(Row + 13, Col + ColConst).value = Nom_outliers
            sheet.Cells(Row + 14, Col + ColConst).value = differences_count
        Eps = Eps - 200
        Row = Row + 16
        # сохраняем рабочую книгу
        wb.Save()
    # закрываем ее
    wb.Close()
    # закрываем COM объект
    Excel.Quit()
