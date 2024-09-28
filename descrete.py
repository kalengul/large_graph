from random import random


class DiscreteRandom:
    # Конструктор класса
    def __init__(self, X=[], P=[]):
        self.X = X  # Значения случайной величины
        self.P = P  # Вероятности появления значений случайной величины
        self.Fx = []  # Значения функции распределения
        if self.P != []:
            self.Fx.append(self.P[0])  # Добавление первого значения функции распределения
            i = 1
            while i < len(self.P):
                self.Fx.append(self.Fx[i - 1] + self.P[i])  # Вычисление значений функции распределения
                i = i + 1

    # Функция итерации
    def __iter__(self):
        return self

    # Функция получения следующего значения случайной величины
    def __next__(self):
        rnd = random()  # Генерация случайного числа
        i = 0
        while (i < len(self.Fx)) and (rnd > self.Fx[i]):
            i = i + 1
        return self.X[i]  # Возвращение значения случайной величины


if __name__ == '__main__':
    X = [1,2,3,4]
    P = [0.1, 0.2, 0.3, 0.4]

    DR = DiscreteRandom(X=X, P=P)
    for i in range(10):
        print(next(DR))