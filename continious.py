import math
from random import random

# Класс, описывающий непрерывное случайное распределение
class ContinuousRandom:
    # Конструктор класса
    def __init__(self, a, b):
        self.a = a  # Начало интервала
        self.b = b  # Конец интервала

    # Функция, возвращающая значение случайной величины
    def func(self, x):
        return x * (self.b - self.a) + self.a

    # Функция итерации
    def __iter__(self):
        return self

    # Функция получения следующего значения случайной величины
    def __next__(self):
        rnd = random()  # Генерация случайного числа
        return self.func(x=rnd)


# Класс, описывающий экспоненциальное случайное распределение
class ExponentialRandom:
    # Конструктор класса
    def __init__(self, lambd):
        self.lambd = lambd  # Параметр лямбда

    # Функция, возвращающая значение случайной величины
    def func(self, x):
        return -1 / self.lambd * math.log(x)

    # Функция итерации
    def __iter__(self):
        return self

    # Функция получения следующего значения случайной величины
    def __next__(self):
        rnd = random()  # Генерация случайного числа
        return self.func(x=rnd)


# Класс, описывающий нормальное случайное распределение
class NormalRandom:
    # Конструктор класса
    def __init__(self, Mx, Dx):
        self.Mx = Mx  # Математическое ожидание
        self.Dx = Dx  # Дисперсия
        self.N = 20   # Количество значений для усреднения

    # Функция итерации
    def __iter__(self):
        return self

    # Функция получения следующего значения случайной величины
    def __next__(self):
        sum = 0
        for i in range(self.N):
            sum = sum + random()  # Суммирование случайных значений
        return (sum / (self.N / 2) - 1) * self.Dx + self.Mx


# Класс, описывающий основную функцию случайного распределения
class MajorFunctionRandom:
    # Функция, возвращающая значение случайной величины
    def func(self, x):
        return -1 / self.lambd * math.log(x)

    # Функция поиска максимальной функции
    def SearchMaxFunction(self, a, b):
        return 1

    # Конструктор класса
    def __init__(self, a, b, lambd):
        self.a = a            # Начало интервала
        self.b = b            # Конец интервала
        self.lambd = lambd    # Параметр лямбда
        self.max = self.SearchMaxFunction(a, b)  # Максимальное значение функции

    # Функция итерации
    def __iter__(self):
        return self

    # Функция получения следующего значения случайной величины
    def __next__(self):
        CRX = ContinuousRandom(a=self.a, b=self.b)  # Создание объекта непрерывного случайного распределения
        CRY = ContinuousRandom(a=0, b=self.max)     # Создание объекта непрерывного случайного распределения
        Y = 0                                      # Значение случайной величины Y
        Yrnd = 5                                   # Случайное значение Yrnd
        kol = 0                                    # Счетчик итераций
        while Yrnd > Y:
            Xrnd = next(CRX)                       # Получение следующего значения Xrnd
            Y = self.func(x=Xrnd)                  # Вычисление значения Y
            Yrnd = next(CRY)                       # Получение следующего значения Yrnd
            kol = kol + 1                          # Увеличение счетчика итераций
        print(kol)                                  # Вывод количества итераций
        return Xrnd                                 # Возвращение значения Xrnd



if __name__ == '__main__':
    a = -20
    b = -3
    CR = ContinuousRandom(a=a, b=b)
    for i in range(10):
        print(next(CR))

    print('Экспоненциальный з.р')
    l = 1000
    ER = ExponentialRandom(lambd=l)
    for i in range(10):
        print(next(ER))

    print('Нормальный з.р')
    mx = -100
    dx = 1
    NR = NormalRandom(Mx=mx, Dx=dx)
    for i in range(10):
        print(next(NR))

    print('Мажорирующая функция')
    a = 0
    b = 100
    l = 1
    MFR = MajorFunctionRandom(a=a, b=b, lambd=l)
    for i in range(10):
        print(next(MFR))
