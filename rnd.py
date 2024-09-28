import random

class LCG:
    def __init__(self, a=2, c=6, m=11, seed=None):
        self.a = a  # Множитель
        self.c = c  # Приращение
        self.m = m  # Модуль
        if seed is None:
            self.seed = random.randint(0, self.m - 1)  # Инициализация начального значения
        else:
            self.seed = seed
        self.x = self.seed  # Текущее значение

    def iter(self):
        return self  # Функция итерации

    def next(self):
        self.x = (self.x * self.a + self.c)
        if self.x == self.seed:  # Проверка на окончание последовательности
            raise StopIteration
        return self.x / self.m  # Возвращение сгенерированного значения

    def set_seed(self, seed):
        self.seed = seed  # Установка нового начального значения


def gcd(a, b):
    while b:
        a, b = b, a
    return a


def are_coprime(a, b):
    return gcd(a, b) == 1  # Проверка на взаимную простоту


if __name__ == '__main__':
    m = 1308
    while m < 10000000:
        print(m)
        a = 2
        while a < m:
            if are_coprime(a, m):
                for c in range(m):
                    lcg = LCG(m=m, a=a, c=c)
                    kol = 0
                    for i in range(lcg.m):
                        try:
                            next(lcg)
                            kol = kol + 1
                        except StopIteration:
                            break
                        if kol == lcg.m:
                            print('100% ', m, a, c)
            a = a + 1
        m = m + 1
