import math

def log_f(x):
    """
    Вычисляет натуральный логарифм функции f(x) = 1000^x / x!.
    """
    return x * math.log(1000) - math.lgamma(x + 1)

def find_max():
    """
    Находит значение x, при котором f(x) достигает максимума.
    """
    prev = log_f(1)  # Начальное значение log(f(1))
    for i in range(2, 10**6):  # Ограничиваем диапазон
        val = log_f(i)
        if val < prev:  # Если значение уменьшилось, возвращаем предыдущий x
            return i - 1
        prev = val  # Обновляем предыдущее значение
    return None  # Если максимум не найден

def main():
    result = find_max()
    if result is not None:
        print(f"Максимум достигается при x = {result}")
    else:
        print("Максимум не найден в заданном диапазоне.")

# Вызов main()
if __name__ == "__main__":
    main()