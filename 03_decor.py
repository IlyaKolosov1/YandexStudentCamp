import sys
from functools import wraps

def takes(*types):
    def decorator(func):
        @wraps(func)  # Сохраняем метаданные функции
        def wrapper(*args):
            # Проверяем типы для переданных аргументов
            for arg, expected_type in zip(args, types):
                if not isinstance(arg, expected_type):
                    raise TypeError
            return func(*args)
        return wrapper
    return decorator

# Выполняем код из stdin
exec(sys.stdin.read())

