import numpy as np


def get_category(value):
    """
    Определяет категорию (Size) и дополнительные биты для значения согласно JPEG.

    Args:
        value (int): DC-разность или AC-коэффициент.

    Returns:
        tuple: (category, amplitude_bits)
            - category (int): Номер категории (0–11 для JPEG).
            - amplitude_bits (str): Бинарная строка дополнительных битов для значения.
    """
    if value == 0:
        return 0, ""

    abs_value = abs(value)
    # Категория: наименьшее n, такое что 2^(n-1) <= |value| < 2^n
    category = int(np.ceil(np.log2(abs_value + 1)))

    # Дополнительные биты
    if value > 0:
        amplitude = value
    else:
        # Для отрицательных чисел: value = -(2^n - 1 - amplitude)
        amplitude = value + (1 << category) - 1

    amplitude_bits = bin(amplitude)[2:].zfill(category)
    return category, amplitude_bits