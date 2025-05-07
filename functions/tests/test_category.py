import numpy as np
from functions.category import get_category


def test_get_category():
    """
    Тестирует функцию get_category на различных входных значениях.
    Проверяет категорию, дополнительные биты и возможность восстановления значения.
    """
    test_cases = [
        (0, (0, "")),  # Категория 0: 0
        (1, (1, "1")),  # Категория 1: 1
        (-1, (1, "0")),  # Категория 1: -1
        (2, (2, "10")),  # Категория 2: 2
        (-2, (2, "01")),  # Категория 2: -2
        (3, (2, "11")),  # Категория 2: 3
        (-3, (2, "00")),  # Категория 2: -3
        (4, (3, "100")),  # Категория 3: 4
        (-4, (3, "011")),  # Категория 3: -4
        (7, (3, "111")),  # Категория 3: 7
        (-7, (3, "000")),  # Категория 3: -7
        (8, (4, "1000")),  # Категория 4: 8
        (-8, (4, "0111")),  # Категория 4: -8
        (15, (4, "1111")),  # Категория 4: 15
        (-15, (4, "0000")),  # Категория 4: -15
    ]

    for value, expected in test_cases:
        category, amplitude_bits = get_category(value)
        assert category == expected[0], (
            f"Ошибка для value={value}: ожидаемая категория {expected[0]}, получено {category}"
        )
        assert amplitude_bits == expected[1], (
            f"Ошибка для value={value}: ожидаемые биты {expected[1]}, получено {amplitude_bits}"
        )

        # Проверка восстановления значения
        if category > 0:
            decoded_amplitude = int(amplitude_bits, 2)
            if decoded_amplitude < (1 << (category - 1)):
                decoded_value = decoded_amplitude - (1 << category) + 1
            else:
                decoded_value = decoded_amplitude
            assert decoded_value == value, (
                f"Ошибка восстановления для value={value}: ожидалось {value}, получено {decoded_value}"
            )

    print("Все тесты пройдены успешно!")


if __name__ == "__main__":
    test_get_category()