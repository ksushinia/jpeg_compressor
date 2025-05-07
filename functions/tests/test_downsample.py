import numpy as np
from functions.downsample import downsample_channel


def test_downsample():
    print("Тест 1: Проверка на матрице 4x4")
    test_matrix = np.array([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [90, 100, 110, 120],
        [130, 140, 150, 160]
    ], dtype=np.float32)

    expected = np.array([
        [(10 + 20 + 50 + 60) / 4, (30 + 40 + 70 + 80) / 4],
        [(90 + 100 + 130 + 140) / 4, (110 + 120 + 150 + 160) / 4]
    ])

    result = downsample_channel(test_matrix)
    print("Ожидаемый результат:\n", expected)
    print("Фактический результат:\n", result)
    print("Совпадает?", np.allclose(result, expected))

    print("\nТест 2: Проверка граничных случаев (некратный размер)")
    test_matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float32)

    expected = np.array([
        [(1 + 2 + 4 + 5) / 4]
    ])

    result = downsample_channel(test_matrix)
    print("Ожидаемый результат:\n", expected)
    print("Фактический результат:\n", result)
    print("Совпадает?", np.allclose(result, expected))

    print("\nТест 3: Проверка на случайных данных")
    np.random.seed(42)
    random_matrix = np.random.randint(0, 256, (8, 8), dtype=np.uint8)

    # Проверяем ручной расчет для первого блока 2x2
    manual_avg = random_matrix[:2, :2].mean()
    result = downsample_channel(random_matrix)
    print("Среднее первого блока 2x2 (ожидаемое):", manual_avg)
    print("Первый пиксель результата:", result[0, 0])
    print("Совпадает?", np.isclose(manual_avg, result[0, 0]))


if __name__ == "__main__":
    test_downsample()