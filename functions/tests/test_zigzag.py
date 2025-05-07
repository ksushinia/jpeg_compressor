import numpy as np
import matplotlib.pyplot as plt
from functions.zigzag import zigzag_scan, inverse_zigzag_scan


def test_zigzag():
    # Тестируем на матрице 8x8
    n = 8
    test_matrix = np.arange(n * n).reshape(n, n)

    print("Исходная матрица:")
    print(test_matrix)

    # Применяем зигзаг-сканирование
    zigzag = zigzag_scan(test_matrix)
    print("\nЗигзаг-последовательность:")
    print(zigzag)

    # Обратное преобразование
    reconstructed = inverse_zigzag_scan(zigzag, n)
    print("\nВосстановленная матрица:")
    print(reconstructed)

    # Проверка корректности
    assert np.array_equal(test_matrix, reconstructed), "Ошибка: матрицы не совпадают!"
    print("\nТест пройден: матрица успешно восстановлена")


def visualize_zigzag():
    # Визуализация порядка сканирования
    n = 8
    matrix = np.zeros((n, n))

    # Заполняем матрицу порядковыми номерами сканирования
    order = 0
    for i in range(2 * n - 1):
        if i % 2 == 0:
            row = min(i, n - 1)
            col = max(0, i - n + 1)
            while row >= 0 and col < n:
                matrix[row][col] = order
                order += 1
                row -= 1
                col += 1
        else:
            col = min(i, n - 1)
            row = max(0, i - n + 1)
            while col >= 0 and row < n:
                matrix[row][col] = order
                order += 1
                row += 1
                col -= 1

    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.title("Порядок зигзаг-сканирования")

    for i in range(n):
        for j in range(n):
            plt.text(j, i, int(matrix[i, j]), ha='center', va='center', color='w')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_zigzag()
    visualize_zigzag()