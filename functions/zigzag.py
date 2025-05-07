import numpy as np


def zigzag_scan(matrix):
    """
    Оптимизированное зигзаг-сканирование матрицы NxN.
    Возвращает 1D массив элементов в порядке зигзаг-сканирования.
    """
    n = len(matrix)
    result = np.zeros(n * n, dtype=matrix.dtype)  # Сохраняем исходный тип данных

    index = 0
    for i in range(2 * n - 1):
        if i % 2 == 0:  # Движение вверх-вправо
            row = min(i, n - 1)
            col = max(0, i - n + 1)
            while row >= 0 and col < n:
                result[index] = matrix[row, col]
                index += 1
                row -= 1
                col += 1
        else:  # Движение вниз-влево
            col = min(i, n - 1)
            row = max(0, i - n + 1)
            while col >= 0 and row < n:
                result[index] = matrix[row, col]
                index += 1
                row += 1
                col -= 1

    return result


def inverse_zigzag_scan(array, n):
    """
    Оптимизированное восстановление матрицы NxN из зигзаг-сканированного массива.
    Принимает на вход array (list или numpy.array) и явно преобразует в float32.
    """
    matrix = np.zeros((n, n), dtype=np.float32)  # Явно задаём тип float32
    pos = 0
    array = np.asarray(array, dtype=np.float32)  # Гарантируем numpy array

    for i in range(2 * n - 1):
        if i % 2 == 0:  # Движение вверх-вправо
            row = min(i, n - 1)
            col = max(0, i - n + 1)
            while row >= 0 and col < n and pos < len(array):
                matrix[row, col] = array[pos]
                row -= 1
                col += 1
                pos += 1
        else:  # Движение вниз-влево
            col = min(i, n - 1)
            row = max(0, i - n + 1)
            while col >= 0 and row < n and pos < len(array):
                matrix[row, col] = array[pos]
                row += 1
                col -= 1
                pos += 1

    return matrix