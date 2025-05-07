import numpy as np

import numpy as np


def dct_matrix(N):
    """
    Генерирует матрицу DCT-II для N-точечного преобразования.

    Параметры:
        N (int): Размер преобразования (обычно 8 для JPEG)

    Возвращает:
        np.ndarray: Матрица DCT размера N x N
    """
    n = np.arange(N)
    k = n.reshape(N, 1)
    # Коэффициенты масштабирования
    C = np.where(n == 0, 1 / np.sqrt(2), 1) * np.sqrt(2 / N)
    # Ядро DCT
    return C * np.cos((2 * k + 1) * n * np.pi / (2 * N))


def dct2(block):
    """
    Быстрое 2D DCT-II через матричные операции.
    (~300x быстрее наивной реализации)

    Параметры:
        block (np.ndarray): Блок изображения N x N

    Возвращает:
        np.ndarray: DCT-коэффициенты блока
    """
    D = dct_matrix(block.shape[0])
    return D @ block @ D.T  # Эквивалентно двум 1D DCT


def idct2(dct_block):
    """
    Быстрое обратное 2D DCT-II через матричные операции.

    Параметры:
        dct_block (np.ndarray): Блок DCT-коэффициентов N x N

    Возвращает:
        np.ndarray: Восстановленный блок изображения
    """
    D = dct_matrix(dct_block.shape[0])
    return D.T @ dct_block @ D  # Транспонированная матрица для обратного преобразования