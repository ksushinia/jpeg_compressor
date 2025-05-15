import numpy as np
from functools import lru_cache


# Кэширование матрицы DCT
@lru_cache(maxsize=None)
def create_dct_matrix(N):
    C = np.zeros((N, N))
    sqrt_N = np.sqrt(N)
    sqrt_2_N = np.sqrt(2 / N)

    n = np.arange(N)
    for k in range(N):
        if k == 0:
            C[k, :] = 1 / sqrt_N
        else:
            C[k, :] = sqrt_2_N * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    return C


# Основная функция DCT (переименованная в dct2)
def dct2(block):
    N = block.shape[0]
    C = create_dct_matrix(N)
    return np.dot(np.dot(C, block), C.T)


# Основная функция обратного DCT (переименованная в idct2)
def idct2(block):
    N = block.shape[0]
    C = create_dct_matrix(N)
    return np.dot(np.dot(C.T, block), C)


# Функция для обработки всех блоков (если нужна)
def apply_dct_to_blocks(blocks):
    dct_blocks = np.zeros_like(blocks, dtype=np.float32)
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            dct_blocks[i, j] = dct2(blocks[i, j])
    return dct_blocks


# Функция для обратного преобразования всех блоков (если нужна)
def apply_idct_to_blocks(blocks):
    idct_blocks = np.zeros_like(blocks, dtype=np.uint8)
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            idct_block = idct2(blocks[i, j])
            idct_blocks[i, j] = np.clip(np.round(idct_block), 0, 255).astype(np.uint8)
    return idct_blocks