import numpy as np


def extract_dc_coefficients(quantized_blocks):
    """
    Извлекает DC-коэффициенты (элемент [0,0]) из всех квантованных блоков.

    Параметры:
        quantized_blocks (np.ndarray): 3D-массив блоков формы (num_blocks_h, num_blocks_w, block_size, block_size)

    Возвращает:
        np.ndarray: 1D-массив DC-коэффициентов в порядке обхода блоков (слева направо, сверху вниз)
    """
    # Преобразуем 3D-массив блоков в список DC-коэффициентов
    dc_coeffs = []
    for row in quantized_blocks:
        for block in row:
            dc_coeffs.append(block[0, 0])
    return np.array(dc_coeffs)


def delta_encode_dc(dc_coeffs):
    """
    Применяет разностное кодирование к DC-коэффициентам.

    Параметры:
        dc_coeffs (np.ndarray): 1D-массив DC-коэффициентов

    Возвращает:
        np.ndarray: 1D-массив дельт (разностей между соседними DC-коэффициентами)
    """
    delta_dc = np.zeros_like(dc_coeffs)
    delta_dc[0] = dc_coeffs[0]  # Первый коэффициент сохраняем как есть
    for i in range(1, len(dc_coeffs)):
        delta_dc[i] = dc_coeffs[i] - dc_coeffs[i - 1]
    return delta_dc


def delta_decode_dc(delta_dc):
    """
    Восстанавливает исходные DC-коэффициенты из дельт.

    Параметры:
        delta_dc (np.ndarray): 1D-массив дельт

    Возвращает:
        np.ndarray: 1D-массив восстановленных DC-коэффициентов
    """
    dc_coeffs = np.zeros_like(delta_dc)
    dc_coeffs[0] = delta_dc[0]  # Первый коэффициент не изменен
    for i in range(1, len(delta_dc)):
        dc_coeffs[i] = dc_coeffs[i - 1] + delta_dc[i]
    return dc_coeffs


def update_blocks_with_dc(quantized_blocks, dc_coeffs):
    """
    Обновляет DC-коэффициенты в блоках после декодирования.

    Параметры:
        quantized_blocks (np.ndarray): 3D-массив блоков
        dc_coeffs (np.ndarray): 1D-массив новых DC-коэффициентов

    Возвращает:
        np.ndarray: Блоки с обновленными DC-коэффициентами
    """
    idx = 0
    updated_blocks = quantized_blocks.copy()
    for i in range(updated_blocks.shape[0]):
        for j in range(updated_blocks.shape[1]):
            updated_blocks[i, j, 0, 0] = dc_coeffs[idx]
            idx += 1
    return updated_blocks