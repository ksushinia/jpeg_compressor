import numpy as np

def get_quantization_matrix(channel,quality=50, block_size=8):
    """
    Генерирует матрицу квантования для заданного качества
    quality: 1-100 (100 - наилучшее качество)
    block_size: размер блока (обычно 8)
    """
    # Стандартная матрица квантования JPEG для яркости (luma)
    std_luma_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    std_chroma_matrix = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ])

    if channel=='Y':
        matrix=std_luma_matrix
    else:
        matrix = std_chroma_matrix
    if block_size != 8:
        # Масштабируем матрицу для других размеров блоков
        matrix = np.kron(matrix, np.ones((block_size//8, block_size//8),dtype=np.float32))

    # Корректировка качества
    if quality < 1:
        quality = 1
    elif quality > 100:
        quality = 100

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    quant_matrix = np.floor((matrix * scale + 50) / 100).astype(np.float32)
    quant_matrix = np.clip(quant_matrix, 1, 255)

    return quant_matrix

def quantize_blocks(dct_blocks, quant_matrix):
    """
    Квантует DCT-блоки с использованием матрицы квантования
    """
    quantized_blocks = np.empty_like(dct_blocks, dtype=np.int32)

    for i in range(dct_blocks.shape[0]):
        for j in range(dct_blocks.shape[1]):
            quantized_blocks[i,j] = np.round(dct_blocks[i,j] / quant_matrix).astype(np.int32)
    return quantized_blocks

def dequantize_blocks(quantized_blocks, quant_matrix):
    """
    Обратное квантование блоков
    """
    dct_blocks = np.empty_like(quantized_blocks, dtype=np.float32)
    for i in range(quantized_blocks.shape[0]):
        for j in range(quantized_blocks.shape[1]):
            dct_blocks[i,j] = quantized_blocks[i,j] * quant_matrix
    return dct_blocks.astype(np.float32)