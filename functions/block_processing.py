import numpy as np


def split_into_blocks(image: np.ndarray, block_size: int = 8, fill_value: int = 0) -> np.ndarray:
    """
    Разбивает изображение на блоки NxN с заполнением при необходимости

    Параметры:
        image (np.ndarray): Входное изображение (H x W) или (H x W x C)
        block_size (int): Размер блока (по умолчанию 8)
        fill_value (int): Значение для заполнения (по умолчанию 0)

    Возвращает:
        np.ndarray: Массив блоков формы (num_blocks_h, num_blocks_w, block_size, block_size[, C])
    """
    # Получаем размеры изображения
    if len(image.shape) == 2:
        h, w = image.shape
        channels = 1
    else:
        h, w, channels = image.shape

    # Вычисляем количество блоков с округлением вверх
    num_blocks_h = (h + block_size - 1) // block_size
    num_blocks_w = (w + block_size - 1) // block_size

    # Вычисляем размеры с дополнением
    padded_h = num_blocks_h * block_size
    padded_w = num_blocks_w * block_size

    # Создаем массив с заполнением
    if len(image.shape) == 2:
        padded = np.full((padded_h, padded_w), fill_value, dtype=image.dtype)
        padded[:h, :w] = image
    else:
        padded = np.full((padded_h, padded_w, channels), fill_value, dtype=image.dtype)
        padded[:h, :w, :] = image

    # Разбиваем на блоки
    if len(image.shape) == 2:
        blocks = padded.reshape(num_blocks_h, block_size, num_blocks_w, block_size)
        blocks = blocks.transpose(0, 2, 1, 3)
    else:
        blocks = padded.reshape(num_blocks_h, block_size, num_blocks_w, block_size, channels)
        blocks = blocks.transpose(0, 2, 1, 3, 4)

    return blocks

def assemble_blocks(blocks):
    """Собирает блоки обратно в изображение"""
    if len(blocks.shape) == 4:  # Для grayscale
        num_h, num_w, bs, _ = blocks.shape
        return blocks.transpose(0, 2, 1, 3).reshape(num_h * bs, num_w * bs)
    else:  # Для color
        num_h, num_w, bs, _, c = blocks.shape
        return blocks.transpose(0, 2, 1, 3, 4).reshape(num_h * bs, num_w * bs, c)