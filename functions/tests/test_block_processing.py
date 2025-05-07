import numpy as np
from functions.block_processing import split_into_blocks


def test_block_processing():
    # Тест 1: Изображение делится нацело
    print("=== Тест 1: Изображение 8x8, блоки 4x4 ===")
    img = np.arange(64).reshape(8, 8)
    blocks = split_into_blocks(img, block_size=4)
    print(f"Форма блоков: {blocks.shape}")
    print("Первый блок:\n", blocks[0, 0])

    # Тест 2: Изображение требует дополнения
    print("\n=== Тест 2: Изображение 5x5, блоки 4x4 ===")
    img = np.arange(25).reshape(5, 5)
    blocks = split_into_blocks(img, block_size=4)
    print(f"Форма блоков: {blocks.shape}")
    print("Последний блок с дополнением:\n", blocks[1, 1])

    # Тест 3: Цветное изображение
    print("\n=== Тест 3: Цветное изображение 6x6x3, блоки 4x4 ===")
    img = np.arange(108).reshape(6, 6, 3)
    blocks = split_into_blocks(img, block_size=4)
    print(f"Форма блоков: {blocks.shape}")
    print("Первый блок (красный канал):\n", blocks[0, 0, :, :, 0])


if __name__ == "__main__":
    test_block_processing()