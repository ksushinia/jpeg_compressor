import numpy as np
import matplotlib.pyplot as plt
from functions.dct import dct2, idct2


def test_dct(block_size=8):
    """
    Тестирование DCT преобразования

    Параметры:
        block_size (int): Размер тестового блока
    """
    # Создаем тестовый блок
    np.random.seed(42)
    original_block = np.random.rand(block_size, block_size) * 255

    # Применяем DCT
    dct_coeffs = dct2(original_block)

    # Применяем обратное DCT
    reconstructed_block = idct2(dct_coeffs)

    # Вычисляем ошибку
    error = np.abs(original_block - reconstructed_block)
    max_error = np.max(error)
    mean_error = np.mean(error)

    # Вывод результатов
    print(f"=== Результаты для блока {block_size}x{block_size} ===")
    print(f"Максимальная ошибка восстановления: {max_error:.6f}")
    print(f"Средняя ошибка восстановления: {mean_error:.6f}")

    # Визуализация
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_block, cmap='gray')
    plt.title("Оригинальный блок")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(np.log1p(np.abs(dct_coeffs)), cmap='gray')
    plt.title("DCT коэффициенты (log scale)")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed_block, cmap='gray')
    plt.title("Восстановленный блок")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    return max_error < 1e-8


if __name__ == "__main__":
    # Тестируем на разных размерах блоков
    for size in [4, 8, 16, 32]:
        print(f"\nТестируем блок {size}x{size}")
        success = test_dct(size)
        print("Тест пройден:", "✓" if success else "✗")