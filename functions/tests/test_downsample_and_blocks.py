import numpy as np
import matplotlib.pyplot as plt
from functions.downsample import downsample_channel
from functions.block_processing import split_into_blocks


def create_test_matrix(size: tuple) -> np.ndarray:
    """Создаёт тестовую матрицу с градиентом заданного размера (h, w)."""
    return np.arange(size[0] * size[1], dtype=np.float32).reshape(size)


def create_test_color_image(size: tuple) -> np.ndarray:
    """Создаёт цветное изображение заданного размера (h, w)."""
    h, w = size
    image = np.zeros((h, w, 3), dtype=np.uint8)

    # Красный квадрат
    image[:h // 2, :w // 2, 0] = 255

    # Зелёный квадрат
    image[h // 2:, w // 2:, 1] = 255

    # Синяя полоса
    image[:, w // 4:3 * w // 4, 2] = 255

    return image


def visualize_matrix(matrix: np.ndarray, title: str):
    """Визуализирует матрицу с аннотациями значений."""
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, cmap='viridis', interpolation='none')
    plt.colorbar()

    # Аннотации значений для маленьких матриц
    if matrix.shape[0] <= 8 and matrix.shape[1] <= 8:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, f"{matrix[i, j]:.1f}",
                         ha="center", va="center", color="w")

    plt.title(title)
    plt.show()


def test_matrix_processing(matrix: np.ndarray, name: str, block_size: int = 8):
    """Полный тест обработки для одной матрицы."""
    print(f"\n=== Тест матрицы {name} {matrix.shape} ===")

    # Оригинал
    visualize_matrix(matrix, f"Оригинал {name}\n{matrix.shape}")

    # Даунсэмплинг
    downsampled = downsample_channel(matrix.copy(), ratio=2)
    visualize_matrix(downsampled,
                     f"Даунсэмплированная {name}\n{downsampled.shape}")

    # Разбиение на блоки
    blocks = split_into_blocks(downsampled, block_size=block_size)
    print(f"Блоки {name}: {blocks.shape}")

    # Визуализация первого блока
    if blocks.size > 0:
        first_block = blocks[0, 0] if len(blocks.shape) == 4 else blocks[0, 0, :, :, 0]
        visualize_matrix(first_block,
                         f"Первый блок {name}\n{first_block.shape}")


def test_color_image_processing(image: np.ndarray, name: str):
    """Тест обработки цветного изображения."""
    print(f"\n=== Тест цветного изображения {name} {image.shape[:2]} ===")

    # Оригинал
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f"Оригинал {name}\n{image.shape[:2]}")
    plt.show()

    # Яркость и цветность
    Y = image.mean(axis=2).astype(np.float32)
    Cb = (image[:, :, 2] - Y).astype(np.float32)
    Cr = (image[:, :, 0] - Y).astype(np.float32)

    # Даунсэмплинг
    Cb_down = downsample_channel(Cb, ratio=2)
    Cr_down = downsample_channel(Cr, ratio=2)

    # Визуализация каналов
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(Y, cmap='gray')
    plt.title("Y канал")

    plt.subplot(1, 3, 2)
    plt.imshow(Cb_down, cmap='gray')
    plt.title("Cb даунсэмплированный")

    plt.subplot(1, 3, 3)
    plt.imshow(Cr_down, cmap='gray')
    plt.title("Cr даунсэмплированный")
    plt.show()

    # Разбиение на блоки 4x4
    y_blocks = split_into_blocks(Y, 4)
    cb_blocks = split_into_blocks(Cb_down, 4)
    cr_blocks = split_into_blocks(Cr_down, 4)
    print(f"Блоки Y: {y_blocks.shape}, Cb: {cb_blocks.shape}, Cr: {cr_blocks.shape}")


def test_downsample_and_blocks():
    # Тестируем матрицы разных размеров
    test_cases = [
        ("3x3", (3, 3)),
        ("5x5", (5, 5)),
        ("17x17", (17, 17)),
        ("20x20", (20, 20))
    ]

    for name, size in test_cases:
        matrix = create_test_matrix(size)
        test_matrix_processing(matrix, name, block_size=4 if size[0] <= 8 else 8)

    # Тестируем цветные изображения
    color_test_cases = [
        ("4x4", (4, 4)),
        ("16x16", (16, 16)),
        ("17x17", (17, 17))
    ]

    for name, size in color_test_cases:
        image = create_test_color_image(size)
        test_color_image_processing(image, name)


if __name__ == "__main__":
    test_downsample_and_blocks()