import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def test_rgb_to_ycbcr():
    # Тест 1: Проверка на известных значениях
    print("=== Тест 1: Проверка на контрольных значениях ===")
    test_pixel = np.array([[[100, 150, 200]]], dtype=np.uint8)

    Y, Cb, Cr = rgb_to_ycbcr(test_pixel)

    print(f"RGB: {test_pixel[0, 0]}")
    print(f"Y: {Y[0, 0]:.2f} (ожидается ~140.75)")
    print(f"Cb: {Cb[0, 0]:.2f} (ожидается ~161.44)")
    print(f"Cr: {Cr[0, 0]:.2f} (ожидается ~98.94)")

    # Тест 2: Проверка граничных случаев
    print("\n=== Тест 2: Граничные случаи ===")
    black = np.array([[[0, 0, 0]]], dtype=np.uint8)
    white = np.array([[[255, 255, 255]]], dtype=np.uint8)

    Y_black, Cb_black, Cr_black = rgb_to_ycbcr(black)
    Y_white, Cb_white, Cr_white = rgb_to_ycbcr(white)

    print(f"Черный RGB: {black[0, 0]} -> YCbCr: {Y_black[0, 0]:.1f}, {Cb_black[0, 0]:.1f}, {Cr_black[0, 0]:.1f}")
    print(f"Белый RGB: {white[0, 0]} -> YCbCr: {Y_white[0, 0]:.1f}, {Cb_white[0, 0]:.1f}, {Cr_white[0, 0]:.1f}")

    # Тест 3: Загрузка тестового изображения
    print("\n=== Тест 3: Визуальная проверка на изображении ===")
    img_path = "/images/Lenna.png"
    img = Image.open(img_path).convert("RGB")
    rgb_array = np.array(img)

    Y, Cb, Cr = rgb_to_ycbcr(rgb_array)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(Y, cmap='gray')
    plt.title("Y канал")

    plt.subplot(1, 3, 2)
    plt.imshow(Cb, cmap='gray')
    plt.title("Cb канал")

    plt.subplot(1, 3, 3)
    plt.imshow(Cr, cmap='gray')
    plt.title("Cr канал")
    plt.show()


def rgb_to_ycbcr(rgb_image):
    """
    (Копия вашей функции для независимой работы теста)
    """
    if rgb_image.dtype != np.float32 and rgb_image.dtype != np.float64:
        rgb_image = rgb_image.astype(np.float32)

    R = rgb_image[:, :, 0]
    G = rgb_image[:, :, 1]
    B = rgb_image[:, :, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

    Y = np.clip(Y, 0, 255)
    Cb = np.clip(Cb, 0, 255)
    Cr = np.clip(Cr, 0, 255)

    return Y, Cb, Cr


if __name__ == "__main__":
    test_rgb_to_ycbcr()