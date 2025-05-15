import numpy as np
from rgb_to_ycbcr import rgb_to_ycbcr,ycbcr_to_rgb

# ---------- Тест ----------

def test_color_conversion():
    # Создаём простое RGB изображение 4x4
    test_img = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
        [[255, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0]],
        [[128, 128, 128], [64, 64, 64], [32, 32, 32], [192, 192, 192]],
        [[200, 100, 50], [50, 100, 200], [0, 128, 255], [128, 0, 255]]
    ], dtype=np.uint8)

    # Преобразуем в YCbCr и обратно
    Y, Cb, Cr = rgb_to_ycbcr(test_img)
    restored_rgb = ycbcr_to_rgb(Y, Cb, Cr)

    # Выводим отклонения
    diff = np.abs(test_img.astype(np.int32) - restored_rgb.astype(np.int32))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print("Максимальное отклонение:", max_diff)
    print("Среднее отклонение:", mean_diff)

    if max_diff <= 2:
        print("✅ Тест пройден: восстановленное изображение близко к оригиналу.")
    else:
        print("❌ Тест не пройден: есть значительные расхождения.")

    # (необязательно) показать изображения
    # cv2.imshow("Original", test_img)
    # cv2.imshow("Restored", restored_rgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

test_color_conversion()
