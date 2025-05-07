import numpy as np
from functions.rle import rle_encode, rle_decode
from functions.zigzag import zigzag_scan, inverse_zigzag_scan


def test_rle():
    # Тестовая матрица 8x8
    test_matrix = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0, 6, 0, 0],
        [0, 0, 0, 0, 0, 0, 7, 0],
        [0, 0, 0, 0, 0, 0, 0, 8]
    ])

    # Применяем зигзаг-сканирование
    zigzag_result = zigzag_scan(test_matrix)

    # Кодируем с помощью RLE
    rle_result = rle_encode(zigzag_result)

    # Декодируем обратно
    decoded_result = rle_decode(rle_result)

    # Преобразуем декодированный список в numpy.array перед inverse_zigzag_scan
    decoded_array = np.array(decoded_result)

    # Восстанавливаем матрицу
    restored_matrix = inverse_zigzag_scan(decoded_array, 8)

    # Проверяем корректность
    assert np.array_equal(test_matrix, restored_matrix), "RLE encoding/decoding failed!"
    print("RLE test passed successfully!")
    print("Zigzag result:", zigzag_result)
    print("RLE result:", rle_result)
    print("Decoded result:", decoded_result)


if __name__ == "__main__":
    test_rle()