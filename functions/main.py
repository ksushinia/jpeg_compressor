import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from rgb_to_ycbcr import rgb_to_ycbcr, ycbcr_to_rgb
from downsample import downsample_channel, visualize_channels
from block_processing import split_into_blocks, assemble_blocks
from dct import dct2, idct2
from quantization import get_quantization_matrix, quantize_blocks, dequantize_blocks
from zigzag import zigzag_scan, inverse_zigzag_scan
from delta_encoding import extract_dc_coefficients, delta_encode_dc, delta_decode_dc, update_blocks_with_dc


def apply_dct_to_blocks(blocks):
    """Применяет DCT ко всем блокам"""
    return np.array([[dct2(block) for block in row] for row in blocks])


def apply_idct_to_blocks(dct_blocks):
    """Применяет обратное DCT ко всем блокам"""
    return np.array([[idct2(block) for block in row] for row in dct_blocks])


def process_channel(channel, quant_matrix, channel_name="", block_size=8):
    """Полный пайплайн обработки канала: DCT -> Квантование -> Обратное DCT"""
    # Разбиваем на блоки
    blocks = split_into_blocks(channel, block_size=block_size)

    # Применяем DCT
    dct_blocks = apply_dct_to_blocks(blocks)

    # Квантуем коэффициенты
    quantized_blocks = quantize_blocks(dct_blocks, quant_matrix)

    # Разностное кодирование DC
    dc_coeffs = extract_dc_coefficients(quantized_blocks)
    delta_dc = delta_encode_dc(dc_coeffs)

    # Здесь можно добавить энтропийное кодирование (например, Хаффмана)
    # В этом примере пропускаем, так как у нас только имитация

    # Обратное преобразование (имитация декодера)
    restored_dc = delta_decode_dc(delta_dc)
    quantized_blocks = update_blocks_with_dc(quantized_blocks, restored_dc)

    # Зигзаг-сканирование (только если нужно для энтропийного кодирования)
    quantized_blocks_zigzag = np.array([[zigzag_scan(block) for block in row] for row in quantized_blocks])

    # Обратное зигзаг-сканирование (имитация декодирования)
    quantized_blocks = np.array(
        [[inverse_zigzag_scan(block, block_size) for block in row] for row in quantized_blocks_zigzag])

    # Обратное квантование
    dequant_blocks = dequantize_blocks(quantized_blocks, quant_matrix)

    # Обратное DCT
    reconstructed_blocks = apply_idct_to_blocks(dequant_blocks)

    # Собираем блоки обратно
    reconstructed = assemble_blocks(reconstructed_blocks)[:channel.shape[0], :channel.shape[1]]

    # Вычисляем ошибку
    error = np.abs(channel - reconstructed)
    print(f"\nОшибка восстановления {channel_name}:")
    print(f"Максимальная: {np.max(error):.2f}")
    print(f"Средняя: {np.mean(error):.2f}")

    return reconstructed, quantized_blocks


def visualize_results(original, reconstructed, dct_coeffs, title):
    """Визуализирует оригинал, восстановленное изображение и DCT-коэффициенты"""
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title(f"Оригинал {title}")

    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed, cmap='gray')
    plt.title(f"Восстановлено {title}")

    plt.subplot(1, 3, 3)
    plt.imshow(np.log1p(np.abs(dct_coeffs)), cmap='gray')
    plt.title(f"DCT коэффициенты {title} (log scale)")

    plt.tight_layout()
    plt.show()


def main():
    # Параметры обработки
    qualities = [25, 75]  # Тестируем оба качества
    block_size = 8

    # 1. Загрузка изображения
    img_path = "C:/Users/79508/Desktop/4 семестри/АИСД/2 лабораторная/коди/images/Lenna.png"
    img = Image.open(img_path).convert("RGB")
    rgb_array = np.array(img)

    # 2. Конвертация в YCbCr
    Y, Cb, Cr = rgb_to_ycbcr(rgb_array)

    # 3. Даунсэмплинг цветовых каналов (4:2:0)
    Cb_down = downsample_channel(Cb, ratio=2)
    Cr_down = downsample_channel(Cr, ratio=2)

    plt.figure(figsize=(12, 8))

    for i, quality in enumerate(qualities, 1):
        # 4. Получаем матрицы квантования
        quant_matrix_luma = get_quantization_matrix(quality, block_size)
        quant_matrix_chroma = get_quantization_matrix(quality + 10, block_size)

        # 5. Обработка каждого канала
        Y_rec, Y_quant = process_channel(Y, quant_matrix_luma, "Y", block_size)
        Cb_rec, _ = process_channel(Cb_down, quant_matrix_chroma, "Cb", block_size)
        Cr_rec, _ = process_channel(Cr_down, quant_matrix_chroma, "Cr", block_size)

        # 6. Апсемплинг цветовых каналов
        Cb_rec = np.kron(Cb_rec, np.ones((2, 2)))[:Cb.shape[0], :Cb.shape[1]]
        Cr_rec = np.kron(Cr_rec, np.ones((2, 2)))[:Cr.shape[0], :Cr.shape[1]]

        # 7. Обратное преобразование в RGB
        rgb_reconstructed = ycbcr_to_rgb(Y_rec, Cb_rec, Cr_rec)

        # Визуализация
        plt.subplot(2, 2, i)
        plt.imshow(rgb_reconstructed)
        plt.title(f"Качество: {quality}%")

        # Визуализация ошибок для Y канала
        plt.subplot(2, 2, i + 2)
        error = np.abs(Y - Y_rec)
        plt.imshow(error, cmap='hot', vmin=0, vmax=32)
        plt.colorbar()
        plt.title(f"Ошибка Y канала (max: {np.max(error):.1f})")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()