import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from rgb_to_ycbcr import rgb_to_ycbcr, ycbcr_to_rgb
from downsample import downsample_channel
from block_processing import split_into_blocks, assemble_blocks
from dct import dct2, idct2
from quantization import get_quantization_matrix, quantize_blocks, dequantize_blocks
from zigzag import zigzag_scan, inverse_zigzag_scan
from delta_encoding import extract_dc_coefficients, delta_encode_dc, delta_decode_dc
from dc_variable_coding import encode_dc_coefficients, decode_dc_coefficients
from ac_variable_coding import encode_ac_coefficients, decode_ac_coefficients

def load_image(path):
    """Загружает изображение и конвертирует в RGB numpy array."""
    print("1. Загрузка изображения...")
    img = Image.open(path).convert("RGB")
    return np.array(img)


def upsample_channel(channel: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Увеличивает разрешение канала до target_shape (с использованием PIL)
    """
    print("  Увеличение разрешения канала...")
    from PIL import Image
    img = Image.fromarray(channel)
    return np.array(img.resize((target_shape[1], target_shape[0]), Image.BILINEAR))


def process_channel(channel: np.ndarray, quant_matrix=None, block_size: int = 8, is_luma: bool = True) -> tuple:
    """
    Исправленная версия с преобразованием типов
    """
    blocks = split_into_blocks(channel, block_size)
    dct_blocks = np.zeros_like(blocks, dtype=np.float32)

    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            dct_blocks[i, j] = dct2(blocks[i, j])

    if quant_matrix is not None:
        quantized_blocks = quantize_blocks(dct_blocks, quant_matrix)

        # DC коэффициенты
        dc_coeffs = []
        for i in range(quantized_blocks.shape[0]):
            for j in range(quantized_blocks.shape[1]):
                dc_value = int(round(float(quantized_blocks[i, j, 0, 0])))
                dc_coeffs.append(dc_value)

        dc_bitstream = encode_dc_coefficients(np.array(dc_coeffs), is_luma)

        # AC коэффициенты
        ac_bitstreams = []
        for i in range(quantized_blocks.shape[0]):
            for j in range(quantized_blocks.shape[1]):
                block = quantized_blocks[i, j].copy()
                block[0, 0] = 0  # Удаляем DC
                zigzag = zigzag_scan(block)[1:]  # AC коэффициенты
                zigzag = [int(round(float(x))) for x in zigzag]  # Преобразуем
                ac_bitstream = encode_ac_coefficients(zigzag, is_luma)
                ac_bitstreams.append(ac_bitstream)

        return dc_bitstream, ac_bitstreams

    return None, None


def inverse_process_channel(dc_bitstream: str, ac_bitstreams: list,
                            quant_matrix=None, original_shape: tuple = None,
                            block_size: int = 8, is_luma: bool = True) -> np.ndarray:
    """
    Обратное преобразование с декодированием RLE и Хаффмана
    """
    h_blocks = original_shape[0] // block_size
    w_blocks = original_shape[1] // block_size

    # Восстановление DC-коэффициентов
    print("  Декодирование DC...")
    dc_coeffs = decode_dc_coefficients(dc_bitstream, h_blocks * w_blocks, is_luma)

    # Восстановление блоков
    restored_blocks = np.zeros((h_blocks, w_blocks, block_size, block_size), dtype=np.float32)

    print("  Декодирование AC...")
    idx = 0
    for i in range(h_blocks):
        for j in range(w_blocks):
            # Восстанавливаем AC-коэффициенты
            ac_coeffs = decode_ac_coefficients(ac_bitstreams[idx], block_size * block_size - 1, is_luma)

            # Собираем блок (DC + AC)
            block = np.zeros((block_size, block_size))
            block[0, 0] = dc_coeffs[idx]
            restored_block = inverse_zigzag_scan(np.insert(ac_coeffs, 0, block[0, 0]), block_size)

            # Обратное квантование
            restored_blocks[i, j] = dequantize_blocks(restored_block[np.newaxis, np.newaxis, :, :], quant_matrix)[0, 0]
            idx += 1

    # Сборка изображения
    print("  Сборка изображения...")
    restored = np.zeros(original_shape, dtype=np.float32)
    for i in range(h_blocks):
        for j in range(w_blocks):
            restored[i * block_size:(i + 1) * block_size,
            j * block_size:(j + 1) * block_size] = idct2(restored_blocks[i, j])

    return restored


def compress_image(image_path, output_path=None, quality=50, block_size=8):
    """
    Полная версия компрессора с RLE и кодированием Хаффмана для DC/AC коэффициентов
    """
    # 1. Загрузка изображения
    print("\nНачало обработки изображения...")
    rgb = load_image(image_path)
    original_size = rgb.nbytes
    height, width = rgb.shape[:2]
    print(f"  Размер исходного изображения: {original_size / 1024:.1f} KB")

    # 2. Конвертация в YCbCr
    print("2. Конвертация RGB -> YCbCr...")
    Y, Cb, Cr = rgb_to_ycbcr(rgb)
    Y = Y.astype(np.float32)
    Cb = Cb.astype(np.float32)
    Cr = Cr.astype(np.float32)

    # 3. Даунсэмплинг Cb и Cr (4:2:0)
    print("3. Даунсэмплинг цветовых компонентов...")
    Cb_down = downsample_channel(Cb, ratio=2)
    Cr_down = downsample_channel(Cr, ratio=2)

    # 4. Получаем матрицы квантования
    print("4. Генерация матриц квантования...")
    luma_quant_matrix = get_quantization_matrix(quality, block_size)
    chroma_quant_matrix = get_quantization_matrix(quality * 0.7, block_size)

    # 5. Обработка каналов с кодированием
    print("5. Обработка яркостного канала (Y)...")
    Y_dc_bits, Y_ac_bits = process_channel(Y, luma_quant_matrix, block_size, is_luma=True)

    print("6. Обработка цветовых каналов (Cb, Cr)...")
    Cb_dc_bits, Cb_ac_bits = process_channel(Cb_down, chroma_quant_matrix, block_size, is_luma=False)
    Cr_dc_bits, Cr_ac_bits = process_channel(Cr_down, chroma_quant_matrix, block_size, is_luma=False)

    # 6. Расчет размера сжатых данных (в байтах)
    def calculate_compressed_size(dc_bits, ac_bits_list):
        size = len(dc_bits)
        for ac_bits in ac_bits_list:
            size += len(ac_bits)
        return (size + 7) // 8  # Округление до целых байтов

    compressed_size = (
            calculate_compressed_size(Y_dc_bits, Y_ac_bits) +
            calculate_compressed_size(Cb_dc_bits, Cb_ac_bits) +
            calculate_compressed_size(Cr_dc_bits, Cr_ac_bits)
    )

    print(f"  Размер после сжатия: {compressed_size / 1024:.2f} KB")
    print(f"  Коэффициент сжатия: {original_size / compressed_size:.2f}x")

    # 7. Восстановление изображения
    print("7. Восстановление яркостного канала...")
    Y_restored = inverse_process_channel(Y_dc_bits, Y_ac_bits, luma_quant_matrix,
                                         (height, width), block_size, is_luma=True)

    print("8. Восстановление цветовых каналов...")
    Cb_restored = upsample_channel(
        inverse_process_channel(Cb_dc_bits, Cb_ac_bits, chroma_quant_matrix,
                                Cb_down.shape, block_size, is_luma=False),
        (height, width)
    )
    Cr_restored = upsample_channel(
        inverse_process_channel(Cr_dc_bits, Cr_ac_bits, chroma_quant_matrix,
                                Cr_down.shape, block_size, is_luma=False),
        (height, width)
    )

    # 8. Конвертация обратно в RGB
    print("9. Конвертация YCbCr -> RGB...")
    rgb_restored = ycbcr_to_rgb(
        np.clip(Y_restored, 0, 255).astype(np.uint8),
        np.clip(Cb_restored, 0, 255).astype(np.uint8),
        np.clip(Cr_restored, 0, 255).astype(np.uint8)
    )

    # 9. Расчет метрик качества
    mse = np.mean((rgb - rgb_restored) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')
    print(f"  PSNR: {psnr:.2f} dB")

    # 10. Визуализация результатов
    print("10. Отображение результатов...")
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title(f"Original\n{original_size / 1024:.2f} KB")

    plt.subplot(1, 2, 2)
    plt.imshow(rgb_restored)
    plt.title(
        f"Restored (Q={quality})\n{compressed_size / 1024:.2f} KB ({original_size / compressed_size:.2f}x)\nPSNR: {psnr:.2f} dB")

    if output_path:
        plt.savefig(output_path)
    plt.show()

    # 11. Возврат метрик и промежуточных данных
    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": original_size / compressed_size,
        "psnr": psnr,
        "quant_matrices": (luma_quant_matrix, chroma_quant_matrix),
        "image_size": (height, width),
        "restored_image": rgb_restored
    }


if __name__ == "__main__":
    images = [
        "C:/Users/79508/Desktop/4 семестри/АИСД/2 лабораторная/коди/images/Lenna.png",
        "C:/Users/79508/Desktop/4 семестри/АИСД/2 лабораторная/коди/images/lake.jpg"
    ]

    for img_path in images:
        print(f"\nОбработка изображения: {os.path.basename(img_path)}")
        metrics = compress_image(img_path, quality=75)  # Фиксированное качество 75