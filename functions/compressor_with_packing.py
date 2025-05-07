import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import struct
import json
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
    Обработка канала изображения (DCT + квантование + кодирование)
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
    Полностью переработанная функция восстановления канала с обработкой ошибок
    """
    try:
        # Проверка и подготовка параметров
        if not original_shape or len(original_shape) != 2:
            raise ValueError("Неверный формат original_shape")

        h, w = original_shape
        if h <= 0 or w <= 0:
            raise ValueError("Неверные размеры изображения")

        h_blocks = (h + block_size - 1) // block_size
        w_blocks = (w + block_size - 1) // block_size

        # Инициализация выходного изображения
        restored = np.zeros((h_blocks * block_size, w_blocks * block_size), dtype=np.float32)

        # Декодирование DC коэффициентов
        dc_coeffs = decode_dc_coefficients(dc_bitstream, h_blocks * w_blocks, is_luma)

        # Обработка каждого блока
        for i in range(h_blocks):
            for j in range(w_blocks):
                idx = i * w_blocks + j
                try:
                    # Получаем DC коэффициент
                    dc_value = dc_coeffs[idx] if idx < len(dc_coeffs) else 0

                    # Декодируем AC коэффициенты
                    ac_coeffs = np.zeros(block_size * block_size - 1)
                    if idx < len(ac_bitstreams) and ac_bitstreams[idx]:
                        ac_coeffs = decode_ac_coefficients(ac_bitstreams[idx],
                                                           block_size * block_size - 1,
                                                           is_luma)

                    # Собираем блок
                    block_1d = np.insert(ac_coeffs, 0, dc_value)
                    block_2d = inverse_zigzag_scan(block_1d, block_size)

                    # Обратное квантование
                    dequant_block = dequantize_blocks(block_2d[np.newaxis, np.newaxis, :, :],
                                                      quant_matrix)[0, 0]

                    # Обратное DCT
                    idct_block = idct2(dequant_block)

                    # Размещаем блок в изображении
                    y_start, y_end = i * block_size, (i + 1) * block_size
                    x_start, x_end = j * block_size, (j + 1) * block_size

                    restored[y_start:y_end, x_start:x_end] = idct_block

                except Exception as e:
                    print(f"Ошибка в блоке ({i},{j}): {str(e)}")
                    # Заполняем блок средним значением DC
                    restored[i * block_size:(i + 1) * block_size,
                    j * block_size:(j + 1) * block_size] = dc_value

        # Обрезаем до исходного размера
        restored = restored[:h, :w]

        # Нормализация и преобразование типа
        restored = np.clip(restored, 0, 255).astype(np.uint8)

        return restored

    except Exception as e:
        print(f"Критическая ошибка в inverse_process_channel: {str(e)}")
        # Возвращаем серое изображение в случае неудачи
        return np.full(original_shape, 128, dtype=np.uint8)


def pack_data(image_size, luma_quant_matrix, chroma_quant_matrix,
              Y_dc_bits, Y_ac_bits, Cb_dc_bits, Cb_ac_bits, Cr_dc_bits, Cr_ac_bits):
    """Упаковывает все данные в байтовую строку без сжатия"""

    def bits_to_bytes(bits):
        padding = (8 - len(bits) % 8) % 8
        bits += '0' * padding
        return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8)), padding, len(bits)

    # Упаковываем DC компоненты
    Y_dc_bytes, Y_dc_pad, Y_dc_bitlen = bits_to_bytes(Y_dc_bits)
    Cb_dc_bytes, Cb_dc_pad, Cb_dc_bitlen = bits_to_bytes(Cb_dc_bits)
    Cr_dc_bytes, Cr_dc_pad, Cr_dc_bitlen = bits_to_bytes(Cr_dc_bits)

    # Упаковываем AC компоненты (сохраняем размер каждого блока)
    Y_ac_bytes_list = []
    Y_ac_bitlens = []
    for bits in Y_ac_bits:
        bytes_data, pad, bitlen = bits_to_bytes(bits)
        Y_ac_bytes_list.append(bytes_data)
        Y_ac_bitlens.append(bitlen)

    Cb_ac_bytes_list = []
    Cb_ac_bitlens = []
    for bits in Cb_ac_bits:
        bytes_data, pad, bitlen = bits_to_bytes(bits)
        Cb_ac_bytes_list.append(bytes_data)
        Cb_ac_bitlens.append(bitlen)

    Cr_ac_bytes_list = []
    Cr_ac_bitlens = []
    for bits in Cr_ac_bits:
        bytes_data, pad, bitlen = bits_to_bytes(bits)
        Cr_ac_bytes_list.append(bytes_data)
        Cr_ac_bitlens.append(bitlen)

    # Метаданные
    meta = {
        'image_size': image_size,
        'quant_matrices': {
            'luma': luma_quant_matrix.tolist(),
            'chroma': chroma_quant_matrix.tolist()
        },
        'dc_info': {
            'Y': {'padding': Y_dc_pad, 'bitlen': Y_dc_bitlen},
            'Cb': {'padding': Cb_dc_pad, 'bitlen': Cb_dc_bitlen},
            'Cr': {'padding': Cr_dc_pad, 'bitlen': Cr_dc_bitlen}
        },
        'ac_info': {
            'Y': {'bitlens': Y_ac_bitlens},
            'Cb': {'bitlens': Cb_ac_bitlens},
            'Cr': {'bitlens': Cr_ac_bitlens}
        }
    }

    # Сериализуем метаданные
    meta_json = json.dumps(meta).encode('utf-8')
    meta_len = len(meta_json)

    # Упаковываем все данные без сжатия (исправленная строка)
    packed = (
            struct.pack('I', meta_len) +
            meta_json +
            Y_dc_bytes +
            Cb_dc_bytes +
            Cr_dc_bytes +
            b''.join(Y_ac_bytes_list) +
            b''.join(Cb_ac_bytes_list) +
            b''.join(Cr_ac_bytes_list))

    return packed


def unpack_data(packed):
    """Распаковывает данные из байтовой строки"""
    # Читаем метаданные
    meta_len = struct.unpack('I', packed[:4])[0]
    meta_json = packed[4:4 + meta_len]
    meta = json.loads(meta_json.decode('utf-8'))

    # Остальные данные
    data = packed[4 + meta_len:]
    pos = 0

    # Читаем DC компоненты
    def read_dc_component(component):
        info = meta['dc_info'][component]
        byte_len = (info['bitlen'] + 7) // 8
        bytes_data = data[pos:pos + byte_len]
        bits = ''.join(f'{byte:08b}' for byte in bytes_data)
        if info['padding'] > 0:
            bits = bits[:-info['padding']]
        return bits, byte_len

    Y_dc_bits, Y_dc_len = read_dc_component('Y')
    pos += Y_dc_len

    Cb_dc_bits, Cb_dc_len = read_dc_component('Cb')
    pos += Cb_dc_len

    Cr_dc_bits, Cr_dc_len = read_dc_component('Cr')
    pos += Cr_dc_len

    # Читаем AC компоненты
    def read_ac_components(component):
        nonlocal pos  # <-- переместили наверх
        ac_bits = []
        for bitlen in meta['ac_info'][component]['bitlens']:
            byte_len = (bitlen + 7) // 8
            bytes_data = data[pos:pos + byte_len]
            bits = ''.join(f'{byte:08b}' for byte in bytes_data)
            padding = (8 - bitlen % 8) % 8
            if padding > 0:
                bits = bits[:-padding]
            ac_bits.append(bits)
            pos += byte_len
        return ac_bits

    Y_ac_bits = read_ac_components('Y')
    Cb_ac_bits = read_ac_components('Cb')
    Cr_ac_bits = read_ac_components('Cr')

    # Проверка целостности данных
    if pos != len(data):
        raise ValueError("Несоответствие размеров при распаковке данных")

    return {
        'image_size': tuple(meta['image_size']),
        'luma_quant_matrix': np.array(meta['quant_matrices']['luma']),
        'chroma_quant_matrix': np.array(meta['quant_matrices']['chroma']),
        'Y_dc_bits': Y_dc_bits,
        'Y_ac_bits': Y_ac_bits,
        'Cb_dc_bits': Cb_dc_bits,
        'Cb_ac_bits': Cb_ac_bits,
        'Cr_dc_bits': Cr_dc_bits,
        'Cr_ac_bits': Cr_ac_bits
    }


def compress_to_file(image_path, output_file, quality=50, block_size=8):
    """Сжимает изображение и сохраняет в файл"""
    # 1. Загрузка изображения
    print("\nНачало обработки изображения...")
    rgb = load_image(image_path)
    height, width = rgb.shape[:2]

    # 2. Конвертация в YCbCr
    print("2. Конвертация RGB -> YCbCr...")
    Y, Cb, Cr = rgb_to_ycbcr(rgb)
    Y = Y.astype(np.float32)
    Cb = Cb.astype(np.float32)
    Cr = Cr.astype(np.float32)

    # 3. Даунсэмплинг
    print("3. Даунсэмплинг цветовых компонентов...")
    Cb_down = downsample_channel(Cb, ratio=2)
    Cr_down = downsample_channel(Cr, ratio=2)

    # 4. Матрицы квантования
    print("4. Генерация матриц квантования...")
    luma_quant_matrix = get_quantization_matrix(quality, block_size)
    chroma_quant_matrix = get_quantization_matrix(quality * 0.7, block_size)

    # 5. Обработка каналов
    print("5. Обработка яркостного канала (Y)...")
    Y_dc_bits, Y_ac_bits = process_channel(Y, luma_quant_matrix, block_size, is_luma=True)

    print("6. Обработка цветовых каналов (Cb, Cr)...")
    Cb_dc_bits, Cb_ac_bits = process_channel(Cb_down, chroma_quant_matrix, block_size, is_luma=False)
    Cr_dc_bits, Cr_ac_bits = process_channel(Cr_down, chroma_quant_matrix, block_size, is_luma=False)

    # 6. Упаковка в байтовую строку
    print("7. Упаковка данных...")
    packed = pack_data(
        image_size=(height, width),
        luma_quant_matrix=luma_quant_matrix,
        chroma_quant_matrix=chroma_quant_matrix,
        Y_dc_bits=Y_dc_bits,
        Y_ac_bits=Y_ac_bits,
        Cb_dc_bits=Cb_dc_bits,
        Cb_ac_bits=Cb_ac_bits,
        Cr_dc_bits=Cr_dc_bits,
        Cr_ac_bits=Cr_ac_bits
    )

    # 7. Сохранение в файл
    with open(output_file, 'wb') as f:
        f.write(packed)

    # 8. Расчет статистики
    original_size = rgb.nbytes
    compressed_size = len(packed)

    print(f"  Размер исходного изображения: {original_size / 1024:.2f} KB")
    print(f"  Размер после сжатия: {compressed_size / 1024:.2f} KB")
    print(f"  Коэффициент сжатия: {original_size / compressed_size:.2f}x")

    return compressed_size


def decompress_from_file(input_file, output_image_path=None):
    """Декомпрессия изображения из файла"""
    print("\nДекомпрессия изображения...")
    # 1. Чтение файла
    with open(input_file, 'rb') as f:
        packed = f.read()

    # 2. Распаковка данных
    print("1. Распаковка данных...")
    data = unpack_data(packed)

    # 3. Восстановление каналов
    print("2. Восстановление яркостного канала (Y)...")
    Y_restored = inverse_process_channel(
        data['Y_dc_bits'], data['Y_ac_bits'],
        data['luma_quant_matrix'], data['image_size'],
        block_size=8, is_luma=True
    )

    print("3. Восстановление цветовых каналов (Cb, Cr)...")
    Cb_restored = upsample_channel(
        inverse_process_channel(
            data['Cb_dc_bits'], data['Cb_ac_bits'],
            data['chroma_quant_matrix'],
            (data['image_size'][0] // 2, data['image_size'][1] // 2),
            block_size=8, is_luma=False
        ),
        data['image_size']
    )

    Cr_restored = upsample_channel(
        inverse_process_channel(
            data['Cr_dc_bits'], data['Cr_ac_bits'],
            data['chroma_quant_matrix'],
            (data['image_size'][0] // 2, data['image_size'][1] // 2),
            block_size=8, is_luma=False
        ),
        data['image_size']
    )

    # 4. Конвертация в RGB
    print("4. Конвертация YCbCr -> RGB...")
    rgb_restored = ycbcr_to_rgb(
        np.clip(Y_restored, 0, 255).astype(np.uint8),
        np.clip(Cb_restored, 0, 255).astype(np.uint8),
        np.clip(Cr_restored, 0, 255).astype(np.uint8)
    )

    # 5. Сохранение или отображение
    if output_image_path:
        Image.fromarray(rgb_restored).save(output_image_path)
        print(f"Изображение сохранено в {output_image_path}")

    return rgb_restored


def show_image_comparison(original, restored, original_size, compressed_size, quality):
    """Показывает сравнение оригинального и восстановленного изображения"""
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title(f"Original\n{original_size / 1024:.2f} KB")

    plt.subplot(1, 2, 2)
    plt.imshow(restored)
    plt.title(f"Restored (Q={quality})\n{compressed_size / 1024:.2f} KB ({original_size / compressed_size:.2f}x)")

    plt.show()


if __name__ == "__main__":
    # Пример использования для Lenna
    input_image = "C:/Users/79508/Desktop/4 семестри/АИСД/2 лабораторная/коди/images/Lenna.png"
    compressed_file = "C:/Users/79508/Desktop/4 семестри/АИСД/2 лабораторная/коди/images/compressed_Lenna.bin"
    output_image = "C:/Users/79508/Desktop/4 семестри/АИСД/2 лабораторная/коди/images/decompressed_Lenna.png"

    print("=== Сжатие изображения Lenna ===")
    compress_to_file(input_image, compressed_file, quality=75)

    print("\n=== Декомпрессия изображения Lenna ===")
    restored = decompress_from_file(compressed_file, output_image)

    original = np.array(Image.open(input_image).convert("RGB"))
    show_image_comparison(original, restored,
                          original.nbytes,
                          os.path.getsize(compressed_file),
                          quality=75)

    # Пример использования для Lake
    input_image2 = "C:/Users/79508/Desktop/4 семестри/АИСД/2 лабораторная/коди/images/lake.jpg"
    compressed_file2 = "C:/Users/79508/Desktop/4 семестри/АИСД/2 лабораторная/коди/images/compressed_lake.bin"
    output_image2 = "C:/Users/79508/Desktop/4 семестри/АИСД/2 лабораторная/коди/images/decompressed_lake.png"

    print("\n=== Сжатие изображения Lake ===")
    compress_to_file(input_image2, compressed_file2, quality=75)

    print("\n=== Декомпрессия изображения Lake ===")
    restored2 = decompress_from_file(compressed_file2, output_image2)

    original2 = np.array(Image.open(input_image2).convert("RGB"))
    show_image_comparison(original2, restored2,
                          original2.nbytes,
                          os.path.getsize(compressed_file2),
                          quality=75)
