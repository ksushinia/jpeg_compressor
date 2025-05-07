import numpy as np
from huffman_table import STD_LUMA_AC_HUFFMAN, STD_CHROMA_AC_HUFFMAN
from rle import rle_encode, rle_decode


def get_category(value):
    """
    Исправленная версия с преобразованием типов
    """
    # Преобразуем в целое число с округлением
    int_value = int(round(float(value)))

    if int_value == 0:
        return 0, ""

    abs_value = abs(int_value)
    category = int(np.ceil(np.log2(abs_value + 1)))

    if int_value > 0:
        amplitude = int_value
    else:
        amplitude = int_value + (1 << category) - 1

    amplitude_bits = bin(amplitude)[2:].zfill(category)
    return category, amplitude_bits


def run_length_encode(ac_coeffs):
    """Выполняет RLE кодирование для AC коэффициентов"""
    rle_pairs = []
    zero_run = 0

    for coeff in ac_coeffs:
        if coeff == 0:
            zero_run += 1
        else:
            rle_pairs.append((zero_run, coeff))
            zero_run = 0

    # Добавляем EOB (End Of Block) если есть оставшиеся нули
    if zero_run > 0:
        rle_pairs.append((0, 0))  # EOB

    return rle_pairs


def encode_ac_coefficients(ac_coeffs, is_luma=True):
    """
    Исправленная версия с обработкой float значений
    """
    huffman_table = STD_LUMA_AC_HUFFMAN if is_luma else STD_CHROMA_AC_HUFFMAN
    bitstream = ""

    # Преобразуем в целые числа
    ac_coeffs = [int(round(float(x))) for x in ac_coeffs]

    # Применяем RLE к AC коэффициентам
    rle_pairs = []
    zero_run = 0
    i = 0
    n = len(ac_coeffs)

    while i < n:
        if ac_coeffs[i] == 0:
            zero_run += 1
            i += 1
            # Если это последний элемент, добавляем EOB
            if i == n:
                rle_pairs.append((0, 0))  # EOB
        else:
            # Добавляем пару (run_length, value)
            rle_pairs.append((zero_run, ac_coeffs[i]))
            zero_run = 0
            i += 1

    # Кодируем RLE-пары
    for run_length, value in rle_pairs:
        if value == 0 and run_length == 0:  # EOB
            bitstream += huffman_table.get((0, 0), "")
            continue

        # Обработка длинных последовательностей нулей (ZRL)
        while run_length >= 16:
            bitstream += huffman_table.get((15, 0), "")
            run_length -= 16

        # Кодируем оставшиеся нули и значение
        category, amplitude_bits = get_category(value)
        if (run_length, category) in huffman_table:
            bitstream += huffman_table[(run_length, category)] + amplitude_bits
        else:
            print(f"Warning: Missing entry for {(run_length, category)}")
            continue

    return bitstream


def decode_ac_coefficients(bitstream, block_size=63, is_luma=True):
    """Декодирование AC-коэффициентов"""
    huffman_table = STD_LUMA_AC_HUFFMAN if is_luma else STD_CHROMA_AC_HUFFMAN
    reverse_huffman = {v: k for k, v in huffman_table.items()}

    ac_coeffs = np.zeros(block_size, dtype=int)
    current_pos = 0
    coeff_index = 0

    while coeff_index < block_size and current_pos < len(bitstream):
        # Поиск кода Хаффмана
        found = False
        for code_len in range(1, 32):
            if current_pos + code_len > len(bitstream):
                break
            code = bitstream[current_pos:current_pos + code_len]
            if code in reverse_huffman:
                run_length, category = reverse_huffman[code]
                current_pos += code_len
                found = True
                break

        if not found:
            raise ValueError("Invalid Huffman code")

        # Обработка специальных случаев
        if run_length == 0 and category == 0:  # EOB
            break
        elif run_length == 15 and category == 0:  # ZRL
            zeros_to_add = min(16, block_size - coeff_index)
            coeff_index += zeros_to_add
            continue

        # Декодирование значения
        value = 0
        if category > 0:
            if current_pos + category > len(bitstream):
                raise ValueError("Not enough bits")

            amplitude_bits = bitstream[current_pos:current_pos + category]
            current_pos += category
            amplitude = int(amplitude_bits, 2)

            if amplitude < (1 << (category - 1)):
                value = amplitude - (1 << category) + 1
            else:
                value = amplitude

        # Заполняем нулями и значение
        if run_length > 0:
            zero_end = min(coeff_index + run_length, block_size)
            ac_coeffs[coeff_index:zero_end] = 0
            coeff_index = zero_end

        if coeff_index < block_size and category > 0:
            ac_coeffs[coeff_index] = value
            coeff_index += 1

    return ac_coeffs
