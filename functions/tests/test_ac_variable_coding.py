import numpy as np
from functions.ac_variable_coding import encode_ac_coefficients, decode_ac_coefficients


def print_debug_info(ac_input, bitstream, decoded, expected):
    """Печатает отладочную информацию для теста"""
    print("\n=== Debug Info ===")
    print(f"Input: {ac_input[:10]}... (len={len(ac_input)})")  # Показываем начало длинных массивов
    print(f"Bitstream: {bitstream[:50]}... (len={len(bitstream)})")
    print(f"Decoded: {decoded[:10]}... (len={len(decoded)})")
    print(f"Expected: {expected[:10]}... (len={len(expected)})")

    # Находим расхождения
    diff_indices = np.where(decoded != expected)[0]
    if len(diff_indices) > 0:
        print("\nMismatches at indices:", diff_indices)
        for idx in diff_indices[:5]:  # Показываем первые 5 расхождений
            print(f"Index {idx}: decoded={decoded[idx]}, expected={expected[idx]}")
    else:
        print("\nPerfect match!")


def test_encode_decode_ac_basic():
    print("\n=== Running test_encode_decode_ac_basic ===")
    ac_input = [0, 0, 0, -5, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    bitstream = encode_ac_coefficients(ac_input, is_luma=True)
    decoded = decode_ac_coefficients(bitstream, block_size=63, is_luma=True)
    expected = np.zeros(63, dtype=int)
    expected[:len(ac_input)] = ac_input

    print_debug_info(ac_input, bitstream, decoded, expected)
    assert np.array_equal(decoded, expected), "Basic encoding-decoding failed"


def test_all_zeros():
    print("\n=== Running test_all_zeros ===")
    ac_input = [0] * 63
    bitstream = encode_ac_coefficients(ac_input, is_luma=False)
    decoded = decode_ac_coefficients(bitstream, block_size=63, is_luma=False)
    expected = np.zeros(63, dtype=int)

    print_debug_info(ac_input, bitstream, decoded, expected)
    assert np.array_equal(decoded, expected), "All zeros test failed"


def test_no_zeros():
    print("\n=== Running test_no_zeros ===")
    ac_input = list(range(1, 64))  # [1, 2, 3, ..., 63]
    bitstream = encode_ac_coefficients(ac_input, is_luma=True)
    decoded = decode_ac_coefficients(bitstream, block_size=63, is_luma=True)
    expected = np.array(ac_input, dtype=int)

    print_debug_info(ac_input, bitstream, decoded, expected)
    assert np.array_equal(decoded, expected), "No zeros test failed"


def test_large_leading_zeros():
    print("\n=== Running test_large_leading_zeros ===")
    ac_input = [0] * 50 + [7, -3, 1]
    bitstream = encode_ac_coefficients(ac_input, is_luma=True)
    decoded = decode_ac_coefficients(bitstream, block_size=63, is_luma=True)
    expected = np.zeros(63, dtype=int)
    expected[:len(ac_input)] = ac_input

    print_debug_info(ac_input, bitstream, decoded, expected)
    assert np.array_equal(decoded, expected), "Large leading zeros test failed"


def test_random_pattern():
    print("\n=== Running test_random_pattern ===")
    ac_input = [0, 4, 0, 0, -1, 0, 2, 0, 0, 0, 0, 3, -3, 0, 0, 1]
    bitstream = encode_ac_coefficients(ac_input, is_luma=False)
    decoded = decode_ac_coefficients(bitstream, block_size=63, is_luma=False)
    expected = np.zeros(63, dtype=int)
    expected[:len(ac_input)] = ac_input

    print_debug_info(ac_input, bitstream, decoded, expected)
    assert np.array_equal(decoded, expected), "Random pattern test failed"


def test_single_nonzero_start():
    print("\n=== Running test_single_nonzero_start ===")
    ac_input = [5] + [0] * 62
    bitstream = encode_ac_coefficients(ac_input, is_luma=True)
    decoded = decode_ac_coefficients(bitstream, block_size=63, is_luma=True)
    expected = np.array(ac_input, dtype=int)

    print_debug_info(ac_input, bitstream, decoded, expected)
    assert np.array_equal(decoded, expected), "Single nonzero start test failed"


def test_multiple_zrls():
    print("\n=== Running test_multiple_zrls ===")
    ac_input = [0] * 32 + [1, 0] * 15 + [2]
    bitstream = encode_ac_coefficients(ac_input, is_luma=False)
    decoded = decode_ac_coefficients(bitstream, block_size=63, is_luma=False)
    expected = np.zeros(63, dtype=int)
    expected[:len(ac_input)] = ac_input

    print_debug_info(ac_input, bitstream, decoded, expected)
    assert np.array_equal(decoded, expected), "Multiple ZRLs test failed"


def test_eob_early():
    print("\n=== Running test_eob_early ===")
    ac_input = [1, 2, 3, 0, 0, 0]  # EOB после 3 элементов
    bitstream = encode_ac_coefficients(ac_input, is_luma=True)
    decoded = decode_ac_coefficients(bitstream, block_size=63, is_luma=True)
    expected = np.zeros(63, dtype=int)
    expected[:len(ac_input)] = ac_input

    print_debug_info(ac_input, bitstream, decoded, expected)
    assert np.array_equal(decoded, expected), "Early EOB test failed"