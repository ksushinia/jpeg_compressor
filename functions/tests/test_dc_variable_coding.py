import numpy as np
from functions.dc_variable_coding import encode_dc_coefficients, decode_dc_coefficients


def print_debug_info(input_data, bitstream, decoded, expected):
    """Печатает отладочную информацию для теста"""
    print("\n=== Debug Info ===")
    print(f"Input: {input_data}")
    print(f"Bitstream: {bitstream[:100]}... (len={len(bitstream)})")
    print(f"Decoded: {decoded}")
    print(f"Expected: {expected}")

    if not np.array_equal(decoded, expected):
        print("\nMismatches:")
        for i, (d, e) in enumerate(zip(decoded, expected)):
            if d != e:
                print(f"Index {i}: decoded={d}, expected={e}")


def test_dc_basic():
    print("\n=== Running test_dc_basic ===")
    dc_input = np.array([5, 7, 10, 12, 15])
    bitstream = encode_dc_coefficients(dc_input, is_luma=True)
    decoded = decode_dc_coefficients(bitstream, len(dc_input), is_luma=True)

    print_debug_info(dc_input, bitstream, decoded, dc_input)
    assert np.array_equal(decoded, dc_input), "Basic DC test failed"


def test_dc_negative_deltas():
    print("\n=== Running test_dc_negative_deltas ===")
    dc_input = np.array([20, 15, 10, 5, 0])
    bitstream = encode_dc_coefficients(dc_input, is_luma=False)
    decoded = decode_dc_coefficients(bitstream, len(dc_input), is_luma=False)

    print_debug_info(dc_input, bitstream, decoded, dc_input)
    assert np.array_equal(decoded, dc_input), "Negative deltas test failed"


def test_dc_zeros():
    print("\n=== Running test_dc_zeros ===")
    dc_input = np.array([0, 0, 0, 0, 0])
    bitstream = encode_dc_coefficients(dc_input, is_luma=True)
    decoded = decode_dc_coefficients(bitstream, len(dc_input), is_luma=True)

    print_debug_info(dc_input, bitstream, decoded, dc_input)
    assert np.array_equal(decoded, dc_input), "All zeros test failed"


def test_dc_large_values():
    print("\n=== Running test_dc_large_values ===")
    dc_input = np.array([1000, 1050, 1100, 900, 950])
    bitstream = encode_dc_coefficients(dc_input, is_luma=False)
    decoded = decode_dc_coefficients(bitstream, len(dc_input), is_luma=False)

    print_debug_info(dc_input, bitstream, decoded, dc_input)
    assert np.array_equal(decoded, dc_input), "Large values test failed"


def test_dc_mixed():
    print("\n=== Running test_dc_mixed ===")
    dc_input = np.array([-5, 3, -2, 0, 1, -1, 0])
    bitstream = encode_dc_coefficients(dc_input, is_luma=True)
    decoded = decode_dc_coefficients(bitstream, len(dc_input), is_luma=True)

    print_debug_info(dc_input, bitstream, decoded, dc_input)
    assert np.array_equal(decoded, dc_input), "Mixed values test failed"


def test_dc_single_block():
    print("\n=== Running test_dc_single_block ===")
    dc_input = np.array([10])
    bitstream = encode_dc_coefficients(dc_input, is_luma=False)
    decoded = decode_dc_coefficients(bitstream, len(dc_input), is_luma=False)

    print_debug_info(dc_input, bitstream, decoded, dc_input)
    assert np.array_equal(decoded, dc_input), "Single block test failed"


def test_dc_chroma_luma_diff():
    print("\n=== Running test_dc_chroma_luma_diff ===")
    dc_input = np.array([10, 20, 30])

    # Тестируем разницу между luma и chroma таблицами
    bitstream_luma = encode_dc_coefficients(dc_input, is_luma=True)
    bitstream_chroma = encode_dc_coefficients(dc_input, is_luma=False)

    assert bitstream_luma != bitstream_chroma, "Luma and Chroma should use different tables"

    decoded_luma = decode_dc_coefficients(bitstream_luma, len(dc_input), is_luma=True)
    decoded_chroma = decode_dc_coefficients(bitstream_chroma, len(dc_input), is_luma=False)

    assert np.array_equal(decoded_luma, dc_input), "Luma decoding failed"
    assert np.array_equal(decoded_chroma, dc_input), "Chroma decoding failed"
    print("Luma and Chroma tables work correctly")


def run_all_dc_tests():
    print("=" * 50)
    print("Starting DC coefficients tests")
    print("=" * 50)

    tests = [
        test_dc_basic,
        test_dc_negative_deltas,
        test_dc_zeros,
        test_dc_large_values,
        test_dc_mixed,
        test_dc_single_block,
        test_dc_chroma_luma_diff
    ]

    for test in tests:
        try:
            test()
            print(f"{test.__name__}: PASSED")
        except AssertionError as e:
            print(f"{test.__name__}: FAILED - {str(e)}")

    print("\nAll DC tests completed!")


if __name__ == "__main__":
    run_all_dc_tests()