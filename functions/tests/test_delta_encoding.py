import numpy as np
from functions.delta_encoding import (
    extract_dc_coefficients,
    delta_encode_dc,
    delta_decode_dc,
    update_blocks_with_dc
)


def test_basic_case():
    """Тест базового случая (положительные значения)"""
    print("\n=== Тест 1: Базовый случай ===")
    quant_blocks = np.array([
        [[[5, 1], [2, 3]]],
        [[[8, 0], [1, 2]]]
    ])
    dc = extract_dc_coefficients(quant_blocks)
    delta = delta_encode_dc(dc)
    restored = delta_decode_dc(delta)

    print("DC:", dc, "-> Дельта:", delta, "-> Восстановленные:", restored)
    assert np.array_equal(dc, restored), "Ошибка в базовом случае!"


def test_negative_values():
    """Тест с отрицательными DC-коэффициентами"""
    print("\n=== Тест 2: Отрицательные значения ===")
    quant_blocks = np.array([
        [[[-3, 1], [2, 3]]],
        [[[4, 0], [-1, 2]]]
    ])
    dc = extract_dc_coefficients(quant_blocks)
    delta = delta_encode_dc(dc)
    restored = delta_decode_dc(delta)

    print("DC:", dc, "-> Дельта:", delta, "-> Восстановленные:", restored)
    assert np.array_equal(dc, restored), "Ошибка с отрицательными значениями!"


def test_single_block():
    """Тест с одним блоком"""
    print("\n=== Тест 3: Один блок ===")
    quant_blocks = np.array([[[[10, 2], [3, 4]]]])
    dc = extract_dc_coefficients(quant_blocks)
    delta = delta_encode_dc(dc)
    restored = delta_decode_dc(delta)

    print("DC:", dc, "-> Дельта:", delta, "-> Восстановленные:", restored)
    assert delta[0] == dc[0], "Ошибка с одним блоком!"
    assert len(delta) == 1, "Дельта должна содержать 1 элемент!"


def test_large_blocks():
    """Тест с большими блоками (8x8)"""
    print("\n=== Тест 4: Блоки 8x8 ===")
    np.random.seed(42)
    quant_blocks = np.random.randint(-50, 50, size=(2, 2, 8, 8))
    # Фиксируем DC-коэффициенты для предсказуемости
    quant_blocks[0, 0, 0, 0] = 100
    quant_blocks[0, 1, 0, 0] = 105
    quant_blocks[1, 0, 0, 0] = 95
    quant_blocks[1, 1, 0, 0] = 110

    dc = extract_dc_coefficients(quant_blocks)
    delta = delta_encode_dc(dc)
    restored = delta_decode_dc(delta)
    updated_blocks = update_blocks_with_dc(quant_blocks, restored)

    print("DC:", dc, "-> Дельта:", delta)
    assert np.array_equal(dc, restored), "Ошибка с блоками 8x8!"
    assert updated_blocks[0, 0, 0, 0] == 100, "Некорректное обновление блока!"


def test_zero_deltas():
    """Тест с одинаковыми DC-коэффициентами (дельта=0)"""
    print("\n=== Тест 5: Нулевые дельты ===")
    quant_blocks = np.array([
        [[[5, 1], [2, 3]]],
        [[[5, 0], [1, 2]]]
    ])
    dc = extract_dc_coefficients(quant_blocks)
    delta = delta_encode_dc(dc)

    print("DC:", dc, "-> Дельта:", delta)
    assert delta[1] == 0, "Дельта должна быть 0 для одинаковых DC!"


def test_update_blocks():
    """Тест корректности обновления блоков"""
    print("\n=== Тест 6: Обновление блоков ===")
    quant_blocks = np.array([
        [[[1, 0], [0, 0]]],
        [[[2, 0], [0, 0]]]
    ])
    new_dc = np.array([10, 20])
    updated = update_blocks_with_dc(quant_blocks, new_dc)

    print("Обновленные блоки:")
    print(updated)
    assert updated[0, 0, 0, 0] == 10, "Ошибка обновления первого блока!"
    assert updated[1, 0, 0, 0] == 20, "Ошибка обновления второго блока!"
    assert np.array_equal(updated[0, 0, 0, 1:], quant_blocks[0, 0, 0, 1:]), "Изменились не-DC коэффициенты!"


if __name__ == "__main__":
    print("=== Запуск расширенных тестов delta_encoding ===")
    test_basic_case()
    test_negative_values()
    test_single_block()
    test_large_blocks()
    test_zero_deltas()
    test_update_blocks()
    print("\nВсе тесты пройдены успешно! ✅")