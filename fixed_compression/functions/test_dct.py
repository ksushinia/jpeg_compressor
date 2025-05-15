import numpy as np
from dct import dct2,dct_matrix,idct2

# ==== ТЕСТ ====

# Тестовый блок 8x8
block = np.array([
    [52, 55, 61, 66, 70, 61, 64, 73],
    [63, 59, 55, 90, 109, 85, 69, 72],
    [62, 59, 68, 113, 144, 104, 66, 73],
    [63, 58, 71, 122, 154, 106, 70, 69],
    [67, 61, 68, 104, 126, 88, 68, 70],
    [79, 65, 60, 70, 77, 68, 58, 75],
    [85, 71, 64, 59, 55, 61, 65, 83],
    [87, 79, 69, 68, 65, 76, 78, 94]
], dtype=np.float32)

# DCT → IDCT
dct_block = dct2(block)
reconstructed = idct2(dct_block)

# Разница
diff = np.abs(block - reconstructed)

# Вывод результатов
print("Исходный блок:\n", block)
print("\nDCT-коэффициенты:\n", np.round(dct_block, 2))
print("\nВосстановленный блок:\n", np.round(reconstructed, 2))
print("\nРазница:\n", np.round(diff, 6))
