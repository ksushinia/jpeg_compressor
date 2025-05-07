import numpy as np
import matplotlib.pyplot as plt
from functions.quantization import get_quantization_matrix


def test_quantization():
    # Тестируем разные уровни качества
    qualities = [10, 20, 30, 50, 70, 90]

    plt.figure(figsize=(15, 8))

    for i, quality in enumerate(qualities, 1):
        quant_matrix = get_quantization_matrix(quality)

        plt.subplot(2, 3, i)
        plt.imshow(quant_matrix, cmap='hot', interpolation='nearest')
        plt.title(f"Quality: {quality}")
        plt.colorbar()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_quantization()