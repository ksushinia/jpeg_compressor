import matplotlib.pyplot as plt
import numpy as np


# Предположим, у тебя есть массивы Y, Cb, Cr одинакового размера (H x W)
# Например:
# Y, Cb, Cr = rgb_to_ycbcr(your_rgb_image)

def show_ycbcr_channels(Y, Cb, Cr):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Y channel (Luminance)')
    plt.imshow(Y, cmap='gray')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title('Cb channel (Chroma Blue)')
    plt.imshow(Cb, cmap='gray')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title('Cr channel (Chroma Red)')
    plt.imshow(Cr, cmap='gray')
    plt.colorbar()

    plt.show()

# Пример вызова
# show_ycbcr_channels(Y, Cb, Cr)
