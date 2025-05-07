import numpy as np
import matplotlib.pyplot as plt


def downsample_channel(channel: np.ndarray, ratio: int = 2) -> np.ndarray:
    """
    Уменьшает разрешение цветового канала (Cb или Cr) в ratio раз.
    """
    h, w = channel.shape
    h_new = h // ratio
    w_new = w // ratio
    downsampled = channel[:h_new * ratio, :w_new * ratio]
    downsampled = downsampled.reshape(h_new, ratio, w_new, ratio).mean(axis=(1, 3))
    return downsampled


def visualize_channels(Y, Cb, Cr, Cb_down, Cr_down):
    """
    Визуализирует оригинальные и даунсэмплированные каналы
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.imshow(Y, cmap='gray')
    plt.title("Y (Luma)")

    plt.subplot(2, 3, 2)
    plt.imshow(Cb, cmap='gray')
    plt.title("Cb (Original)")

    plt.subplot(2, 3, 3)
    plt.imshow(Cr, cmap='gray')
    plt.title("Cr (Original)")

    plt.subplot(2, 3, 5)
    plt.imshow(Cb_down, cmap='gray')
    plt.title("Cb (Downsampled)")

    plt.subplot(2, 3, 6)
    plt.imshow(Cr_down, cmap='gray')
    plt.title("Cr (Downsampled)")

    plt.tight_layout()
    plt.show()