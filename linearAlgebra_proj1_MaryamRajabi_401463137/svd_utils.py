import numpy as np


# -----------------------------
# Compress one channel using SVD
# -----------------------------
def compress_channel(channel, k):

    U, S, VT = np.linalg.svd(channel, full_matrices=False)

    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]

    compressed = U_k @ S_k @ VT_k

    compressed = np.clip(compressed, 0, 255)

    return compressed.astype(np.uint8)


# -----------------------------
# Compress RGB or Gray image
# -----------------------------
def compress_image(image, k):

    # Gray Image
    if len(image.shape) == 2:
        return compress_channel(image, k)

    # RGB Image
    r = compress_channel(image[:, :, 0], k)
    g = compress_channel(image[:, :, 1], k)
    b = compress_channel(image[:, :, 2], k)

    return np.dstack((r, g, b))


# -----------------------------
# PSNR
# -----------------------------
def calculate_psnr(original, compressed):

    mse = np.mean((original.astype(np.float64) -
                   compressed.astype(np.float64)) ** 2)

    if mse == 0:
        return 100

    PIXEL_MAX = 255.0

    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    return psnr


# -----------------------------
# Compression Ratio (Optional)
# -----------------------------
def compression_ratio(k, rows, cols):

    original = rows * cols

    compressed = k * (rows + cols + 1)

    return compressed / original


# -----------------------------
# Estimated Saved Percentage
# -----------------------------
def saved_percentage(k, rows, cols):

    ratio = compression_ratio(k, rows, cols)

    return (1 - ratio) * 100