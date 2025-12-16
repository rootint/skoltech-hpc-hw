import numpy as np
from numba import njit, prange


@njit(parallel=True)
def kuwahara_numba(img, radius):
    """
    Applies the Kuwahara filter to a color image.

    Args:
        img: Numpy array of shape (H, W, 3) - expected to be floats or ints
        radius: The size of the window (integer). Higher = more abstract.

    Returns:
        The filtered image.
    """
    h, w, c = img.shape

    output = np.zeros_like(img)

    for y in prange(radius, h - radius):
        for x in range(radius, w - radius):

            tl = img[y - radius : y + 1, x - radius : x + 1]

            tr = img[y - radius : y + 1, x : x + radius + 1]

            bl = img[y : y + radius + 1, x - radius : x + 1]

            br = img[y : y + radius + 1, x : x + radius + 1]

            regions = (tl, tr, bl, br)

            min_variance = 1e99
            best_mean = np.zeros(3, dtype=np.float64)

            for region in regions:

                mean_val = np.array(
                    [
                        np.mean(region[:, :, 0]),
                        np.mean(region[:, :, 1]),
                        np.mean(region[:, :, 2]),
                    ]
                )

                var_val = (
                    np.var(region[:, :, 0])
                    + np.var(region[:, :, 1])
                    + np.var(region[:, :, 2])
                )

                if var_val < min_variance:
                    min_variance = var_val
                    best_mean = mean_val

            output[y, x, 0] = best_mean[0]
            output[y, x, 1] = best_mean[1]
            output[y, x, 2] = best_mean[2]

    return output
