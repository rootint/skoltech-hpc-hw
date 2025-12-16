import numpy as np
from numba import njit, prange


@njit(parallel=True)
def kuwahara_numba(img, radius):
    height, width, channels = img.shape

    output = np.zeros_like(img)

    v_img = np.zeros((height, width), dtype=np.float64)

    for i in prange(height):
        for j in range(width):

            r = img[i, j, 0]
            g = img[i, j, 1]
            b = img[i, j, 2]

            max_val = r
            if g > max_val:
                max_val = g
            if b > max_val:
                max_val = b

            v_img[i, j] = max_val

    for y in prange(radius, height - radius):
        for x in range(radius, width - radius):

            y_start_tl, y_end_tl = y - radius, y + 1
            x_start_tl, x_end_tl = x - radius, x + 1

            y_start_tr, y_end_tr = y - radius, y + 1
            x_start_tr, x_end_tr = x, x + radius + 1

            y_start_bl, y_end_bl = y, y + radius + 1
            x_start_bl, x_end_bl = x - radius, x + 1

            y_start_br, y_end_br = y, y + radius + 1
            x_start_br, x_end_br = x, x + radius + 1

            ranges = (
                (y_start_tl, y_end_tl, x_start_tl, x_end_tl),
                (y_start_tr, y_end_tr, x_start_tr, x_end_tr),
                (y_start_bl, y_end_bl, x_start_bl, x_end_bl),
                (y_start_br, y_end_br, x_start_br, x_end_br),
            )

            min_variance = 1e20
            best_mean = np.zeros(3, dtype=np.float64)

            for r_idx in range(4):
                ys, ye, xs, xe = ranges[r_idx]

                v_slice = v_img[ys:ye, xs:xe]
                current_var = np.var(v_slice)

                if current_var < min_variance:
                    min_variance = current_var

                    r_mean = np.mean(img[ys:ye, xs:xe, 0])
                    g_mean = np.mean(img[ys:ye, xs:xe, 1])
                    b_mean = np.mean(img[ys:ye, xs:xe, 2])

                    best_mean[0] = r_mean
                    best_mean[1] = g_mean
                    best_mean[2] = b_mean

            output[y, x, 0] = best_mean[0]
            output[y, x, 1] = best_mean[1]
            output[y, x, 2] = best_mean[2]

    return output
