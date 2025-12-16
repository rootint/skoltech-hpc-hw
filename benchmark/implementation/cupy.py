import cupy as cp
from cupyx.scipy.ndimage import uniform_filter


def kuwahara_cupy(img_np, radius):
    img = cp.array(img_np, dtype=cp.float32)

    v = img.max(axis=2)
    v_sq = v**2

    w_size = radius + 1

    shift = radius // 2

    offsets = [(-shift, -shift), (-shift, shift), (shift, -shift), (shift, shift)]

    variances = []
    means_rgb = []

    for off_y, off_x in offsets:

        mean_v = uniform_filter(v, size=w_size, origin=(off_y, off_x), mode="reflect")
        mean_v2 = uniform_filter(
            v_sq, size=w_size, origin=(off_y, off_x), mode="reflect"
        )

        var = mean_v2 - (mean_v**2)
        variances.append(var)

        mean_rgb_quad = uniform_filter(
            img, size=(w_size, w_size, 1), origin=(off_y, off_x, 0), mode="reflect"
        )
        means_rgb.append(mean_rgb_quad)

    stack_var = cp.stack(variances)
    stack_rgb = cp.stack(means_rgb)

    best_idx = cp.argmin(stack_var, axis=0)

    best_idx_broad = best_idx[..., cp.newaxis]

    result = cp.take_along_axis(stack_rgb, best_idx_broad[cp.newaxis, ...], axis=0)

    result = result[0]

    return cp.asnumpy(result)
