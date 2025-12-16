import numpy as np


def kuwahara_naive(image, radius):
    height, width = image.shape[:2]
    output = np.zeros_like(image)

    v_channel = np.zeros((height, width), dtype=image.dtype)
    for i in range(height):
        for j in range(width):
            r = image[i, j, 0]
            g = image[i, j, 1]
            b = image[i, j, 2]
            v_channel[i, j] = max(r, g, b)

    print(f"Processing image {width}x{height} with radius={radius}...")

    # Double loop over all pixels
    for y in range(height):
        if y % 100 == 0:
            print(f"  Progress: {y}/{height} rows")

        for x in range(width):
            quadrants = [
                # Q1: bottom-right
                (
                    max(0, y),
                    min(height, y + radius + 1),
                    max(0, x),
                    min(width, x + radius + 1),
                ),
                # Q2: bottom-left
                (
                    max(0, y),
                    min(height, y + radius + 1),
                    max(0, x - radius),
                    min(width, x + 1),
                ),
                # Q3: top-left
                (
                    max(0, y - radius),
                    min(height, y + 1),
                    max(0, x - radius),
                    min(width, x + 1),
                ),
                # Q4: top-right
                (
                    max(0, y - radius),
                    min(height, y + 1),
                    max(0, x),
                    min(width, x + radius + 1),
                ),
            ]

            min_variance = float("inf")
            best_mean = np.zeros(3, dtype=image.dtype)

            for y_start, y_end, x_start, x_end in quadrants:

                v_region = v_channel[y_start:y_end, x_start:x_end]

                variance = np.var(v_region)

                if variance < min_variance:
                    min_variance = variance

                    rgb_region = image[y_start:y_end, x_start:x_end]

                    best_mean[0] = np.mean(rgb_region[:, :, 0])
                    best_mean[1] = np.mean(rgb_region[:, :, 1])
                    best_mean[2] = np.mean(rgb_region[:, :, 2])

            output[y, x] = best_mean

    return output
