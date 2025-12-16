import numpy as np
from numba import njit, prange

@njit(parallel=True)
def kuwahara_numba(img, radius):
    """
    Numba implementation using HSV Value channel for variance.
    
    Args:
        img: Float64 Numpy array (H, W, 3). 
             Ensure image is normalized 0.0-1.0 or 0-255.
        radius: Window radius (int).
    """
    height, width, channels = img.shape
    
    # 1. Prepare Output
    output = np.zeros_like(img)
    
    # 2. Generate 'Value' Channel (Brightness)
    #    In HSV, V = max(R, G, B).
    #    We manually loop to create this because Numba handles explicit loops 
    #    often faster than array syntax like np.max(img, axis=2)
    v_img = np.zeros((height, width), dtype=np.float64)
    
    for i in prange(height):
        for j in range(width):
            # Find max of r, g, b manually
            r = img[i, j, 0]
            g = img[i, j, 1]
            b = img[i, j, 2]
            
            # Simple max logic
            max_val = r
            if g > max_val:
                max_val = g
            if b > max_val:
                max_val = b
            
            v_img[i, j] = max_val

    # 3. The Filter Loop
    #    We iterate through every pixel. 'prange' parallelizes the rows.
    for y in prange(radius, height - radius):
        for x in range(radius, width - radius):
            
            # --- DEFINE RANGES ---
            # We explicitly define start/end points to avoid creating 
            # 'slice' objects, which Numba handles but integers are cleaner.
            
            # Top-Left (y-r -> y+1, x-r -> x+1)
            y_start_tl, y_end_tl = y - radius, y + 1
            x_start_tl, x_end_tl = x - radius, x + 1
            
            # Top-Right (y-r -> y+1, x -> x+r+1)
            y_start_tr, y_end_tr = y - radius, y + 1
            x_start_tr, x_end_tr = x, x + radius + 1
            
            # Bottom-Left (y -> y+r+1, x-r -> x+1)
            y_start_bl, y_end_bl = y, y + radius + 1
            x_start_bl, x_end_bl = x - radius, x + 1
            
            # Bottom-Right (y -> y+r+1, x -> x+r+1)
            y_start_br, y_end_br = y, y + radius + 1
            x_start_br, x_end_br = x, x + radius + 1
            
            # Store ranges in a tuple so we can loop over them
            # (y_start, y_end, x_start, x_end)
            ranges = (
                (y_start_tl, y_end_tl, x_start_tl, x_end_tl),
                (y_start_tr, y_end_tr, x_start_tr, x_end_tr),
                (y_start_bl, y_end_bl, x_start_bl, x_end_bl),
                (y_start_br, y_end_br, x_start_br, x_end_br)
            )
            
            min_variance = 1e20 # Arbitrary large number
            best_mean = np.zeros(3, dtype=np.float64)
            
            # --- CHECK QUADRANTS ---
            for r_idx in range(4):
                ys, ye, xs, xe = ranges[r_idx]
                
                # A. Check Variance on the V_IMG (Brightness)
                #    Numba is smart enough to optimize this slice + np.var
                v_slice = v_img[ys:ye, xs:xe]
                current_var = np.var(v_slice)
                
                # B. Track Winner
                if current_var < min_variance:
                    min_variance = current_var
                    
                    # C. Calculate Mean on the COLOR IMAGE
                    #    Only done if this is currently the best candidate (or at the end).
                    #    However, in this structure, we have to calculate it to save it.
                    #    (Optimizing this "lazy evaluation" is a future step).
                    
                    # Note: We cannot use axis=(0,1) in Numba easily for 3D arrays 
                    # in older versions, so we calculate mean per channel manually 
                    # or use 3 np.mean calls.
                    r_mean = np.mean(img[ys:ye, xs:xe, 0])
                    g_mean = np.mean(img[ys:ye, xs:xe, 1])
                    b_mean = np.mean(img[ys:ye, xs:xe, 2])
                    
                    best_mean[0] = r_mean
                    best_mean[1] = g_mean
                    best_mean[2] = b_mean
            
            # 4. Assign to Output
            output[y, x, 0] = best_mean[0]
            output[y, x, 1] = best_mean[1]
            output[y, x, 2] = best_mean[2]

    return output