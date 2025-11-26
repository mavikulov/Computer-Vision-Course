import numpy as np

# Read the implementation of the align_image function in pipeline.py
# to see, how these functions will be used for image alignment.


def extract_channel_plates(raw_img, crop):
    H, W = raw_img.shape
    channel_height = H // 3
    new_raw_img = raw_img[:channel_height * 3, :]  

    blue_channel = new_raw_img[:channel_height, :]
    green_channel = new_raw_img[channel_height:2 * channel_height, :]
    red_channel = new_raw_img[2 * channel_height:3 * channel_height, :]

    blue_coords = np.array([0, 0])
    green_coords = np.array([channel_height, 0])
    red_coords = np.array([2 * channel_height, 0])

    if crop:
        crop_margin_h = int(0.1 * channel_height) 
        crop_margin_w = int(0.1 * W)          

        blue_channel = blue_channel[crop_margin_h:-crop_margin_h, crop_margin_w:-crop_margin_w]
        green_channel = green_channel[crop_margin_h:-crop_margin_h, crop_margin_w:-crop_margin_w]
        red_channel = red_channel[crop_margin_h:-crop_margin_h, crop_margin_w:-crop_margin_w]
    
        blue_coords += np.array([crop_margin_h, crop_margin_w])
        green_coords += np.array([crop_margin_h, crop_margin_w])
        red_coords += np.array([crop_margin_h, crop_margin_w])

    unaligned_rgb = (red_channel, green_channel, blue_channel)
    coords = (red_coords, green_coords, blue_coords)

    return unaligned_rgb, coords


def MSE(I_1, I_2):
    return np.mean((I_1 - I_2) ** 2)


def cross_corelation(I_1, I_2):
    return np.sum(I_1 * I_2) / np.sqrt(np.sum(I_1 ** 2) * np.sum(I_2 ** 2))


def build_image_pyramid(img, min_size):
    pyramid = [img]
    sampled_img = img
    while sampled_img.shape[0] > min_size and sampled_img.shape[1] > min_size:
        sampled_img = downsample(sampled_img)
        pyramid.append(sampled_img)
    
    return pyramid


def downsample(img):
    H, W = img.shape
    H -= H % 2
    W -= W % 2
    return (img[0:H:2, 0:W:2] + img[1:H:2, 0:W:2] + img[0:H:2, 1:W:2] + img[1:H:2, 1:W:2]) / 4.0


def normalize_image(img):
    img_mean = np.mean(img)
    img_std = np.clip(np.std(img), a_min=1e-9, a_max=None)
    return (img - img_mean) / img_std


def get_best_shifts(img_a, img_b, range_x, range_y, metric):
    best_score = -float('inf') if metric == 'ncc' else float('inf')
    best_shift = (0, 0)
    
    for dx in range(range_x[0], range_x[1]):
        for dy in range(range_y[0], range_y[1]):
            shifted_img_b = np.roll(img_b, (dy, dx), axis=(0, 1))
            
            if metric == 'mse':
                score = MSE(img_a, shifted_img_b)
                if score < best_score:
                    best_score = score
                    best_shift = (dx, dy)
            elif metric == 'ncc':
                score = cross_corelation(img_a, shifted_img_b)
                if score > best_score:
                    best_score = score
                    best_shift = (dx, dy)

    return best_shift


def find_relative_shift_pyramid(img_a, img_b, metric='mse', min_size=500, max_shift=14):
    ref_pyramid = build_image_pyramid(img_a, min_size)
    mov_pyramid = build_image_pyramid(img_b, min_size)

    x_shift, y_shift = 0, 0
    pyramid_levels = len(ref_pyramid)
    
    for level in reversed(range(pyramid_levels)):
        level_ref = ref_pyramid[level]
        level_mov = mov_pyramid[level]

        if level == len(ref_pyramid) - 1:
            range_x = (-max_shift, max_shift + 1)
            range_y = (-max_shift, max_shift + 1)
        else:
            current_shift_x = int(x_shift * 2)
            current_shift_y = int(y_shift * 2)
            current_max_shift = max(int(max_shift / (2 ** level)), 1)
            range_x = (current_shift_x - current_max_shift, current_shift_x + current_max_shift + 1)
            range_y = (current_shift_y - current_max_shift, current_shift_y + current_max_shift + 1)

        x_shift, y_shift = get_best_shifts(level_ref, level_mov, range_x, range_y, metric)

    a_to_b = np.array([-y_shift, -x_shift])
    return a_to_b


def find_absolute_shifts(
    crops,
    crop_coords,
    find_relative_shift_fn,
):
    red_crop, green_crop, blue_crop = crops
    red_crop_coords, green_crop_coords, blue_crop_coords = crop_coords
    relative_shift_r = find_relative_shift_fn(green_crop, red_crop)
    relative_shift_b = find_relative_shift_fn(green_crop, blue_crop)
    r_to_g = red_crop_coords + relative_shift_r - green_crop_coords
    b_to_g = blue_crop_coords + relative_shift_b - green_crop_coords
    return -r_to_g, -b_to_g


def get_bounds(y, x, channel, shift_y, shift_x):
    H, W = channel.shape
    return (
        y + shift_y, y + shift_y + H, 
        x + shift_x, x + shift_x + W
    )


def create_aligned_image(
    channels,
    channel_coords,
    r_to_g,
    b_to_g,
):
    red_channel, green_channel, blue_channel = channels
    red_coords, green_coords, blue_coords = channel_coords
    red_y, red_x = red_coords[:2]
    green_y, green_x = green_coords[:2]
    blue_y, blue_x = blue_coords[:2]
    red_shift_y, red_shift_x = r_to_g
    blue_shift_y, blue_shift_x = b_to_g
    
    red_bounds = get_bounds(red_y, red_x, red_channel, red_shift_y, red_shift_x)
    green_bounds = get_bounds(green_y, green_x, green_channel, 0, 0)
    blue_bounds = get_bounds(blue_y, blue_x, blue_channel, blue_shift_y, blue_shift_x)
    
    y_start = max(red_bounds[0], green_bounds[0], blue_bounds[0])
    y_end = min(red_bounds[1], green_bounds[1], blue_bounds[1])
    x_start = max(red_bounds[2], green_bounds[2], blue_bounds[2])
    x_end = min(red_bounds[3], green_bounds[3], blue_bounds[3])
    
    red_crop = red_channel[
        int(y_start - red_bounds[0]): int(y_end - red_bounds[0]),
        int(x_start - red_bounds[2]): int(x_end - red_bounds[2])
    ]
    
    green_crop = green_channel[
        int(y_start - green_bounds[0]): int(y_end - green_bounds[0]),
        int(x_start - green_bounds[2]): int(x_end - green_bounds[2])
    ]
    
    blue_crop = blue_channel[
        int(y_start - blue_bounds[0]): int(y_end - blue_bounds[0]),
        int(x_start - blue_bounds[2]): int(x_end - blue_bounds[2])
    ]

    return np.dstack([red_crop, green_crop, blue_crop])


def find_relative_shift_fourier(img_a, img_b):
    C_uv = np.fft.ifft2(np.conj(np.fft.fft2(img_a)) * np.fft.fft2(img_b))
    optimal_shifts = np.array(np.unravel_index(np.argmax(np.abs((C_uv))), img_a.shape))
    if optimal_shifts[0] > img_a.shape[0] // 2:
        optimal_shifts[0] -= img_a.shape[0]
    if optimal_shifts[1] > img_a.shape[1] // 2:
        optimal_shifts[1] -= img_a.shape[1]
    return optimal_shifts


if __name__ == "__main__":

    import common
    import pipeline

    # Read the source image and the corresponding ground truth information
    test_path = "tests/05_unittest_align_image_pyramid_img_small_input/00"
    raw_img, (r_point, g_point, b_point) = common.read_test_data(test_path)

    # Draw the same point on each channel in the original
    # raw image using the ground truth coordinates
    visualized_img = pipeline.visualize_point(raw_img, r_point, g_point, b_point)
    common.save_image(f"gt_visualized.png", visualized_img)

    for method in ["pyramid", "fourier"]:
        # Run the whole alignment pipeline
        r_to_g, b_to_g, aligned_img = pipeline.align_image(raw_img, method)
        common.save_image(f"{method}_aligned.png", aligned_img)

        # Draw the same point on each channel in the original
        # raw image using the predicted r->g and b->g shifts
        # (Compare with gt_visualized for debugging purposes)
        r_pred = g_point - r_to_g
        b_pred = g_point - b_to_g
        visualized_img = pipeline.visualize_point(raw_img, r_pred, g_point, b_pred)

        r_error = abs(r_pred - r_point)
        b_error = abs(b_pred - b_point)
        print(f"{method}: {r_error = }, {b_error = }")

        common.save_image(f"{method}_visualized.png", visualized_img)