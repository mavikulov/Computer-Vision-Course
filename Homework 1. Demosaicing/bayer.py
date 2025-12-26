import numpy as np


def get_bayer_masks(n_rows, n_cols):
    """
    :param n_rows: `int`, number of rows
    :param n_cols: `int`, number of columns

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.bool_`
        containing red, green and blue Bayer masks
    """
    red_tile = np.array([
        [0, 1],
        [0, 0]
    ], dtype=np.bool)

    green_tile = np.array([
        [1, 0],
        [0, 1]
    ], dtype=np.bool)

    blue_tile = np.array([
        [0, 0],
        [1, 0]
    ], dtype=np.bool)

    red_tiles = np.tile(red_tile, (n_rows // 2 + 1, n_cols // 2 + 1))[:n_rows, :n_cols]
    green_tiles = np.tile(green_tile, (n_rows // 2 + 1, n_cols // 2 + 1))[:n_rows, :n_cols]
    blue_tiles = np.tile(blue_tile, (n_rows // 2 + 1, n_cols // 2 + 1))[:n_rows, :n_cols]
    return np.dstack([red_tiles, green_tiles, blue_tiles])


def get_colored_img(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        each channel contains known color values or zeros
        depending on Bayer masks
    """
    bayer_masks = get_bayer_masks(*raw_img.shape)
    red_channel = raw_img * bayer_masks[:, :, 0]
    green_channel = raw_img * bayer_masks[:, :, 1]
    blue_channel = raw_img * bayer_masks[:, :, 2]
    return np.dstack([red_channel, green_channel, blue_channel])


def get_raw_img(colored_img):
    """
    :param colored_img:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        colored image

    :return:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image as captured by camera
    """
    n_rows, n_cols, _ = colored_img.shape
    bayer_masks = get_bayer_masks(n_rows, n_cols)
    raw_img = (
          colored_img[:, :, 0] * bayer_masks[:, :, 0] + 
          colored_img[:, :, 1] * bayer_masks[:, :, 1] + 
          colored_img[:, :, 2] * bayer_masks[:, :, 2]
    )
    return raw_img


def convolve2d_fast_numpy(image, kernel):
    """
    Implement fast version of convolve with "stride tricks" instead of using scipy.ndimage.convolve
    """
    kH, kW = kernel.shape
    H, W = image.shape
    pH, pW = kH // 2, kW // 2
    image_padded = np.pad(image, ((pH, pH), (pW, pW)), mode='reflect')
    shape = (H, W, kH, kW)
    strides = (
        image_padded.strides[0], image_padded.strides[1], 
        image_padded.strides[0], image_padded.strides[1]
    )
    
    windows = np.lib.stride_tricks.as_strided(image_padded, shape=shape, strides=strides)
    return np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))


def bilinear_interpolation(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)`, and dtype `np.uint8`,
        result of bilinear interpolation
    """
    H, W = raw_img.shape
    bayer_masks = get_bayer_masks(H, W)
    result = get_colored_img(raw_img).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.float32)

    for channel in range(3):
        known_values_mask = bayer_masks[:, :, channel]
        neighbor_count = convolve2d_fast_numpy(known_values_mask.astype(np.float32), kernel)
        neighbor_sum = convolve2d_fast_numpy(result[:, :, channel], kernel)

        with np.errstate(divide='ignore', invalid='ignore'):
            interpolated_values = neighbor_sum / neighbor_count
            interpolated_values[np.isnan(interpolated_values)] = 0
        result[:, :, channel] += (1 - known_values_mask) * interpolated_values
        
    return np.clip(result, 0, 255).astype(np.uint8)


def improved_interpolation(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`, raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        result of improved interpolation
    """
    image_data = raw_img.astype('float64')
    H, W = image_data.shape

    red_blue_filter = np.array(
        [
            [0.0,  0.0,  0.0625, 0.0,  0.0],
            [0.0, -0.125, 0.0,   -0.125, 0.0],
            [-0.125, 0.5,  0.625, 0.5,  -0.125],
            [0.0, -0.125, 0.0,   -0.125, 0.0],
            [0.0,  0.0,  0.0625, 0.0,  0.0]
        ]
    )

    cross_color_filter = np.array(
        [
            [0.0,   0.0,   -0.1875, 0.0,   0.0],
            [0.0,   0.25,   0.0,    0.25,  0.0],
            [-0.1875, 0.0,   0.75,   0.0,  -0.1875],
            [0.0,   0.25,   0.0,    0.25,  0.0],
            [0.0,   0.0,   -0.1875, 0.0,   0.0]
        ]
    )

    green_reconstruction = np.array(
        [
            [0.0,  0.0,  -0.125, 0.0,  0.0],
            [0.0,  0.0,   0.25,  0.0,  0.0],
            [-0.125, 0.25, 0.5,   0.25, -0.125],
            [0.0,  0.0,   0.25,  0.0,  0.0],
            [0.0,  0.0,  -0.125, 0.0,  0.0]
        ]
    )

    blue_red_filter = red_blue_filter.T

    color_masks = get_bayer_masks(H, W)
    red_mask, green_mask, blue_mask = (
        color_masks[:, :, 0],
        color_masks[:, :, 1],
        color_masks[:, :, 2]
    )

    red_channel = image_data * red_mask
    green_channel = image_data * green_mask
    blue_channel = image_data * blue_mask
    
    green_interpolated = convolve2d_fast_numpy(image_data, green_reconstruction)
    green_channel = np.where(red_mask | blue_mask, green_interpolated, green_channel)
    
    rb_interp_1 = convolve2d_fast_numpy(image_data, red_blue_filter)
    rb_interp_2 = convolve2d_fast_numpy(image_data, blue_red_filter) 
    cross_interp = convolve2d_fast_numpy(image_data, cross_color_filter)
    
    red_rows = (np.sum(red_mask, axis=1) > 0)[:, np.newaxis]
    red_cols = (np.sum(red_mask, axis=0) > 0)[np.newaxis, :]
    blue_rows = (np.sum(blue_mask, axis=1) > 0)[:, np.newaxis] 
    blue_cols = (np.sum(blue_mask, axis=0) > 0)[np.newaxis, :]
    
    red_row_mask = np.tile(red_rows, (1, W))
    red_col_mask = np.tile(red_cols, (H, 1))
    blue_row_mask = np.tile(blue_rows, (1, W))
    blue_col_mask = np.tile(blue_cols, (H, 1))
    
    red_condition_1 = red_row_mask & blue_col_mask 
    red_condition_2 = blue_row_mask & red_col_mask   
    red_condition_3 = blue_row_mask & blue_col_mask 
    
    red_channel = np.where(red_condition_1, rb_interp_1, red_channel)
    red_channel = np.where(red_condition_2, rb_interp_2, red_channel)
    red_channel = np.where(red_condition_3, cross_interp, red_channel)
     
    blue_condition_1 = blue_row_mask & red_col_mask 
    blue_condition_2 = red_row_mask & blue_col_mask  
    blue_condition_3 = red_row_mask & red_col_mask 
    
    blue_channel = np.where(blue_condition_1, rb_interp_1, blue_channel)
    blue_channel = np.where(blue_condition_2, rb_interp_2, blue_channel) 
    blue_channel = np.where(blue_condition_3, cross_interp, blue_channel)
    
    result_image = np.dstack([red_channel, green_channel, blue_channel])
    return np.clip(np.round(result_image), 0, 255).astype('uint8')


def MSE(image_1, image_2):
    """
    :param image_1: 
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        predicted image
    :param image_2: 
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        ground truth image
        
    :return:
        `float`, MSE metric
    """
    metric_value = np.mean((image_1 - image_2)**2)
    if metric_value < 1e-9:
        raise ValueError("It might cause overflow encountering!")
    return metric_value


def compute_psnr(img_pred, img_gt):
    """
    :param img_pred:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        predicted image
    :param img_gt:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        ground truth image

    :return:
        `float`, PSNR metric
    """
    img_pred = img_pred.astype(np.float64)
    img_gt = img_gt.astype(np.float64)
    mse_score = MSE(img_pred, img_gt)
    return 10 * np.log10(np.max(img_gt**2) / mse_score)


if __name__ == "__main__":
    from PIL import Image

    raw_img_path = "tests/04_unittest_bilinear_img_input/02.png"
    raw_img = np.array(Image.open(raw_img_path))

    img_bilinear = bilinear_interpolation(raw_img)
    Image.fromarray(img_bilinear).save("bilinear.png")

    img_improved = improved_interpolation(raw_img)
    Image.fromarray(img_improved).save("improved.png")
