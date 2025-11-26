import numpy as np
import scipy.fft


def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    padding = [
        ((shape[0] - h.shape[0] + 1) // 2, (shape[0] - h.shape[0]) // 2),
        ((shape[1] - h.shape[1] + 1) // 2, (shape[1] - h.shape[1]) // 2)
    ]
    padded_h = np.pad(h, padding)
    return scipy.fft.fft2(scipy.fft.ifftshift(padded_h))


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    H_inv = np.zeros(H.shape, dtype=complex)
    nonzero_mask = np.abs(H) > threshold
    H_inv[nonzero_mask] = 1 / H[nonzero_mask]
    return H_inv


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    H_inv = inverse_kernel(fourier_transform(h, blurred_img.shape), threshold=threshold)
    G = fourier_transform(blurred_img, blurred_img.shape)
    return scipy.fft.fftshift(np.absolute(scipy.fft.ifft2(G * H_inv)))


def wiener_filtering(blurred_img, h, K=0.00009):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    H = fourier_transform(h, blurred_img.shape)
    G = fourier_transform(blurred_img, blurred_img.shape)
    F_hat = np.conj(H) * G / (np.conj(H) * H + K)
    return scipy.fft.fftshift(np.absolute(scipy.fft.ifft2(F_hat)))


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    max_pixel_value = 255
    rmse = np.sqrt(np.mean((img1 - img2) ** 2))
    return 20 * np.log10(max_pixel_value / rmse)
