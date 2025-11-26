import io
import pickle
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio

# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """

    row_means = np.mean(matrix, axis=1, keepdims=True)
    X_meaned = matrix - row_means
    cov_matrix = np.dot(X_meaned, X_meaned.T) / (X_meaned.shape[1] - 1)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
    top_indices = np.argsort(eigen_values)[-p:][::-1]
    eigen_vectors = eigen_vectors[:, top_indices]
    projection = np.dot(eigen_vectors.T, X_meaned)
    return eigen_vectors, projection, row_means.ravel()


def pca_decompression(compressed):
    """Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """

    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!
        eigen_vectors, projection, mean = comp
        decomp_channel = eigen_vectors @ projection + mean[:, None]
        result_img.append(np.clip(decomp_channel, 0, 255).astype(np.uint8))

    return np.stack(result_img, axis=-1)


def pca_visualize():
    plt.clf()
    img = imread("cat.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            # Your code here
            compressed.append(pca_compression(img[:, :, j], p))

        axes[i // 3, i % 3].imshow(pca_decompression(compressed))
        axes[i // 3, i % 3].set_title("Компонент: {}".format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """

    first_channel = img[:, :, 0] * 0.299 + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    second_channel = 128 + img[:, :, 0] * (-0.1687) - 0.3313 * img[:, :, 1] + 0.5 * img[:, :, 2]
    third_channel = 128 + img[:, :, 0] * 0.5 - 0.4187 * img[:, :, 1] -0.0813 * img[:, :, 2]
    return np.dstack((first_channel, second_channel, third_channel)).astype(np.uint8)


def ycbcr2rgb(img):
    """Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """
    first_channel = img[:, :, 0] + 1.402 * (img[:, :, 2] - 128.)
    second_channel = img[:, :, 0] - 0.34414 * (img[:, :, 1] - 128) - 0.71414 * (img[:, :, 2] - 128.)
    third_channel = img[:, :, 0] + 1.77 * (img[:, :, 1] - 128)
    return np.clip(np.dstack((first_channel, second_channel, third_channel)), 0, 255).astype(np.uint8)


def get_gauss_1():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here
    YCbCR = rgb2ycbcr(rgb_img)
    YCbCR[:, :, 1] = gaussian_filter(YCbCR[:, :, 1], 10)
    YCbCR[:, :, 2] = gaussian_filter(YCbCR[:, :, 2], 10)
    rgb_img = ycbcr2rgb(YCbCR)
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here
    YCbCR = rgb2ycbcr(rgb_img)
    YCbCR[:, :, 0] = gaussian_filter(YCbCR[:, :, 0], 10)
    rgb_img = ycbcr2rgb(YCbCR)
    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B]
    Выход: цветовая компонента размера [A // 2, B // 2]
    """
    return gaussian_filter(component, 10)[::2, ::2]


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """
    U, V = block.shape # U = V = 8
    result = np.zeros((U, V))
    is_edge = lambda x: 1 / np.sqrt(2) if x == 0 else 1
    for u in range(U):
        for v in range(V):
          cos = 0
          for x in range(U):
            for y in range(V):
              cos += block[x, y] * np.cos((2 * x + 1) * u * np.pi / (2 * U)) * np.cos((2 * y + 1) * v * np.pi / (2 * V))
          result[u, v] = 1 / 4 * is_edge(u) * is_edge(v) * cos
    return result


# Матрица квантования яркости
y_quantization_matrix = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

# Матрица квантования цвета
color_quantization_matrix = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]
)


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """

    return np.round(block / quantization_matrix)


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100

    if 1 <= q <= 50:
      S = 5000 / q
    elif 50 <= q <= 99:
      S = 200 - 2 * q
    else:
      S = 1

    own_quant_matr = ((50 + S * default_quantization_matrix) / 100).astype(int)
    own_quant_matr[own_quant_matr == 0] = 1
    return own_quant_matr


def zigzag(block):
    "Сначала была имплементирована на чистом python"
    "Но потом в процессе дебага поменял на https://stackoverflow.com/questions/39440633/matrix-to-vector-with-python-numpy"
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """
    return np.concatenate(
        [np.diagonal(block[::-1, :], k)[::(2 * (k % 2) - 1)] for k in range(1 - block.shape[0], block.shape[0])]
    )


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """
    compressed = []
    zero_count = 0  

    for value in zigzag_list:
        if value == 0:
            zero_count += 1  
        else:
            if zero_count > 0:
                compressed.append(0)  
                compressed.append(zero_count)  
                zero_count = 0  
            compressed.append(value)  

    if zero_count > 0:
        compressed.append(0)
        compressed.append(zero_count)

    return compressed


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # Your code here

    # Переходим из RGB в YCbCr
    YCbCr = rgb2ycbcr(img)
    # Уменьшаем цветовые компоненты
    y, Cb, Cr = np.transpose(YCbCr, (2, 0, 1))
    Cb = downsampling(Cb)
    Cr = downsampling(Cr)
    reduced_channels = [y, Cb, Cr]
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    block_size = 8
    quant_matrices = [quantization_matrixes[0], quantization_matrixes[1], quantization_matrixes[1]]
    result = []
    
    for channel in range(len(reduced_channels)):
        result.append([])
        for h in range(0, reduced_channels[channel].shape[0] - (block_size - 1), block_size):
            for w in range(0, reduced_channels[channel].shape[1] - (block_size - 1), 8):
                block = reduced_channels[channel][h: h + 8, w: w + 8] - 128
    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
                quantized = quantization(dct(block), quant_matrices[channel])
                result[-1].append(compression(zigzag(quantized)))
    return result


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """

    decompressed = []
    i = 0

    while i < len(compressed_list):
        if compressed_list[i] == 0:
            zero_count = compressed_list[i + 1]
            decompressed.extend([0] * zero_count)  
            i += 2  
        else:
            decompressed.append(compressed_list[i])  
            i += 1  

    return decompressed


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """

    indexes = [
        0,  1,  5,  6, 14, 15, 27, 28, 
        2,  4,  7, 13, 16, 26, 29, 42, 
        3,  8, 12, 17, 25, 30, 41, 43, 
        9, 11, 18, 24, 31, 40, 44, 53, 
        10, 19, 23, 32, 39, 45, 52, 54, 
        20, 22, 33, 38, 46, 51, 55, 60, 
        21, 34, 37, 47, 50, 56, 59, 61, 
        35, 36, 48, 49, 57, 58, 62, 63, 
    ]

    return np.array(input)[indexes].reshape(8, 8)


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """

    return block * quantization_matrix


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """

    U, V = block.shape 
    res = np.zeros((U, V))
    is_edge = lambda x: 1 / np.sqrt(2) if x == 0 else 1
    for x in range(U):
        for y in range(V):
            cos = 0
            for u in range(U):
                for v in range(V):
                    alpha_u, alpha_v = is_edge(u), is_edge(v)
                    cos += alpha_u * alpha_v * block[u, v] * np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
            res[x, y] = 1 / 4 * cos
    res = np.round(res)
    return res


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """

    A, B = component.shape
    upsampled = np.repeat(np.repeat(component, 2, axis=0), 2, axis=1)
    return upsampled.reshape(2 * A, 2 * B)


def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """

    H, W, _ = result_shape
    block_size = 8
    quant_matrices = [quantization_matrixes[0], quantization_matrixes[1], quantization_matrixes[1]]
    sizes = ((H, W), (H // 2, W // 2), (H // 2, W // 2))
    up_channels = []

    for channel_idx in range(len(quant_matrices)):
        channel = np.zeros(sizes[channel_idx])
        for j in range(len(result[channel_idx])):
            result[channel_idx][j] = inverse_dct(inverse_quantization(inverse_zigzag(
                inverse_compression(result[channel_idx][j])), quant_matrices[channel_idx])
            )

        k = 0
        for i in range(0, sizes[channel_idx][0] - (block_size - 1), block_size):
            for j in range(0, sizes[channel_idx][1] - (block_size - 1), block_size):
                channel[i: i + block_size, j: j + block_size] = result[channel_idx][k] + 128
                k += 1

        up_channels.append(channel)

    up_channels[1] = upsampling(up_channels[1])
    up_channels[2] = upsampling(up_channels[2])
    return np.clip(ycbcr2rgb(np.dstack(up_channels)), 0, 255).astype(np.uint8)


def jpeg_visualize():
    plt.clf()
    img = imread("Lenna.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        # Your code here
        
        compressed = jpeg_compression(
            img, 
            quantization_matrixes=(
                own_quantization_matrix(y_quantization_matrix, p), 
                (own_quantization_matrix(color_quantization_matrix, p))
            )
        )
        decompressed = jpeg_decompression(
            compressed, 
            img.shape, 
            quantization_matrixes=(
                own_quantization_matrix(y_quantization_matrix, p), 
                (own_quantization_matrix(color_quantization_matrix, p)))
            )

        axes[i // 3, i % 3].imshow(decompressed)
        axes[i // 3, i % 3].set_title("Quality Factor: {}".format(p))

    fig.savefig("jpeg_visualization.png")


def get_deflated_bytesize(data):
    raw_data = pickle.dumps(data)
    with io.BytesIO() as buf:
        with (
            zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf,
            zipf.open("data", mode="w") as handle,
        ):
            handle.write(raw_data)
            handle.flush()
            handle.close()
            zipf.close()
        buf.flush()
        return buf.getbuffer().nbytes


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg';
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    if c_type.lower() == "jpeg":
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
        compressed_size = get_deflated_bytesize(compressed)

    elif c_type.lower() == "pca":
        compressed = [
            pca_compression(c.copy(), param)
            for c in img.transpose(2, 0, 1).astype(np.float64)
        ]

        img = pca_decompression(compressed)
        compressed_size = sum(d.nbytes for c in compressed for d in c)

    raw_size = img.nbytes
    return img, compressed_size / raw_size


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Compression Ratio для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    ratio = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title("Quality Factor vs PSNR for {}".format(c_type.upper()))
    ax1.plot(param_list, psnr, "tab:orange")
    ax1.set_ylim(13, 64)
    ax1.set_xlabel("Quality Factor")
    ax1.set_ylabel("PSNR")

    ax2.set_title("PSNR vs Compression Ratio for {}".format(c_type.upper()))
    ax2.plot(psnr, ratio, "tab:red")
    ax2.set_xlim(13, 30)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("PSNR")
    ax2.set_ylabel("Compression Ratio")
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "pca", [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "jpeg", [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")


if __name__ == "__main__":
    pca_visualize()
    get_gauss_1()
    get_gauss_2()
    jpeg_visualize()
    get_pca_metrics_graph()
    get_jpeg_metrics_graph()
