from interface import *

from numpy.lib.stride_tricks import sliding_window_view


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            return parameter - self.lr * parameter_grad

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            updater.inertia = self.momentum * updater.inertia + self.lr * parameter_grad
            return parameter - updater.inertia

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        return np.maximum(inputs, 0)

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        return grad_outputs * (self.forward_inputs >= 0)


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, d)), output values

            n - batch size
            d - number of units
        """
        # I use LogSumExp trick from wikipedia https://en.wikipedia.org/wiki/LogSumExp
        c = np.max(inputs, axis=1, keepdims=True)
        lse_inputs = c + np.log(np.sum(np.exp(inputs - c), axis=1, keepdims=True))
        return np.exp(inputs - lse_inputs)

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of units
        """
        c = np.sum(self.forward_outputs * grad_outputs, axis=1, keepdims=True)
        return self.forward_outputs * (grad_outputs - c)


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        (input_units,) = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name="weights",
            shape=(output_units, input_units),
            initializer=he_initializer(input_units),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_units,),
            initializer=np.zeros,
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, c)), output values

            n - batch size
            d - number of input units
            c - number of output units
        """
        return inputs @ self.weights.T + self.biases

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of input units
            c - number of output units
        """
        self.biases_grad[...] = np.sum(grad_outputs, axis=0)
        self.weights_grad[...] = grad_outputs.T @ self.forward_inputs
        return grad_outputs @ self.weights


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((1,)), mean Loss scalar for batch

            n - batch size
            d - number of units
        """
        y_pred_clipped = y_pred.clip(eps, 1. - eps)
        loss_per_sample = -np.sum(y_gt * np.log(y_pred_clipped), axis=1)
        return np.mean(loss_per_sample, keepdims=True)

    def gradient_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((n, d)), dLoss/dY_pred

            n - batch size
            d - number of units
        """
        y_pred_clipped = y_pred.clip(eps, 1. - eps)
        return -(y_gt / y_pred_clipped) / y_gt.shape[0]


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # 1) Create a Model
    loss = CategoricalCrossentropy()
    optimizer = SGD(lr=0.001)
    model = Model(loss, optimizer)

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(units=512, input_shape=(784, )))
    model.add(ReLU())
    model.add(Dense(256))
    model.add(ReLU())
    model.add(Dense(128))
    model.add(ReLU())
    model.add(Dense(64))
    model.add(ReLU())
    model.add(Dense(10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(
        x_train=x_train,
        y_train=y_train,
        batch_size=64,
        epochs=10,
        shuffle=True,
        verbose=True,
        x_valid=x_valid,
        y_valid=y_valid
    )

    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get("USE_FAST_CONVOLVE", False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # It's easier for me use these names of dimensions
    N, C_in, H_in, W_in = inputs.shape
    C_out, C_in, kH, kW = kernels.shape

    if padding > 0:
        padded_inputs = np.pad(
            inputs,
            pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant'
        )
    else:
        padded_inputs = inputs

    patches = sliding_window_view(
        padded_inputs, 
        window_shape=(kH, kW),
        axis=(2, 3)
    )

    kernels_flipped = np.flip(kernels, axis=(2, 3))
    patches = patches.transpose(0, 2, 3, 1, 4, 5) 
    N, H_out, W_out, C_in, kH, kW = patches.shape
    elements_in_window = C_in * kH * kW

    patches_flat = patches.reshape(N, H_out, W_out, elements_in_window)
    kernels_flat = kernels_flipped.reshape(C_out, elements_in_window)      
    out = np.matmul(patches_flat, kernels_flat.T)
    return out.transpose(0, 3, 1, 2)


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name="kernels",
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_channels,),
            initializer=np.zeros,
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, c, h, w)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        convolve_result = convolve(
            inputs=inputs,
            kernels=self.kernels,
            padding=(self.kernel_size - 1) // 2
        )
        return convolve_result + self.biases[None, ..., None, None]

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        kernels_flipped = np.flip(self.kernels, axis=(2, 3))
        inputs_flipped = np.flip(self.forward_inputs, axis=(2, 3))
        padding = (self.kernel_size - 1) // 2

        self.kernels_grad = convolve(
            inputs_flipped.transpose((1, 0, 2, 3)),
            grad_outputs.transpose(1, 0, 2, 3),
            padding
        ).transpose(1, 0, 2, 3)

        self.biases_grad = np.sum(grad_outputs, axis=(0, 2, 3))
        return convolve(grad_outputs, kernels_flipped.transpose(1, 0, 2, 3), padding)


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode="max", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {"avg", "max"}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, ih, iw)), input values

            :return: np.array((n, d, oh, ow)), output values

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        N, C, H_in, W_in = inputs.shape
        kH = kW = self.pool_size
        H_out, W_out = H_in // kH, W_in // kW

        windows = inputs.reshape(N, C, H_out, kH, W_out, kW).transpose(0, 1, 2, 4, 3, 5)

        if self.pool_mode == "avg":
            return windows.mean(axis=(4, 5))
        else:
            windows = windows.reshape(N, C, H_out, W_out, kH * kW)
            idx = windows.argmax(axis=-1)            
            self.forward_idxs = idx.astype(np.int64)
            return np.take_along_axis(windows, idx[..., None], axis=-1)[..., 0]

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

        :return: np.array((n, d, ih, iw)), dLoss/dInputs

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        N, C, H_in, W_in = grad_outputs.shape[0], *self.input_shape
        kH = kW = self.pool_size
        H_out, W_out = H_in // kH, W_in // kW

        if self.pool_mode == "avg":
            grad_windows = np.tile(grad_outputs[..., None, None] / (kH * kW), (1, 1, 1, 1, kH, kW))
        else:
            flattened_grad = np.zeros((N, C, H_out, W_out, kH * kW))
            np.put_along_axis(flattened_grad, self.forward_idxs[..., None], grad_outputs[..., None], axis=-1)
            grad_windows = flattened_grad.reshape(N, C, H_out, W_out, kH, kW)

        return grad_windows.transpose(0, 1, 2, 4, 3, 5).reshape(N, C, H_in, W_in)


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name="beta",
            shape=(input_channels,),
            initializer=np.zeros,
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name="gamma",
            shape=(input_channels,),
            initializer=np.ones,
        )

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, d, h, w)), output values

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        if self.is_training:
            mean = np.mean(inputs, axis=(0, 2, 3), keepdims=True)
            var = np.var(inputs, axis=(0, 2, 3), keepdims=True)
            X_hat = (inputs - mean) / np.sqrt(var + eps)

            self.running_mean = (
                self.momentum * self.running_mean + (1.0 - self.momentum) * np.ravel(mean)
            )

            self.running_var = (
                self.momentum * self.running_var + (1.0 - self.momentum) * np.ravel(var)
            )

            self.forward_centered_inputs = inputs - mean 
            self.forward_inverse_std = 1.0 / np.sqrt(var + eps)
            self.forward_normalized_inputs = X_hat
        else:
            mean = (inputs - self.running_mean[None, :, None, None])
            var = np.sqrt(self.running_var[None, :, None, None] + eps)
            X_hat = mean / var

        return self.gamma[None, :, None, None] * X_hat + self.beta[None, :, None, None]

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # inspired by the next article 
        # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        N, C, H, W = grad_outputs.shape
        elements_per_channel = N * H * W
        X_hat = self.forward_normalized_inputs   
        inverse_std = self.forward_inverse_std

        self.beta_grad = np.sum(grad_outputs, axis=(0, 2, 3))
        self.gamma_grad = np.sum(grad_outputs * X_hat, axis=(0, 2, 3))

        dX_hat = grad_outputs * self.gamma[None, :, None, None]
        sum_dX_hat = np.sum(dX_hat, axis=(0, 2, 3), keepdims=True)                
        sum_dxhat_xhat = np.sum(dX_hat * X_hat, axis=(0, 2, 3), keepdims=True)   

        return (1.0 / (elements_per_channel)) * inverse_std * (
            (elements_per_channel * dX_hat) - sum_dX_hat - (X_hat * sum_dxhat_xhat)
        )


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (int(np.prod(self.input_shape)),)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, (d * h * w))), output values

            n - batch size
            d - number of input channels
            (h, w) - image shape
        """
        return inputs.reshape(-1, self.output_shape[0])

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of units
            (h, w) - input image shape
        """
        return grad_outputs.reshape(-1, *self.input_shape)


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        if self.is_training:
            mask = (np.random.uniform(size=inputs.shape) > self.p).astype(inputs.dtype)
            self.forward_mask = mask
            return inputs * mask
        else:
            return (1 - self.p) * inputs 

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        if self.is_training:
            return grad_outputs * self.forward_mask
        else:
            return (1 - self.p) * grad_outputs


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    loss = CategoricalCrossentropy()
    optimizer = SGDMomentum(lr=0.05, momentum=0.6)
    model = Model(loss, optimizer)

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Conv2D(16, input_shape=(3, 32, 32)))
    model.add(BatchNorm())
    model.add(ReLU())       
    model.add(Pooling2D(pool_mode="max"))  

    model.add(Conv2D(32))
    model.add(BatchNorm())  
    model.add(ReLU())       
    model.add(Pooling2D(pool_mode="max"))

    model.add(Conv2D(64))
    model.add(BatchNorm())
    model.add(ReLU())       
    model.add(Pooling2D())  
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(ReLU())       
    model.add(Dropout(0.2))  

    model.add(Dense(128))
    model.add(ReLU())

    model.add(Dense(10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        batch_size=128,
        epochs=10
    )

    # your code here /\
    return model


# ============================================================================
