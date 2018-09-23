# -*- coding: utf-8 -*-

import os

import numpy as np

from keras.layers import Input, Dense, Activation, Flatten, Add, Lambda, Concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.engine.network import Network, Layer
from keras.initializers import TruncatedNormal
import keras.backend as K


def set_trainable(model, prefix_list, trainable=False):
    for prefix in prefix_list:
        for layer in model.layers:
            if layer.name.startswith(prefix):
                layer.trainable = trainable
    return model

def mapping_function_Unet(input_shape, base_name, num_res_blocks):
    initializer = TruncatedNormal(mean=0, stddev=0.2, seed=42)
    x = in_x = Input(shape=input_shape)

    # size→size//2→size//4→size//8
    x = Conv2D(32, kernel_size=7, strides=1, padding="same", kernel_initializer=initializer,
               use_bias=False,
               name=base_name + "_conv1")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn1")(x)
    conv1 = LeakyReLU(0.2)(x)
    x = Conv2D(64, kernel_size=3, strides=2, padding="same", kernel_initializer=initializer,
               use_bias=False,
               name=base_name + "_conv2")(conv1)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn2")(x)
    conv2 = LeakyReLU(0.2)(x)
    x = Conv2D(128, kernel_size=3, strides=2, padding="same", kernel_initializer=initializer,
               use_bias=False,
               name=base_name + "_conv3")(conv2)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn3")(x)
    conv3 = LeakyReLU(0.2)(x)
    x = conv3

    for i in range(num_res_blocks):
        x = residual_block(x, base_name=base_name, block_num=i, initializer=initializer)

    x = Concatenate(axis=3)([x, conv3])

    # size//8→size//4→size//2→size
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer,
                        name=base_name + "_deconv2")(x)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn6")(x)
    x = Activation("relu")(x)

    x = Concatenate(axis=3)([x, conv2])

    x = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer,
                        name=base_name + "_deconv3")(x)


    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn7")(x)
    x = Activation("relu")(x)

    x = Concatenate(axis=3)([x, conv1])

    out = Conv2DTranspose(3, kernel_size=7, strides=1, padding='same', activation="tanh",
                          kernel_initializer=initializer, name=base_name + "_out")(x)
    network = Network(in_x, out, name=base_name)
    return network

def mapping_function(input_shape, base_name, num_res_blocks):
    initializer = TruncatedNormal(mean=0, stddev=0.2, seed=42)
    x = in_x = Input(shape=input_shape)

    # size→size//2→size//4→size//8
    x = Conv2D(32, kernel_size=7, strides=1, padding="same", kernel_initializer=initializer,
               use_bias=False,
               name=base_name + "_conv1")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn1")(x)
    x = Activation("relu")(x)
    x = Conv2D(64, kernel_size=3, strides=2, padding="same", kernel_initializer=initializer,
               use_bias=False,
               name=base_name + "_conv2")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn2")(x)
    x = Activation("relu")(x)
    x = Conv2D(128, kernel_size=3, strides=2, padding="same", kernel_initializer=initializer,
               use_bias=False,
               name=base_name + "_conv3")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn3")(x)
    x = Activation("relu")(x)

    for i in range(num_res_blocks):
        x = residual_block(x, base_name=base_name, block_num=i, initializer=initializer)

    # size//8→size//4→size//2→size
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer,
                        name=base_name + "_deconv2")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn6")(x)
    x = Activation("relu")(x)
    x = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer,
                        name=base_name + "_deconv3")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn7")(x)
    x = Activation("relu")(x)
    out = Conv2DTranspose(3, kernel_size=7, strides=1, padding='same', activation="tanh",
                          kernel_initializer=initializer, name=base_name + "_out")(x)
    network = Network(in_x, out, name=base_name)
    return network


def discriminator(input_shape, base_name, num_res_blocks=0,is_wgangp=False, use_res=False):
    initializer_d = TruncatedNormal(mean=0, stddev=0.1, seed=42)

    D = in_D = Input(shape=input_shape)
    D = Conv2D(64, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv1")(D)
    D = LeakyReLU(0.2)(D)
    D = Conv2D(128, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv2")(D)
    D = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn1")(D)
    D = LeakyReLU(0.2)(D)

    D = Conv2D(256, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv3")(D)
    D = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn2")(D)
    D = LeakyReLU(0.2)(D)

    if use_res:
        for i in range(5):
            D = residual_block(D, base_name=base_name, block_num=i,
                               initializer=initializer_d, num_channels=256, is_wgangp=is_wgangp)

    D = Conv2D(512, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv4")(D)
    D = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn3")(D)
    D = LeakyReLU(0.2)(D)

    if not is_wgangp:
        D = Flatten()(D)
        D = Dense(units=128, name=base_name + "_dense1")(D)
        D = LeakyReLU(0.2)(D)
        out = Dense(units=1, activation="sigmoid", name=base_name + "_out")(D)
    else:
        #D = GlobalAveragePooling2D()(D)
        D = Flatten()(D)
        D = Dense(units=128, name=base_name + "_dense1")(D)
        D = LeakyReLU(0.2)(D)
        out = Dense(units=1, activation=None, name=base_name + "_out")(D)
    network = Network(in_D, out, name=base_name)

    return network


def residual_block(x, base_name, block_num, initializer, num_channels=128,is_wgangp=False):
    y = Conv2D(num_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=initializer, use_bias=False,
               name=base_name + "_resblock" + str(block_num) + "_conv1")(x)
    if not is_wgangp:
        y = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_resblock" + str(block_num) + "_bn1")(y)
    y = Activation("relu")(y)
    y = Conv2D(num_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=initializer, use_bias=False,
               name=base_name + "_resblock" + str(block_num) + "_conv2")(y)
    if not is_wgangp:
        y = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_resblock" + str(block_num) + "_bn2")(y)
    return Add()([x, y])


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty(D, input_merged, base_name):
    gradients = K.gradients(D, [input_merged])[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))

    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    out = Network(input=[D, input_merged], output=[gradient_penalty], name=base_name+"_gp")
    return out



def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.

    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!

    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.

    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(1e-8 + gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

def recursive_repr(fillvalue='...'):
    'Decorator to make a repr function return fillvalue for a recursive call'

    def decorating_function(user_function):
        repr_running = set()

        def wrapper(self):
            key = id(self), get_ident()
            if key in repr_running:
                return fillvalue
            repr_running.add(key)
            try:
                result = user_function(self)
            finally:
                repr_running.discard(key)
            return result

        # Can't use functools.wraps() here because of bootstrap issues
        wrapper.__module__ = getattr(user_function, '__module__')
        wrapper.__doc__ = getattr(user_function, '__doc__')
        wrapper.__name__ = getattr(user_function, '__name__')
        wrapper.__qualname__ = getattr(user_function, '__qualname__')
        wrapper.__annotations__ = getattr(user_function, '__annotations__', {})
        return wrapper

    return decorating_function


def get_ident():  # real signature unknown; restored from __doc__
    """
    get_ident() -> integer

    Return a non-zero integer that uniquely identifies the current thread
    amongst other threads that exist simultaneously.
    This may be used to identify per-thread resources.
    Even though on some platforms threads identities may appear to be
    allocated consecutive numbers starting at 1, this behavior should not
    be relied upon, and the number should be seen purely as a magic cookie.
    A thread's identity may be reused for another thread after it exits.
    """
    return 0


class partial:
    """New function with partial application of the given arguments
    and keywords.
    """

    __slots__ = "func", "args", "keywords", "__dict__", "__weakref__"

    def __new__(*args, **keywords):
        if not args:
            raise TypeError("descriptor '__new__' of partial needs an argument")
        if len(args) < 2:
            raise TypeError("type 'partial' takes at least one argument")
        cls, func, *args = args
        if not callable(func):
            raise TypeError("the first argument must be callable")
        args = tuple(args)

        if hasattr(func, "func"):
            args = func.args + args
            tmpkw = func.keywords.copy()
            tmpkw.update(keywords)
            keywords = tmpkw
            del tmpkw
            func = func.func

        self = super(partial, cls).__new__(cls)

        self.func = func
        self.args = args
        self.keywords = keywords
        return self

    def __call__(*args, **keywords):
        if not args:
            raise TypeError("descriptor '__call__' of partial needs an argument")
        self, *args = args
        newkeywords = self.keywords.copy()
        newkeywords.update(keywords)
        return self.func(*self.args, *args, **newkeywords)

    @recursive_repr()
    def __repr__(self):
        qualname = type(self).__qualname__
        args = [repr(self.func)]
        args.extend(repr(x) for x in self.args)
        args.extend(f"{k}={v!r}" for (k, v) in self.keywords.items())
        if type(self).__module__ == "functools":
            return f"functools.{qualname}({', '.join(args)})"
        return f"{qualname}({', '.join(args)})"

    def __reduce__(self):
        return type(self), (self.func,), (self.func, self.args,
               self.keywords or None, self.__dict__ or None)

    def __setstate__(self, state):
        if not isinstance(state, tuple):
            raise TypeError("argument to __setstate__ must be a tuple")
        if len(state) != 4:
            raise TypeError(f"expected 4 items in state, got {len(state)}")
        func, args, kwds, namespace = state
        if (not callable(func) or not isinstance(args, tuple) or
           (kwds is not None and not isinstance(kwds, dict)) or
           (namespace is not None and not isinstance(namespace, dict))):
            raise TypeError("invalid partial state")

        args = tuple(args) # just in case it's a subclass
        if kwds is None:
            kwds = {}
        elif type(kwds) is not dict: # XXX does it need to be *exactly* dict?
            kwds = dict(kwds)
        if namespace is None:
            namespace = {}

        self.__dict__ = namespace
        self.func = func
        self.args = args
        self.keywords = kwds

try:
    from _functools import partial
except ImportError:
    pass


def save_weights(model, path, counter, base_name=""):
    filename = base_name +str(counter) + ".hdf5"
    output_path = os.path.join(path, filename)
    model.save_weights(output_path)