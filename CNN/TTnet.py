"""
TT-map classifier: BNS injection vs. noise.

Rebuilt from measurements on real maps rather than spectrogram intuition:

  * maps are 256 x 512 uint8, grey levels ~56-127, sigma ~6.4
  * the injected feature is a nearly-flat ridge, ~6 px (axis-1) x ~97 px
    (axis-2) = 0.12% of the 131,072 pixels, peaking ~6 sigma above background
  * the background is strongly anisotropic: ~64% of its power sits in a 30 deg
    wedge, i.e. it is itself made of oriented streaks
  * the field is close to stationary: row-mean spread is 0.71 vs a pixel std of
    6.33, and the end-to-end drift along axis-1 is ~0.2 sigma

Consequences baked into `ttmap_net` below:
  * elongated oriented stem kernels, because the discriminant is orientation
    contrast against the ambient streak texture, not brightness alone
  * axis-1 resolution held to 4x total downsampling (the ridge is only 6 px
    thick); axis-2 downsampled 16x (the ridge is 97 px long)
  * GeM head, because global average pooling divides a 0.12%-occupancy feature
    by ~800
  * SE blocks in the deeper stages to supply the global "how much ambient
    texture is there at this orientation" context that the local ridge
    detector cannot see
  * no per-row normalization and no coordinate channel: measured unnecessary
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers

L2 = 1e-4


# --------------------------------------------------------------------------- #
#  Custom layers
# --------------------------------------------------------------------------- #
@keras.utils.register_keras_serializable(package="ttmap")
class StandardizePerImage(layers.Layer):
    """(x - mean) / std per map.

    Preferable to Rescaling(1/255) here: the maps only occupy grey levels
    ~56-127, so 1/255 compresses everything into [0.22, 0.50] and leaves the
    first BatchNorm to undo it. Also removes any per-map gain drift for free.
    """

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = float(eps)

    def call(self, x):
        mu = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        sd = tf.math.reduce_std(x, axis=[1, 2, 3], keepdims=True)
        return (x - mu) / (sd + self.eps)

    def get_config(self):
        cfg = super().get_config(); cfg.update(eps=self.eps); return cfg


@keras.utils.register_keras_serializable(package="ttmap")
class GeMPooling2D(layers.Layer):
    """Generalized-mean pooling: (mean(x^p))^(1/p).

    p -> 1 is global average pooling, p -> inf is global max. With the feature
    covering 0.12% of the map, GAP is the wrong end of that family.
    """

    def __init__(self, p=3.0, learnable=True, eps=1e-6, clip=1e4, **kwargs):
        super().__init__(**kwargs)
        self.init_p = float(p); self.learnable = bool(learnable)
        self.eps = float(eps); self.clip = float(clip)

    def build(self, input_shape):
        self.p = self.add_weight(
            name="p", shape=(),
            initializer=keras.initializers.Constant(self.init_p),
            trainable=self.learnable)
        super().build(input_shape)

    def call(self, x):
        x = tf.clip_by_value(x, self.eps, self.clip)
        p = tf.maximum(self.p, 1.0)
        return tf.pow(tf.reduce_mean(tf.pow(x, p), axis=[1, 2]), 1.0 / p)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update(p=self.init_p, learnable=self.learnable,
                   eps=self.eps, clip=self.clip)
        return cfg


@keras.utils.register_keras_serializable(package="ttmap")
class RowNorm(layers.Layer):
    """Per-row standardization across axis-2. Off by default: measured
    unnecessary on these maps (row-mean spread is 11% of the pixel std).
    Keep it if your real dataset has a non-stationary background."""

    def __init__(self, axis=2, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.axis = int(axis); self.eps = float(eps)

    def call(self, x):
        mu = tf.reduce_mean(x, axis=self.axis, keepdims=True)
        sd = tf.math.reduce_std(x, axis=self.axis, keepdims=True)
        return (x - mu) / (sd + self.eps)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(axis=self.axis, eps=self.eps); return cfg


# --------------------------------------------------------------------------- #
#  Blocks
# --------------------------------------------------------------------------- #
def se_block(x, reduction=4):
    """Squeeze-and-excitation. reduction=4, not 16: at 16-40 channels a
    reduction of 16 leaves a 1-2 unit bottleneck, which is not a bottleneck."""
    f = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(max(f // reduction, 4), activation="relu")(s)
    s = layers.Dense(f, activation="sigmoid")(s)
    s = layers.Reshape((1, 1, f))(s)
    return layers.Multiply()([x, s])


def _shortcut(skip, filters, stride, l2):
    if tuple(stride) != (1, 1) or skip.shape[-1] != filters:
        skip = layers.Conv2D(filters, 1, strides=stride, padding="same",
                             kernel_initializer="he_normal",
                             kernel_regularizer=regularizers.l2(l2))(skip)
        skip = layers.BatchNormalization()(skip)
    return skip


def basic_block(x, filters, stride=(1, 1), use_se=False, kernel=3,
                l2=L2, se_reduction=4):
    skip = x
    for i, st in enumerate((stride, (1, 1))):
        x = layers.Conv2D(filters, kernel, strides=st, padding="same",
                          kernel_initializer="he_normal",
                          kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        if i == 0:
            x = layers.ReLU()(x)
    if use_se:
        x = se_block(x, se_reduction)
    x = layers.Add()([x, _shortcut(skip, filters, stride, l2)])
    return layers.ReLU()(x)


def bottleneck_block(x, filters, stride=(1, 1), use_se=True, kernel=3,
                     l2=L2, se_reduction=4):
    out = filters * 4
    skip = x
    x = layers.Conv2D(filters, 1, padding="same", kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel, strides=stride, padding="same",
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(out, 1, padding="same", kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    if use_se:
        x = se_block(x, se_reduction)
    x = layers.Add()([x, _shortcut(skip, out, stride, l2)])
    return layers.ReLU()(x)


def separable_block(x, filters, stride=(1, 1), use_se=False, kernel=3,
                    l2=L2, se_reduction=4):
    """Depthwise-separable residual block: 9f + f*f' instead of 9*f*f'.
    At a fixed 20k budget this buys roughly 3x the depth."""
    stride = tuple(stride)
    if stride != (1, 1):
        # SeparableConv2D rejects unequal row/col strides, so downsample with
        # average pooling. It also anti-aliases, which strided convs do not.
        x = layers.AveragePooling2D(stride, strides=stride, padding="same")(x)
    skip = x
    for i in range(2):
        x = layers.SeparableConv2D(filters, kernel, padding="same",
                                   depthwise_initializer="he_normal",
                                   pointwise_initializer="he_normal",
                                   pointwise_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        if i == 0:
            x = layers.ReLU()(x)
    if use_se:
        x = se_block(x, se_reduction)
    x = layers.Add()([x, _shortcut(skip, filters, (1, 1), l2)])
    return layers.ReLU()(x)


BLOCKS = {"basic": basic_block, "bottleneck": bottleneck_block,
          "separable": separable_block}


# --------------------------------------------------------------------------- #
#  Builder
# --------------------------------------------------------------------------- #
def build_net(shape=(256, 512, 1), classes=2, block="separable",
              stem_filters=16, stem_kernel=(5, 15), stem_stride=(1, 2),
              stages=(16, 24, 40), stage_strides=((1, 2), (2, 2), (2, 2)),
              blocks_per_stage=2, kernel=(3, 5), use_se=False,
              se_from_stage=None, se_reduction=4, dropout_rate=0.2,
              stem_maxpool=False, head="gem", gem_p=3.0,
              standardize=True, row_norm=False, l2=L2, name=None):
    """
    stage_strides  : per-stage (s_axis1, s_axis2) applied to the first block.
    se_from_stage  : enable SE only from this stage index onward (e.g. 1).
                     Overrides use_se when set.
    head           : "gem" | "avgmax" | "gap"
    """
    if block not in BLOCKS:
        raise ValueError(f"block must be one of {list(BLOCKS)}, got {block!r}")
    if len(stage_strides) != len(stages):
        raise ValueError("stage_strides must be the same length as stages")
    blk = BLOCKS[block]

    inputs = layers.Input(shape=shape)
    x = inputs
    if standardize:
        x = StandardizePerImage()(x)
    if row_norm:
        x = RowNorm()(x)

    x = layers.Conv2D(stem_filters, stem_kernel, strides=stem_stride,
                      padding="same", kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if stem_maxpool:
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    for i, (f, stride) in enumerate(zip(stages, stage_strides)):
        se = use_se if se_from_stage is None else (i >= se_from_stage)
        for b in range(blocks_per_stage):
            x = blk(x, f, stride=stride if b == 0 else (1, 1), use_se=se,
                    kernel=kernel, l2=l2, se_reduction=se_reduction)

    if head == "gem":
        x = GeMPooling2D(p=gem_p)(x)
    elif head == "avgmax":
        x = layers.Concatenate()([layers.GlobalAveragePooling2D()(x),
                                  layers.GlobalMaxPooling2D()(x)])
    else:
        x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name=name)


# --------------------------------------------------------------------------- #
#  Presets
# --------------------------------------------------------------------------- #
def paper_net(shape=(256, 512, 1), classes=2):
    """Original baseline, ~20k params, unchanged apart from bug fixes.
    Kept so the comparison is like-for-like."""
    return build_net(shape, classes, block="basic", stem_filters=8,
                     stem_kernel=3, stem_stride=(2, 2), kernel=3,
                     stages=(8, 16, 32), stage_strides=((1, 1), (2, 2), (2, 2)),
                     blocks_per_stage=1, head="gap", standardize=False,
                     dropout_rate=0.30, l2=1e-3, name="paper_net")


def ttmap_net(shape=(256, 512, 1), classes=2):
    """Recommended. Same parameter budget, matched to the measured geometry.
    Receptive field 65 x 799 px against a 6 x 97 px feature.

    The stem strides (2,4) rather than (1,2): a strided convolution still
    evaluates its kernel at full input resolution and only subsamples the
    response, and the response of a (5,15) ridge filter is smooth on that
    scale. Measured on the real map, stride (2,4) retains 99.7% of the ridge
    peak while cutting the model from 193 to 61 MMACs. Do not go to stride 4
    on axis-1: that drops the peak to 81%."""
    return build_net(shape, classes, block="separable",
                     stem_filters=16, stem_kernel=(5, 15), stem_stride=(2, 4),
                     stages=(16, 24, 40),
                     stage_strides=((1, 2), (1, 2), (2, 2)),
                     blocks_per_stage=2, kernel=(3, 5),
                     se_from_stage=1, head="gem", name="ttmap_net")


def wide_net(shape=(256, 512, 1), classes=2):
    """Same design, no parameter budget. Use when a GPU is available."""
    return build_net(shape, classes, block="separable",
                     stem_filters=32, stem_kernel=(5, 15), stem_stride=(2, 4),
                     stages=(32, 64, 96, 128),
                     stage_strides=((1, 2), (1, 2), (2, 2), (1, 2)),
                     blocks_per_stage=3, kernel=(3, 5),
                     se_from_stage=1, head="gem", dropout_rate=0.3,
                     name="wide_net")


if __name__ == "__main__":
    for name, fn in [("paper_net", paper_net), ("ttmap_net", ttmap_net),
                     ("wide_net", wide_net)]:
        m = fn()
        d = sum(1 for l in m.layers
                if isinstance(l, (layers.Conv2D, layers.SeparableConv2D)))
        print(f"{name:12s} {m.count_params():>10,d} params   {d:>3d} conv layers")
