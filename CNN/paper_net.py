import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

L2 = 1e-3


#----------- SE block
def se_block(x, reduction = 16):
    f = x.shape[-1]
    s = layers.GlobalAveragePoolin2D(x)
    s = layers.Dense(max(f // reduction, 1), activation = "relu")(s)
    s = layers.Dense(f, activation = "sigmoid")(s)
    s = layers.reshape((1,1,f))(s)
    return layers.Multiply()([x,s])

#----------- Basic CNN Block
def basic_block(x, filters, downsample=False, use_se = False):
    stride = 2 if downsample else 1
    skip = x

    x = layers.Conv2D(filters, 3, strides = stride, padding="same",
                      kernel_initializer = "he_normal",
                      kernel_regularizer = regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, strides=1, padding = "same",
                      kernel_initializer = "he_normal",
                      kernel_regularizer = regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    if use_se:
        x = se_block()

    # Shortcut when dimensions change
    if downsample or skip.shape[-1] != filters:
        skip = layers.Conv2D(filters, 1, strides=stride, padding = "same",
                             kernel_initializer = "he_normal",
                             kernel_regularizer = regularizers.l2(L2))(skip)
        skip = layers.BatchNormalization()(skip)

    x = layers.Add()([x, skip])
    return layers.ReLU()(x)

#---------- Bottlneck block + SE
def bottneck_block(x, filters, downsample=False, use_se=True):
    stride = 2 if downsample else 1
    skip = x

    x = layers.Conv2D(filters, 1, strides = stride, padding = "same",
                      kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, strides = 1, padding = "same",
                      kernel_regularizer = regularizers.l2(L2))(x)
    x =	layers.BatchNormalization()(x)
    x =	layers.ReLU()(x)
    x = layers.Conv2D(filters*4, 1, strides = 1, padding = "same",
                      kernel_regularizer = regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    if use_se:
        x = se_block
    if downsample or skip.shape[-1] != filter *4:
        skip = layers.Conv2D(filters * 4, 1, strides = stride, padding = "same", kernel_regularizer = regularizers.l2(L2))(skip)
        skip = layers.BatchNormalization()(skip)

    x = layers.Add()([x, skip])
    return layers.ReLU()(x)

#-------- Parametric Builder
def build_resnet(shape = (512, 256, 1), classes = 2, block = 'basic',
                 stem_filters = 6, stages = (8, 16, 32), blocks_per_stage = 1,
                 use_se = False, dropout_rate = 0.30, stem_maxpool = False):

    blk = basic_block if block == "basic" else bottlenck_block

    inputs = layers.Input(shape=shape)
    x = layers.Conv2D(stem_filters, 3, strides = 2, padding = "same",
                      kernel_initializer = "he_normal",
                      kernel_regularizer = regularizers.l2(L2))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if stem_maxpool:
        x = layers.MaxPooling2D(3, strides = 2, padding = "same")(x)

    for i, f in enumerate(stages):
        for b in range(blocks_per_stage):
            downsample = (b == 0 and i > 0) # only first stage-block
            x = blk(x,f,downsample=downsample, use_se=use_se)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(classes, activation = "softmax")(x)
    return models.Model(inputs, outputs)
        
#------- preset

def paper_net(shape = (512, 256, 1), classes = 2):
    # 20k parameters
    return build_resnet(shape, classes, block="basic",
                        stem_filters = 8, stages = (8,16,32),
                        blocks_per_stage = 1)

def small_se_net(shape=(512, 256, 1), classes=2):
    # 20k params with bottleneck+SE
    return build_resnet(shape, classes, block="bottleneck",
                        stem_filters=32, stages=(16,), blocks_per_stage=3,
                        use_se=True, stem_maxpool=True)

def wide_net(shape=(512,256,1), classes = 2):
    # stronger net, you need GPU (but can handle bigger images)
    return build_resnet(shape, classes, block = "bottleneck",
                        stem_filters = 64, stages = (64, 128),
                        block_per_stage = 3,
                        use_se = True, stem_maxpool = True)

if __name__ == "__main__":
    for name, fn in [("paper_net", paper_net),
                     ("small_se_net", small_se_net),
                     ("wide_net", wide_net)]:
        print(f"{name:14s} {fn().count_params():>10,d} params")
