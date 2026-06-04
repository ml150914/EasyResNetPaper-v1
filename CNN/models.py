import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import Lambda

"""---------------------- CNN/ResNet Models ----------------
    This macros contains several possible appliable CNN/ResNet models
    along with the basic bricks that can constitutes custom model

"""

#---------->se-block: arXiv:1709.01507 [cs.CV]
def se_block(input_tensor, reduction=16):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // reduction, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([input_tensor, se])

#----------> Bottlneck block
def bottleneck_block(x, filters, strides=1, projection_shortcut=False):
    shortcut = x

    # 1x1 Convolution (Reduce)
    x = layers.Conv2D(filters, (1, 1), strides=strides, padding='same',
                      kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 3x3 Convolution
    x = layers.Conv2D(filters, (3, 3), strides=1, padding='same',
                      kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 1x1 Convolution (Expand)
    x = layers.Conv2D(filters * 4, (1, 1), strides=1, padding='same',
                      kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)

    x = se_block(x)
    
    # Projection shortcut if dimensions change
    if projection_shortcut:
        shortcut = layers.Conv2D(filters * 4, (1, 1), strides=strides, padding='same',
                                 kernel_regularizer=regularizers.l2(1e-3))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add shortcut and main path
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

#---------> residual_block
def residual_block(x, number_of_filters, match_filter_size=False, shortcut_type = 'identity'):
    """
	    Residual block with
    """
    initializer = tf.keras.initializers.HeNormal()  # define once
    # Create skip connection
    x_skip = x
    
    # Perform the original mapping
    if match_filter_size:
        x = layers.Conv2D(number_of_filters, kernel_size=(3, 3), strides=(2,2), kernel_initializer=initializer, padding="same")(x_skip)
    else:
        x = layers.Conv2D(number_of_filters, kernel_size=(3, 3), strides=(1,1), kernel_initializer=initializer, padding="same")(x_skip)

    x = layers.BatchNormalization(axis=3)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(number_of_filters, kernel_size=(3, 3), kernel_initializer=initializer, padding="same")(x)
    x = layers.BatchNormalization(axis=3)(x)

    # Shortcut path: projection always when match_filter_size is True
    if match_filter_size:
        x_skip = layers.Conv2D(number_of_filters, kernel_size=(1, 1), strides=(2, 2),
                               kernel_initializer=initializer, padding="same")(x_skip)


    # Add the skip connection to the regular mapping
    x = layers.Add()([x, x_skip])

    # Nonlinearly activate the result
    x = layers.Activation("relu")(x)
    
    # Return the result
    return x

#------> Multiple Residual Block
def ResidualBlocks(x, filters, stack_n, shortcut_type='identity'):
	"""
		Set up the residual blocks.
	"""
		
	# Set initial filter size
	filter_size = filters

	# Paper: "Then we use a stack of 6n layers (...)
	#	with 2n layers for each feature map size."
	# 6n/2n = 3, so there are always 3 groups.
	for layer_group in range(3):

		# Each block in our code has 2 weighted layers,
		# and each group has 2n such blocks,
		# so 2n/2 = n blocks per group.
		for block in range(stack_n):

			# Perform filter size increase at every
			# first layer in the 2nd block onwards.
			# Apply Conv block for projecting the skip
			# connection.
			if layer_group > 0 and block == 0:
				filter_size *= 2
				x = residual_block(x, filter_size, match_filter_size=True, shortcut_type=shortcut_type)
			else:
				x = residual_block(x, filter_size, match_filter_size=True, shortcut_type=shortcut_type)

	# Return final layer
	return x
    
#--------> Custom ResNet Model
#ResNet-Scratch
def ResNetScratch(shape=(128, 256, 1), classes=2, stack_n=2, initial_filters=32, shortcut_type='identity', data_augmentation = None):
    """
	    Base structure of the model, with residual blocks
	    attached.
    """
    
    # Define model structure
    # logits are returned because Softmax is pushed to loss function.
    inputs = layers.Input(shape=shape)
    x = inputs 
    x = layers.Conv2D(initial_filters, kernel_size=(3, 3), strides=(2, 2), 
		      padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-3))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = ResidualBlocks(x, filters=initial_filters, stack_n=stack_n, shortcut_type=shortcut_type)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model
	


#SmallNet
def SmallNet(shape=(128, 256, 1), classes=2, data_augmentation=None, dropout_rate = 0.30):
     inputs = layers.Input(shape=shape)
     x = inputs
     x = layers.Conv2D(32, (7, 7), strides=2, padding='same',
                       kernel_regularizer=regularizers.l2(1e-3))(x)
     x = layers.BatchNormalization()(x)
     x = layers.ReLU()(x)
     x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

     # Stage 1 (2 residual blocks → 4 conv layers)
     x = bottleneck_block(x, 16, projection_shortcut=True)
     x = bottleneck_block(x, 16)
     x = bottleneck_block(x, 16)

     # Global Average Pooling and Dense Layer
     x = layers.GlobalAveragePooling2D()(x)
     x = layers.Dropout(dropout_rate)(x)
     outputs = layers.Dense(classes, activation='softmax')(x)
     model = models.Model(inputs=inputs, outputs=outputs)
     return model


#EasyResNet
def EasyResNet(shape=(128, 256, 1), classes=2, data_augmentation=None, dropout_rate = 0.30):
    inputs = layers.Input(shape=shape)
    x = inputs
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same',
                      kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Stage 1 (2 residual blocks → 4 conv layers)
    x = residual_block(x, 64, projection_shortcut=True)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    # Global Average Pooling and Dense Layer
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

#EasyResNet1                                                                                            
def EasyResNet1(shape=(128, 256, 1), classes=2, data_augmentation=None, dropout_rate = 0.25):
    inputs = layers.Input(shape=shape)
    x = inputs
    x = layers.Conv2D(128, (7, 7), strides=2, padding='same',
                      kernel_regularizer=regularizers.l2(1e-2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Stage 1 (2 residual blocks → 4 conv layers)
    x = bottleneck_block(x, 64, projection_shortcut=True)
    x = bottleneck_block(x, 64)
    x = bottleneck_block(x, 64)

    # Global Average Pooling and Dense Layer
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

#EasyResNet2
def EasyResNet2(shape=(128, 256, 1), classes=2, data_augmentation=None, dropout_rate = 0.25):
    inputs = layers.Input(shape=shape)
    x = inputs
    x = layers.Conv2D(128, (7, 7), strides=2, padding='same',
                      kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Stage 1 (2 residual blocks → 4 conv layers)
    x = bottleneck_block(x, 128, projection_shortcut=True)
    x = bottleneck_block(x, 128)
    x = bottleneck_block(x, 128)

    # Global Average Pooling and Dense Layer
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

#EasyResNet3
def EasyResNet3(shape=(128, 256, 1), classes=2, data_augmentation=None, dropout_rate = 0.30):
    inputs = layers.Input(shape=shape)
    x = inputs
    x = layers.Conv2D(32, (7, 7), strides=2, padding='same',
                      kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Stage 1 (2 residual blocks → 4 conv layers)
    x = bottleneck_block(x, 32, projection_shortcut=True)
    x = bottleneck_block(x, 32)
    x = bottleneck_block(x, 32)

    # Global Average Pooling and Dense Layer
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

#EasyResNet4
def EasyResNet4(shape=(128, 256, 1), classes=2, data_augmentation=None, dropout_rate = 0.30):
    inputs = layers.Input(shape=shape)
    x = inputs
    x = layers.Conv2D(32, (7, 7), strides=2, padding='same',
                      kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Stage 1 (2 residual blocks → 4 conv layers)
    x = bottleneck_block(x, 64, projection_shortcut=True)
    x = bottleneck_block(x, 64)
    x = bottleneck_block(x, 64)

    # Global Average Pooling and Dense Layer
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

#------- ResNet16
def ResNet16(shape=(128, 256, 1), classes=2, data_augmentation=None, dropout_rate=0.30):
    inputs = layers.Input(shape=shape)
    x = inputs
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same',
                      kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
     
    # Stage 1 (2 residual blocks → 4 conv layers)
    x = bottleneck_block(x, 64, projection_shortcut=True)
    x = bottleneck_block(x, 64)
    x = bottleneck_block(x, 64)
    
    # Stage 2 (2 residual blocks → 4 conv layers)
    x = bottleneck_block(x, 128, projection_shortcut=True)
    x = bottleneck_block(x, 128)
    x = bottleneck_block(x, 128)

    # Global Average Pooling and Dense Layer
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

#------ResNet16-smaller
def ResNet16_smaller(shape=(128, 256, 1), classes=2, data_augmentation=None, dropout_rate=0.30):
    inputs = layers.Input(shape=shape)
    x = inputs
    x = layers.Conv2D(16, (7, 7), strides=2, padding='same',
                      kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Stage 1 (2 residual blocks → 4 conv layers)
    x = bottleneck_block(x, 32, projection_shortcut=True)
    x = bottleneck_block(x, 32)
    x = bottleneck_block(x, 32)

    # Global Average Pooling and Dense Layer
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model
#------ResNet16-var1
def ResNet16_var1(shape=(128, 256, 1), classes=2, data_augmentation=None, dropout_rate=0.30):
    inputs = layers.Input(shape=shape)
    x = inputs
    x = layers.Conv2D(32, (7, 7), strides=2, padding='same',
                      kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Stage 1 (2 residual blocks → 4 conv layers)
    x = bottleneck_block(x, 64, projection_shortcut=True)
    x = bottleneck_block(x, 64)
    x = bottleneck_block(x, 64)

    # Global Average Pooling and Dense Layer
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


#------- ResNet27
def ResNet27(shape=(128, 256, 1), classes=2, data_augmentation=None, dropout_rate=0.30):
     inputs = layers.Input(shape=shape)
     x = inputs
     x = layers.Conv2D(64, (7, 7), strides=2, padding='same',
                       kernel_regularizer=regularizers.l2(1e-3))(x)
     x = layers.BatchNormalization()(x)
     x = layers.ReLU()(x)
     x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

     # Stage 1 (2 residual blocks → 4 conv layers)
     x = bottleneck_block(x, 64, projection_shortcut=True)
     x = bottleneck_block(x, 64)
     x = bottleneck_block(x, 64)

     # Stage 2 (2 residual blocks → 4 conv layers)
     x = bottleneck_block(x, 128, projection_shortcut=True)
     x = bottleneck_block(x, 128)
     x = bottleneck_block(x, 128)

     # Stage 2 (2 residual blocks → 4 conv layers)
     x = bottleneck_block(x, 256, projection_shortcut=True)
     x = bottleneck_block(x, 256)
     x = bottleneck_block(x, 256)
     
     # Global Average Pooling and Dense Layer
     x = layers.GlobalAveragePooling2D()(x)
     x = layers.Dropout(dropout_rate)(x)
     outputs = layers.Dense(classes, activation='softmax')(x)

     model = models.Model(inputs=inputs, outputs=outputs)
     return model

#DeepNet
def DeepNet(shape=(128, 256, 1), classes=2, data_augmentation=None, dropout_rate=0.2):
    inputs = layers.Input(shape=shape)
    x = inputs

    # Initial convolutional layer
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same',
                      kernel_regularizer=regularizers.l2(5e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Stage 1
    x = bottleneck_block(x, 64, projection_shortcut=True)
    x = bottleneck_block(x, 64)
    x = bottleneck_block(x, 64)

    # Stage 2
    x = bottleneck_block(x, 128, projection_shortcut=True)
    x = bottleneck_block(x, 128)
    x = bottleneck_block(x, 128)

    # Stage 3
    x = bottleneck_block(x, 256, projection_shortcut=True)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)

    # Stage 4
    x = bottleneck_block(x, 512, projection_shortcut=True)
    x = bottleneck_block(x, 512)
    x = bottleneck_block(x, 512)

    # Global Average Pooling and Dropout before output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)  # Dropout before dense layer
    outputs = layers.Dense(classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model
