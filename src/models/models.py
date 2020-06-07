import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Input, LeakyReLU, BatchNormalization, \
                                    Activation, Add, GlobalAveragePooling2D, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import Constant
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import yaml
import os

def resnet50v2(model_config, input_shape, metrics, n_classes, mixed_precision=False, output_bias=None):
    '''
    Defines a model based on a pretrained ResNet50V2 for multiclass X-ray classification.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param n_classes: # of classes in data
    :param mixed_precision: Whether to use mixed precision (use if you have GPU with compute capacity >= 7.0)
    :param output_bias: bias initializer of output layer
    :return: a Keras Model object with the architecture defined in this method
    '''

    # Set hyperparameters
    nodes_dense0 = model_config['NODES_DENSE0']
    nodes_dense1 = model_config['NODES_DENSE1']
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    l2_lambda = model_config['L2_LAMBDA']
    optimizer = Adam(learning_rate=lr)
    if mixed_precision:
        tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
    print("MODEL CONFIG: ", model_config)

    if output_bias is not None:
        output_bias = Constant(output_bias)     # Set initial output bias

    # Start with pretrained ResNet50V2
    X_input = Input(input_shape, name='input')
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)
    for layer in base_model.layers:
        if 'conv5' not in layer.name and 'conv4' not in layer.name:
            layer.trainable = False
    X = base_model.output

    # Add custom top layers
    X = GlobalAveragePooling2D()(X)
    X = Dropout(dropout)(X)
    X = Dense(nodes_dense0, kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda))(X)
    X = LeakyReLU()(X)
    X = Dropout(dropout)(X)
    X = Dense(nodes_dense1, kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda))(X)
    X = LeakyReLU()(X)
    X = Dense(n_classes, bias_initializer=output_bias)(X)
    Y = Activation('softmax', dtype='float32', name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model


def resnet101v2(model_config, input_shape, metrics, n_classes, mixed_precision=False, output_bias=None):
    '''
    Defines a model based on a pretrained ResNet50V2 for multiclass X-ray classification.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param n_classes: # of classes in data
    :param mixed_precision: Whether to use mixed precision (use if you have GPU with compute capacity >= 7.0)
    :param output_bias: bias initializer of output layer
    :return: a Keras Model object with the architecture defined in this method
    '''

    # Set hyperparameters
    nodes_dense0 = model_config['NODES_DENSE0']
    nodes_dense1 = model_config['NODES_DENSE1']
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    l2_lambda = model_config['L2_LAMBDA']
    optimizer = Adam(learning_rate=lr)
    print("MODEL CONFIG: ", model_config)
    if mixed_precision:
        tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
        
    if output_bias is not None:
        output_bias = Constant(output_bias)     # Set initial output bias

    # Start with pretrained ResNet101V2
    X_input = Input(input_shape, name='input')
    base_model = ResNet101V2(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)
    X = base_model.output

    # Add custom top layers
    X = GlobalAveragePooling2D()(X)
    X = Dropout(dropout)(X)
    X = Dense(nodes_dense0, kernel_initializer='he_uniform', activation='relu', activity_regularizer=l2(l2_lambda))(X)
    X = Dropout(dropout)(X)
    X = Dense(nodes_dense1, kernel_initializer='he_uniform', activation='relu', activity_regularizer=l2(l2_lambda))(X)
    X = Dense(n_classes, bias_initializer=output_bias)(X)
    Y = Activation('softmax', dtype='float32', name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model


def inceptionv3(model_config, input_shape, metrics, n_classes, mixed_precision=False, output_bias=None):
    '''
    Defines a model based on a pretrained InceptionV3 for multiclass US classification.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param n_classes: # of classes in data
    :param mixed_precision: Whether to use mixed precision (use if you have GPU with compute capacity >= 7.0)
    :param output_bias: bias initializer of output layer
    :return: a Keras Model object with the architecture defined in this method
    '''

    # Set hyperparameters
    nodes_dense0 = model_config['NODES_DENSE0']
    nodes_dense1 = model_config['NODES_DENSE1']
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    l2_lambda = model_config['L2_LAMBDA']
    if model_config['OPTIMIZER'] == 'sgd':
        optimizer = SGD(learning_rate=lr, momentum=0.9)
    else:
        optimizer = Adam(learning_rate=lr)
    print("MODEL CONFIG: ", model_config)
    if mixed_precision:
        tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    if output_bias is not None:
        output_bias = Constant(output_bias)     # Set initial output bias

    # Start with pretrained InceptionV3
    X_input = Input(input_shape, name='input')
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)
    for layer in base_model.layers[:290]:
        layer.trainable = False
    for layer in base_model.layers[290:]:
        layer.trainable = True
        if 'conv' in layer.name:
            setattr(layer, 'activity_regularizer', l2(l2_lambda))
    X = base_model.output

    # Add custom top layers
    X = GlobalAveragePooling2D()(X)
    X = Dropout(dropout)(X)
    X = Dense(nodes_dense0, activation='relu', activity_regularizer=l2(l2_lambda))(X)
    #X = LeakyReLU()(X)
    X = BatchNormalization()(X)
    #X = Dropout(dropout)(X)
    #X = Dense(nodes_dense1, activity_regularizer=l2(l2_lambda))(X)
    #X = LeakyReLU()(X)
    X = Dropout(dropout)(X)
    X = Dense(n_classes, bias_initializer=output_bias)(X)
    Y = Activation('softmax', dtype='float32', name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model


def mobilenetv2(model_config, input_shape, metrics, n_classes, mixed_precision=False, output_bias=None):
    '''
    Defines a model based on a pretrained MobileNetV2 for multiclass US classification.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param n_classes: # of classes in data
    :param mixed_precision: Whether to use mixed precision (use if you have GPU with compute capacity >= 7.0)
    :param output_bias: bias initializer of output layer
    :return: a Keras Model object with the architecture defined in this method
    '''

    # Set hyperparameters
    nodes_dense0 = model_config['NODES_DENSE0']
    nodes_dense1 = model_config['NODES_DENSE1']
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    l2_lambda = model_config['L2_LAMBDA']
    if model_config['OPTIMIZER'] == 'sgd':
        optimizer = SGD(learning_rate=lr, momentum=0.9)
    else:
        optimizer = Adam(learning_rate=lr)
    print("MODEL CONFIG: ", model_config)
    if mixed_precision:
        tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    if output_bias is not None:
        output_bias = Constant(output_bias)     # Set initial output bias

    # Start with pretrained MobileNetV2
    X_input = Input(input_shape, name='input')
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)
    
    for layer in base_model.layers:
        if 'Conv_1' in layer.name:
            layer.trainable = True
            if 'Conv_1' == layer.name:
                setattr(layer, 'activity_regularizer', l2(l2_lambda))
            print("Trainable layer", layer.name)
        else:
            layer.trainable = False
    X = base_model.output

    # Add custom top layers
    X = GlobalAveragePooling2D()(X)
    X = Dropout(dropout)(X)
    X = Dense(nodes_dense0, activation='relu', activity_regularizer=l2(l2_lambda))(X)
    #X = LeakyReLU()(X)
    X = BatchNormalization()(X)
    #X = Dropout(dropout)(X)
    #X = Dense(nodes_dense1, activity_regularizer=l2(l2_lambda))(X)
    #X = LeakyReLU()(X)
    X = Dropout(dropout)(X)
    X = Dense(n_classes, bias_initializer=output_bias)(X)
    Y = Activation('softmax', dtype='float32', name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model
    

def vgg16(model_config, input_shape, metrics, n_classes, mixed_precision=False, output_bias=None):
    '''
    Defines a model based on a pretrained ResNet50V2 for multiclass X-ray classification.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param n_classes: # of classes in data
    :param mixed_precision: Whether to use mixed precision (use if you have GPU with compute capacity >= 7.0)
    :param output_bias: bias initializer of output layer
    :return: a Keras Model object with the architecture defined in this method
    '''

    # Set hyperparameters
    nodes_dense0 = model_config['NODES_DENSE0']
    nodes_dense1 = model_config['NODES_DENSE1']
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    l2_lambda = model_config['L2_LAMBDA']
    optimizer = Adam(learning_rate=lr)
    frozen_layers = model_config['FROZEN_LAYERS']
    print("MODEL CONFIG: ", model_config)
    if mixed_precision:
        tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    if output_bias is not None:
        output_bias = Constant(output_bias)     # Set initial output bias

    # Start with pretrained VGG16
    X_input = Input(input_shape, name='input')
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)
    
    # Freeze desired conv layers set in config.yml
    for layers in range(len(frozen_layers)):
        layer2freeze = frozen_layers[layers]
        print('Freezing layer: ' + str(layer2freeze))
        base_model.layers[layer2freeze].trainable = False

    # Add regularization to VGG16 conv layers
    for layer in base_model.layers:
        idx = 0
        if base_model.layers[0].trainable and 'conv' in layer.name:
            setattr(layer, 'activity_regulizer', l2(l2_lambda))
            print('Adding regularization to: ' + str(base_model.layers[layers]))
        idx += 1
    
    X = base_model.output

    # Add custom top layers
    X = GlobalAveragePooling2D()(X)
    X = Dropout(dropout)(X)
    X = Dense(nodes_dense0, kernel_initializer='he_uniform', activation='relu', activity_regularizer=l2(l2_lambda))(X)
    X = Dropout(dropout)(X)
    #X = Dense(nodes_dense1, kernel_initializer='he_uniform', activation='relu', activity_regularizer=l2(l2_lambda))(X)
    X = Dense(n_classes, bias_initializer=output_bias)(X)
    Y = Activation('softmax', dtype='float32', name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model

def conv_block(units, dropout=0.2, activation='relu', block=1, layer=1):

    def layer_wrapper(inp):
        x = Conv2D(units, (3, 3), padding='same', name='block{}_conv{}'.format(block, layer))(inp)
        x = BatchNormalization(name='block{}_bn{}'.format(block, layer))(x)
        #x = Activation(activation, name='block{}_act{}'.format(block, layer))(x)
        x = LeakyReLU()(x)
        return x

    return layer_wrapper
        

def custom_vgg16(model_config, input_shape, metrics, n_classes, mixed_precision = False, output_bias = None, activation='relu'):
    """Instantiates the VGG16 architecture with Batch Normalization
    :param input_tensor: Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
    :param input_shape: shape tuple
    :param classes: optional number of classes to classify images
    :returnL: a VGG16 Keras model with BN & dropout
    """

    # Set hyperparameters
    nodes_dense0 = model_config['NODES_DENSE0']
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    l2_lambda = model_config['L2_LAMBDA']
    optimizer = Adam(learning_rate=lr)
    frozen_layers = model_config['FROZEN_LAYERS']
    print("MODEL CONFIG: ", model_config)
    
    if mixed_precision:
        tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    if output_bias is not None:
        output_bias = Constant(output_bias)     # Set initial output bias

    X_input = Input(input_shape, name='input')

    # Block 1
    X = conv_block(32, dropout=dropout, activation=activation, block=1, layer=1)(X_input)
    X = conv_block(32, dropout=dropout, activation=activation, block=1, layer=2)(X)
    X = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')(X)

    # Block 2
    X = conv_block(64, dropout=dropout, activation=activation, block=2, layer=1)(X)
    X = conv_block(64, dropout=dropout, activation=activation, block=2, layer=2)(X)
    X = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool')(X)

    # Block 3
    X = conv_block(128, dropout=dropout, activation=activation, block=3, layer=1)(X)
    X = conv_block(128, dropout=dropout, activation=activation, block=3, layer=2)(X)
    X = conv_block(128, dropout=dropout, activation=activation, block=3, layer=3)(X)
    X = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool')(X)

    # Block 4
    X = conv_block(256, dropout=dropout, activation=activation, block=4, layer=1)(X)
    X = conv_block(256, dropout=dropout, activation=activation, block=4, layer=2)(X)
    X = conv_block(256, dropout=dropout, activation=activation, block=4, layer=3)(X)
    X = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool')(X)

    # Block 5
    X = conv_block(256, dropout=dropout, activation=activation, block=5, layer=1)(X)
    X = conv_block(256, dropout=dropout, activation=activation, block=5, layer=2)(X)
    X = conv_block(256, dropout=dropout, activation=activation, block=5, layer=3)(X)
    X = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool')(X)

    # Add custom top layers
    X = GlobalAveragePooling2D()(X)
    X = Dropout(dropout)(X)
    X = Dense(nodes_dense0, kernel_initializer='he_uniform', activation='relu', activity_regularizer=l2(l2_lambda))(X)
    X = Dropout(dropout)(X)
    X = Dense(n_classes, bias_initializer=output_bias)(X)
    Y = Activation('softmax', dtype='float32', name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)

    # Add regularization to VGG16 conv layers
    '''
    for layer in model.layers:
        idx = 0
        if 'conv' in layer.name:
            setattr(layer, 'activity_regulizer', l2(l2_lambda))
            print('Adding regularization to: ' + str(base_model.layers[layers]))
        idx += 1
    '''

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model

def convolutional_block(X, kernel_size, filters, stage, block, s=2):
    '''
    Implementation of a convolutional block to be used in a custom ResNet
    :param X: input tensor
    :param kernel_size: kernel size for middle convolutional layer
    :param filters: list the number of filters in the CONV2D layers of the main path
    :param stage: a number for naming the layers, depending on their position in the network
    :param block: to name the layers, depending on their position in the network
    :param s: stride in the CONV2D layers
    :return output of the convolutional block
    '''

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    f1, f2, f3 = filters

    X_res = X # Residual connection
    X = Conv2D(f1, (1, 1), strides=s, name=conv_name_base + '2a', kernel_initializer='he_uniform')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = LeakyReLU()(X)

    X = Conv2D(f2, kernel_size, strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer='he_uniform')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = LeakyReLU()(X)

    X = Conv2D(f3, (1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer='he_uniform')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_res = Conv2D(f3, (1, 1), strides=s, padding='valid', name=conv_name_base + '1',
                        kernel_initializer='he_uniform')(X_res)
    X_res = BatchNormalization(axis=3, name=bn_name_base + '1')(X_res)

    X = Add()([X, X_res])
    X = LeakyReLU()(X)
    return X


def identity_block(X, kernel_size, filters, stage, block):
    '''
    Implementation of an identify block to be used in a custom ResNet
    :param X: input tensor
    :param kernel_size: kernel size for middle convolutional layer
    :param filters: list the number of filters in the CONV2D layers of the main path
    :param stage: a number for naming the layers, depending on their position in the network
    :param block: to name the layers, depending on their position in the network
    :return output of the convolutional block
    '''

    # Define naming strategy
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    f1, f2, f3 = filters

    X_res = X   # Residual connection
    X = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer='he_uniform')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = LeakyReLU()(X)

    X = Conv2D(filters=f2, kernel_size=kernel_size, strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer='he_uniform')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = LeakyReLU()(X)

    X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer='he_uniform')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = Add()([X, X_res])
    X = LeakyReLU()(X)
    return X


def custom_resnet(model_config, input_shape, metrics, n_classes, mixed_precision=False, output_bias=None):
    '''
    Defines a deep convolutional neural network model with residual connections for multiclass image classification.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param n_classes: # of classes in data
    :param mixed_precision: Whether to use mixed precision (use if you have GPU with compute capacity >= 7.0)
    :param output_bias: bias initializer of output layer
    :return: a Keras Model object with the architecture defined in this method
    '''

    # Set hyperparameters
    nodes_dense0 = model_config['NODES_DENSE0']
    nodes_dense1 = model_config['NODES_DENSE1']
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    l2_lambda = model_config['L2_LAMBDA']
    optimizer = Adam(learning_rate=lr)
    init_filters = model_config['INIT_FILTERS']
    filter_exp_base = model_config['FILTER_EXP_BASE']
    res_blocks = model_config['RES_BLOCKS']
    kernel_size = eval(model_config['KERNEL_SIZE'])
    max_pool_size = eval(model_config['MAXPOOL_SIZE'])
    strides = eval(model_config['STRIDES'])
    print("MODEL CONFIG: ", model_config)
    pad = kernel_size[0] // 2
    if mixed_precision:
        tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    if output_bias is not None:
        output_bias = Constant(output_bias)     # Set initial output bias

    # Input layer
    X_input = Input(input_shape)
    X = X_input
    X = ZeroPadding2D((pad, pad))(X)

    # Initialize the model with a convolutional layer
    X = Conv2D(init_filters, (7,7), strides=strides, name = 'conv0', kernel_initializer='he_uniform')(X)
    X = BatchNormalization(axis = 3, name='bn_conv0')(X)
    X = LeakyReLU()(X)
    X = MaxPool2D(max_pool_size, padding='same', name='maxpool0')(X)

    # Add residual blocks
    for i in range(res_blocks):
        f1 = f2 = init_filters * (filter_exp_base ** i)
        f3 = init_filters * (filter_exp_base ** (i + 2))
        X = convolutional_block(X, kernel_size=kernel_size, filters=[f1, f2, f3], stage=(i+1), block='a', s=strides)
        X = identity_block(X, kernel_size=kernel_size, filters=[f1, f2, f3], stage=(i+1), block='b')
        X = identity_block(X, kernel_size=kernel_size, filters=[f1, f2, f3], stage=(i+1), block='c')

    # Add fully connected layers

    X = AveragePooling2D(strides, name='avgpool0')(X)
    X = Flatten()(X)
    X = Dropout(dropout)(X)
    X = Dense(nodes_dense0, kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda), name='fc0')(X)
    X = LeakyReLU()(X)
    X = Dropout(dropout)(X)
    X = Dense(n_classes, bias_initializer=output_bias)(X)
    Y = Activation('softmax', dtype='float32', name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model
    
def custom_ffcnn(model_config, input_shape, metrics, n_classes, mixed_precision=False, output_bias=None):
    '''
    Defines a feedforward convolutional neural network model with residual connections for multiclass image classification.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param n_classes: # of classes in data
    :param mixed_precision: Whether to use mixed precision (use if you have GPU with compute capacity >= 7.0)
    :param output_bias: bias initializer of output layer
    :return: a Keras Model object with the architecture defined in this method
    '''
    
    # Set hyperparameters
    nodes_dense0 = model_config['NODES_DENSE0']
    nodes_dense1 = model_config['NODES_DENSE1']
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    l2_lambda = model_config['L2_LAMBDA']
    optimizer = Adam(learning_rate=lr)
    init_filters = model_config['INIT_FILTERS']
    filter_exp_base = model_config['FILTER_EXP_BASE']
    n_blocks = model_config['BLOCKS']
    kernel_size = eval(model_config['KERNEL_SIZE'])
    max_pool_size = eval(model_config['MAXPOOL_SIZE'])
    strides = eval(model_config['STRIDES'])
    pad = kernel_size[0] // 2
    print("MODEL CONFIG: ", model_config)
    if mixed_precision:
        tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
    
    if output_bias is not None:
        output_bias = Constant(output_bias)     # Set initial output bias

    # Input layer
    X_input = Input(input_shape)
    X = X_input
    X = ZeroPadding2D((pad, pad))(X)
    
    # Add blocks of convolutions and max pooling
    for i in range(n_blocks):
        filters = init_filters * (2 ** i)
        X = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', name='conv2d_block' + str(i) + '_0',
                   kernel_initializer='he_uniform', activation='relu', activity_regularizer=l2(l2_lambda))(X)
        X = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', name='conv2d_block' + str(i) + '_1',
                   kernel_initializer='he_uniform', activation='relu', activity_regularizer=l2(l2_lambda))(X)
        X = BatchNormalization(axis=3, name='bn_block' + str(i))(X)
        X = MaxPool2D(max_pool_size, padding='same', name='maxpool' + str(i))(X)
    
    # Model head
    X = GlobalAveragePooling2D(name='gloval_avgpool')(X)
    X = Dropout(dropout)(X)
    X = Dense(nodes_dense0, kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda), activation='relu', name='fc0')(X)
    X = Dropout(dropout)(X)
    X = Dense(nodes_dense1, kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda), activation='relu', name='fc1')(X)
    X = Dense(n_classes, bias_initializer=output_bias)(X)
    Y = Activation('softmax', dtype='float32', name='output')(X)
    
    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model

'''
For figuring out VGG16 architecture & how to manipulate freezing desired blocks

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
model_config = cfg['NN']['VGG16']
input_shape= cfg['DATA']['IMG_DIM']
metrics = cfg['TRAIN']['METRIC_PREFERENCE']

vgg16(model_config = model_config, input_shape = [244,244,3], metrics = metrics, n_classes = 3, output_bias=None)
'''