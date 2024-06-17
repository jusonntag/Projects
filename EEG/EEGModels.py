"""
 ARL_EEGModels - A collection of Convolutional Neural Network models for EEG
 Signal Processing and Classification, using Keras and Tensorflow

 Requirements:
    (1) tensorflow == 2.X (as of this writing, 2.0 - 2.3 have been verified
        as working)
 
 To run the EEG/MEG ERP classification sample script, you will also need

    (4) mne >= 0.17.1
    (5) PyRiemann >= 0.2.5
    (6) scikit-learn >= 0.20.1
    (7) matplotlib >= 2.2.3
    
 To use:
    
    (1) Place this file in the PYTHONPATH variable in your IDE (i.e.: Spyder)
    (2) Import the model as
        
        from EEGModels import EEGNet    
        
        model = EEGNet(nb_classes = ..., Chans = ..., Samples = ...)
        
    (3) Then compile and fit the model
    
        model.compile(loss = ..., optimizer = ..., metrics = ...)
        fitted    = model.fit(...)
        predicted = model.predict(...)

 Portions of this project are works of the United States Government and are not
 subject to domestic copyright protection under 17 USC Sec. 105.  Those 
 portions are released world-wide under the terms of the Creative Commons Zero 
 1.0 (CC0) license.  
 
 Other portions of this project are subject to domestic copyright protection 
 under 17 USC Sec. 105.  Those portions are licensed under the Apache 2.0 
 license.  The complete text of the license governing this material is in 
 the file labeled LICENSE.TXT that is a part of this project's official 
 distribution. 
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv3D, MaxPooling3D, AveragePooling3D, Add, GlobalAveragePooling2D, Reshape, Multiply, Lambda, Concatenate, GlobalMaxPooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K


def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:
        
        1. Depthwise Convolutions to learn spatial filters within a 
        temporal convolution. The use of the depth_multiplier option maps 
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn 
        spatial filters within each filter in a filter-bank. This also limits 
        the number of free parameters to fit when compared to a fully-connected
        convolution. 
        
        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions. 
        
    
    While the original paper used Dropout, we found that SpatialDropout2D 
    sometimes produced slightly better results for classification of ERP 
    signals. However, SpatialDropout2D significantly reduced performance 
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.
        
    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the 
    kernel lengths for double the sampling rate, etc). Note that we haven't 
    tested the model performance with this rule so this may not work well. 
    
    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.

    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D. 

    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1) # 
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense) # change to softmax for more than 2 classes
    
    return Model(inputs=input1, outputs=softmax)




def EEGNet_SSVEP(nb_classes = 12, Chans = 8, Samples = 256, 
             dropoutRate = 0.5, kernLength = 256, F1 = 96, 
             D = 1, F2 = 96, dropoutType = 'Dropout'):
    """ SSVEP Variant of EEGNet, as used in [1]. 

    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. 
      D               : number of spatial filters to learn within each temporal
                        convolution.
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
      
      
    [1]. Waytowich, N. et. al. (2018). Compact Convolutional Neural Networks
    for Classification of Asynchronous Steady-State Visual Evoked Potentials.
    Journal of Neural Engineering vol. 15(6). 
    http://iopscience.iop.org/article/10.1088/1741-2552/aae5d8

    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense')(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)



def EEGNet_old(nb_classes, Chans = 64, Samples = 128, regRate = 0.0001,
           dropoutRate = 0.25, kernels = [(2, 32), (8, 4)], strides = (2, 4)):
    """ Keras Implementation of EEGNet_v1 (https://arxiv.org/abs/1611.08024v2)

    This model is the original EEGNet model proposed on arxiv
            https://arxiv.org/abs/1611.08024v2
    
    with a few modifications: we use striding instead of max-pooling as this 
    helped slightly in classification performance while also providing a 
    computational speed-up. 
    
    Note that we no longer recommend the use of this architecture, as the new
    version of EEGNet performs much better overall and has nicer properties.
    
    Inputs:
        
        nb_classes     : total number of final categories
        Chans, Samples : number of EEG channels and samples, respectively
        regRate        : regularization rate for L1 and L2 regularizations
        dropoutRate    : dropout fraction
        kernels        : the 2nd and 3rd layer kernel dimensions (default is 
                         the [2, 32] x [8, 4] configuration)
        strides        : the stride size (note that this replaces the max-pool
                         used in the original paper)
    
    """

    # start the model
    input_main   = Input((Chans, Samples))
    layer1       = Conv2D(16, (Chans, 1), input_shape=(Chans, Samples, 1),
                                 kernel_regularizer = l1_l2(l1=regRate, l2=regRate))(input_main)
    layer1       = BatchNormalization()(layer1)
    layer1       = Activation('elu')(layer1)
    layer1       = Dropout(dropoutRate)(layer1)
    
    permute_dims = 2, 1, 3
    permute1     = Permute(permute_dims)(layer1)
    
    layer2       = Conv2D(4, kernels[0], padding = 'same', 
                            kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
                            strides = strides)(permute1)
    layer2       = BatchNormalization()(layer2)
    layer2       = Activation('elu')(layer2)
    layer2       = Dropout(dropoutRate)(layer2)
    
    layer3       = Conv2D(4, kernels[1], padding = 'same',
                            kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
                            strides = strides)(layer2)
    layer3       = BatchNormalization()(layer3)
    layer3       = Activation('elu')(layer3)
    layer3       = Dropout(dropoutRate)(layer3)
    
    flatten      = Flatten(name = 'flatten')(layer3)
    
    dense        = Dense(nb_classes, name = 'dense')(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)



def DeepConvNet(nb_classes, Chans = 64, Samples = 256,
                dropoutRate = 0.5):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.
    
    This implementation assumes the input is a 2-second EEG signal sampled at 
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference. 
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10
    
    Note that this implementation has not been verified by the original 
    authors. 
    
    """

    # start the model
    input_main   = Input((Chans, Samples, 1))
    block1       = Conv2D(25, (1, 5), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(25, (Chans, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1       = Activation('elu')(block1)
    block1       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1       = Dropout(dropoutRate)(block1)
  
    block2       = Conv2D(50, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block2       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block2)
    block2       = Activation('elu')(block2)
    block2       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2       = Dropout(dropoutRate)(block2)
    
    block3       = Conv2D(100, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    block3       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block3)
    block3       = Activation('elu')(block3)
    block3       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3       = Dropout(dropoutRate)(block3)
    
    block4       = Conv2D(200, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block3)
    block4       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block4)
    block4       = Activation('elu')(block4)
    block4       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4       = Dropout(dropoutRate)(block4)
    
    flatten      = Flatten()(block4)
    
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense) # softmax
    
    return Model(inputs=input_main, outputs=softmax)


# need these for ShallowConvNet
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))   


def ShallowConvNet(nb_classes, Chans = 64, Samples = 128, dropoutRate = 0.5):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.
    
    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in 
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is 
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25    
    
    Note that this implementation has not been verified by the original 
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations. 
    """

    # start the model
    input_main   = Input((Chans, Samples, 1))
    block1       = Conv2D(40, (1, 13), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(40, (Chans, 1), use_bias=False, 
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1       = Activation(log)(block1)
    block1       = Dropout(dropoutRate)(block1)
    flatten      = Flatten()(block1)
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)


#def CNN_LSTM(nb_classes, Chans 64, Samples = 128, kernel = 64):

    input        = Input(shape = (Chans, Samples, 1))

    block1       = SeparableConv2D(32, (1, kernel), input_shape = (Chans, Samples, 1), padding = 'same', use_bias = False)(input)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = MaxPooling2D((1, 8))(block1)
    block1       = Dropout(0.5)(block1)


class Conv2dWithConstraint(layers.Conv2D):
    def __init__(self, filters, kernel_size, do_weight_norm=True, max_norm=1, **kwargs):
        super(Conv2dWithConstraint, self).__init__(
            filters, kernel_size, **kwargs)
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm

    def call(self, inputs):
        if self.do_weight_norm:
            self.kernel = tf.clip_by_norm(
                self.kernel, self.max_norm, axes=[0])
        return super(Conv2dWithConstraint, self).call(inputs)


class LinearWithConstraint(layers.Dense):
    def __init__(self, units, do_weight_norm=True, max_norm=1, **kwargs):
        super(LinearWithConstraint, self).__init__(units, **kwargs)
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm

    def call(self, inputs):
        if self.do_weight_norm:
            self.kernel = tf.clip_by_norm(
                self.kernel, self.max_norm, axes=[0])
        return super(LinearWithConstraint, self).call(inputs)

class VarLayer(tf.keras.layers.Layer):
    '''
    The variance layer: calculates the variance of the data along given 'axis'
    '''
    def __init__(self, axis, **kwargs):
        super(VarLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return tf.math.reduce_variance(x, axis=self.axis, keepdims=True)

class StdLayer(tf.keras.layers.Layer):
    '''
    The standard deviation layer: calculates the std of the data along given 'axis'
    '''
    def __init__(self, axis, **kwargs):
        super(StdLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return tf.math.reduce_std(x, axis=self.axis, keepdims=True)

class LogVarLayer(tf.keras.layers.Layer):
    '''
    The log variance layer: calculates the log variance of the data along given 'axis'
    (natural logarithm)
    '''
    def __init__(self, axis, **kwargs):
        super(LogVarLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return tf.math.log(tf.clip_by_value(tf.math.reduce_variance(x, axis=self.axis, keepdims=True), 1e-6, 1e6))

class MeanLayer(tf.keras.layers.Layer):
    '''
    The mean layer: calculates the mean of the data along given 'axis'
    '''
    def __init__(self, axis, **kwargs):
        super(MeanLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return tf.math.reduce_mean(x, axis=self.axis, keepdims=True)

class MaxLayer(tf.keras.layers.Layer):
    '''
    The max layer: calculates the max of the data along given 'axis'
    '''
    def __init__(self, axis, **kwargs):
        super(MaxLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return tf.math.reduce_max(x, axis=self.axis, keepdims=True)

class Swish(tf.keras.layers.Layer):
    '''
    The swish layer: implements the swish activation function
    '''
    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)

    def call(self, x):
        return x * tf.sigmoid(x)


class FBCNet(models.Model):
    '''
    DOES NOT WORK
        FBNet with seperate variance for every 1s. 
        The data input is in a form of batch x 1 x chan x time x filterBand
    '''
    def __init__(self, n_chan, n_time, n_class=2, n_bands=9, m=32,
                 temporal_layer='LogVarLayer', stride_factor=4, do_weight_norm=True, **kwargs):
        super(FBCNet, self).__init__(**kwargs)

        self.n_bands = n_bands
        self.m = m
        self.stride_factor = stride_factor

        # create all the parallel SCB
        self.scb = self.SCB(m, n_chan, self.n_bands, do_weight_norm=do_weight_norm)

        # Formulate the temporal aggregator
        self.temporal_layer = globals()[temporal_layer](axis=3)

        # The final fully connected layer
        self.last_layer = self.LastBlock(
            self.m*self.n_bands*self.stride_factor, n_class, do_weight_norm=do_weight_norm)

    def SCB(self, m, n_chan, n_bands, do_weight_norm=True):
        return models.Sequential([
            Conv2dWithConstraint(m*n_bands, (n_chan, 1), groups=n_bands,
                                 max_norm=2, do_weight_norm=do_weight_norm),
            layers.BatchNormalization(),
            layers.Activation('swish')
        ])

    def LastBlock(self, in_f, out_f, do_weight_norm=True):
        return models.Sequential([
            LinearWithConstraint(out_f, max_norm=0.5,
                                 do_weight_norm=do_weight_norm),
            layers.Softmax(axis=1)
        ])

    def call(self, inputs):
        x = tf.squeeze(tf.transpose(inputs, perm=(0, 4, 2, 3, 1)), axis=4)
        x = self.scb(x)
        x = tf.reshape(x, [*x.shape[0:2], self.stride_factor, int(x.shape[3]/self.stride_factor)])
        x = self.temporal_layer(x)
        x = tf.reshape(x, (x.shape[0], -1))
        x = self.last_layer(x)
        return x

    """def NFEEG(nb_classes, Chans = 64, Samples = 250):
        input = Input(shape=(Samples, 1, Chans))
    
        # Block 1
        block1 = BatchNormalization()(input)
        block1 = Conv2D(96, (30, 1), padding='same', kernel_regularizer=l2())(block1)
        block1 = BatchNormalization()(block1)
        block1 = Conv2D(96, (30, 1), padding='same', kernel_regularizer=l2())(block1)
        block1 = BatchNormalization()(block1)
        block1 = DepthwiseConv2D((30, 1), depth_multiplier=1, depthwise_constraint=max_norm(1.))(block1)
        block1 = Activation('elu')(block1)
        block1 = AveragePooling2D((2, 1))(block1)
        block1 = Dropout(0.5)(block1)
    
        # Block 2
        block2 = Conv2D(64, (15, 1), padding='same', kernel_regularizer=l2())(block1)
        block2 = BatchNormalization()(block2)
        block2 = Conv2D(64, (15, 1), padding='same', kernel_regularizer=l2())(block2)
        block2 = BatchNormalization()(block2)
        block2 = SeparableConv2D(64, (15, 1), padding='same', pointwise_constraint=max_norm(1.))(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((2, 1))(block2)
        block2 = Dropout(0.5)(block2)
    
        # Block 3
        block3 = Conv2D(64, (7, 1), padding='same', kernel_regularizer=l2())(block2)
        block3 = BatchNormalization()(block3)
        block3 = Conv2D(64, (7, 1), padding='same', kernel_regularizer=l2())(block3)
        block3 = BatchNormalization()(block3)
        block3 = DepthwiseConv2D((7, 1), depth_multiplier=2, depthwise_constraint=max_norm(1.))(block3)
        block3 = Activation('elu')(block3)
        block3 = AveragePooling2D((2, 1))(block3)
        block3 = Dropout(0.5)(block3)
    
        # Block 4
        block4 = Conv2D(32, (3, 1), padding='same', kernel_regularizer=l2())(block3)
        block4 = BatchNormalization()(block4)
        block4 = Conv2D(32, (3, 1), padding='same', kernel_regularizer=l2())(block4)
        block4 = BatchNormalization()(block4)
        block4 = SeparableConv2D(32, (3, 1), padding='same', pointwise_constraint=max_norm(1.))(block4)
        block4 = Activation('elu')(block4)
        block4 = AveragePooling2D((2, 1))(block4)
        block4 = Dropout(0.5)(block4)
    
        # Flatten and Dense layer
        flatten = Flatten(name='flatten')(block4)
        dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(0.5))(flatten)
        softmax = Activation('softmax', name='softmax')(dense)
    
        # Create model
        return Model(inputs=input, outputs=softmax)"""

def NFEEG(nb_classes, Chans = 64, Samples = 250):
    input = Input(shape=(Samples, 1, Chans))

    # Block 1
    block1 = BatchNormalization()(input)
    block1 = Conv2D(96, (30, 1), kernel_regularizer=l2())(block1)
    block1 = BatchNormalization()(block1)
    block1 = Conv2D(96, (30, 1), kernel_regularizer=l2())(block1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((30, 1), depth_multiplier=1, depthwise_constraint=max_norm(1.))(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((2, 1))(block1)
    block1 = Dropout(0.5)(block1)

    # Block 2
    block2 = Conv2D(64, (15, 1), kernel_regularizer=l2())(block1)
    block2 = BatchNormalization()(block2)
    block2 = Conv2D(64, (15, 1), kernel_regularizer=l2())(block2)
    block2 = BatchNormalization()(block2)
    block2 = SeparableConv2D(64, (15, 1), pointwise_constraint=max_norm(1.))(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((2, 1))(block2)
    block2 = Dropout(0.5)(block2)

    # Block 3
    block3 = Conv2D(64, (7, 1), kernel_regularizer=l2())(block2)
    block3 = BatchNormalization()(block3)
    block3 = Conv2D(64, (7, 1), kernel_regularizer=l2())(block3)
    block3 = BatchNormalization()(block3)
    block3 = DepthwiseConv2D((7, 1), depth_multiplier=2, depthwise_constraint=max_norm(1.))(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((2, 1))(block3)
    block3 = Dropout(0.5)(block3)

    # Block 4
    block4 = Conv2D(32, (3, 1), kernel_regularizer=l2())(block3)
    block4 = BatchNormalization()(block4)
    block4 = Conv2D(32, (3, 1), kernel_regularizer=l2())(block4)
    block4 = BatchNormalization()(block4)
    block4 = SeparableConv2D(32, (3, 1), pointwise_constraint=max_norm(1.))(block4)
    block4 = Activation('elu')(block4)
    block4 = AveragePooling2D((2, 1))(block4)
    block4 = Dropout(0.5)(block4)

    # Flatten and Dense layer
    flatten = Flatten(name='flatten')(block4)
    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    # Create model
    return Model(inputs=input, outputs=softmax)

def TSPFE_layer(X, H, W):
    # Initialize weights
    M = tf.Variable(tf.random.normal(shape=(H, H)), trainable=True, name='M')
    wf = tf.Variable(tf.random.normal(shape=(W, W)), trainable=True, name='wf')
    bf = tf.Variable(tf.zeros(shape=(W,)), trainable=True, name='bf')
    relu = tf.keras.layers.ReLU()

    # Calculate Q = X^T * M * X
    Q = tf.matmul(tf.matmul(tf.transpose(X, perm=[0, 2, 1]), M), X)
    
    # Diagonalization of M: M = P^(-1) * D * P
    eigvals, P = tf.linalg.eigh(M)
    D = tf.linalg.diag(eigvals)
    
    # Orthogonal projection
    M_diag = tf.matmul(tf.matmul(P, D), tf.transpose(P))
    Q = tf.matmul(tf.matmul(tf.transpose(X, perm=[0, 2, 1]), M_diag), X)
    
    # Normalize Q
    Qc = tf.nn.softmax(Q, axis=-1)
    Qr = tf.nn.softmax(tf.transpose(Q, perm=[0, 2, 1]), axis=-1)
    
    # Feature extraction
    Fc = tf.matmul(X, Qc)
    Fr = tf.matmul(X, Qr)
    
    # Gating mechanism
    Fc = Fc * relu(tf.matmul(Fc, wf) + bf)
    Fr = Fr * relu(tf.matmul(Fr, wf) + bf)
    
    # Combine features
    FTSP = tf.concat([Fc, Fr], axis=-1)
    
    return FTSP

def TSPNet(nb_classes, Chans = 64, Samples = 250):
    input = Input(shape = (Chans, Samples, 1))

    block1 = Conv2D(32, (1,7), (1,2), padding = 'same')(input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    block1_1 = Conv2D(64, (1,1), (1,2), padding = 'same')(block1)
    block1_1 = BatchNormalization()(block1_1)
    block1_1 = Activation('relu')(block1_1)
    block1_1 = MaxPooling2D((1,2), (1,2))(block1_1)

    block1_2 = Conv2D(64, (1,3), (1,2), padding = 'same')(block1)
    block1_2 = BatchNormalization()(block1_2)
    block1_2 = Activation('relu')(block1_2)
    block1_2 = MaxPooling2D((1,2), (1,2))(block1_2)

    block2 = Add()([block1_1, block1_2])
    
    block2_1 = Conv2D(128, (1,1), (1,2), padding = 'same')(block2)
    block2_1 = BatchNormalization()(block2_1)
    block2_1 = Activation('relu')(block2_1)
    block2_1 = MaxPooling2D((1,2), (1,2))(block2_1)

    block2_2 = Conv2D(128, (1,3), (1,2), padding = 'same')(block2)
    block2_2 = BatchNormalization()(block2_2)
    block2_2 = Activation('relu')(block2_2)
    block2_2 = MaxPooling2D((1,2), (1,2))(block2_2)

    block3 = Add()([block2_1, block2_2])

    block3_1 = Conv2D(256, (1,1), (1,2), padding = 'same')(block3)
    block3_1 = BatchNormalization()(block3_1)
    block3_1 = Activation('relu')(block3_1)
    block3_1 = MaxPooling2D((1,2), (1,2))(block3_1)

    block3_2 = Conv2D(256, (1,3), (1,2), padding = 'same')(block3)
    block3_2 = BatchNormalization()(block3_2)
    block3_2 = Activation('relu')(block3_2)
    block3_2 = MaxPooling2D((1,2), (1,2))(block3_2)

    block4 = Add()([block3_1, block3_2])

    block4_1 = Conv2D(512, (1,1), (1,1), padding = 'same')(block4)
    block4_1 = BatchNormalization()(block4_1)
    block4_1 = Activation('relu')(block4_1)
    block4_1 = MaxPooling2D((2,1), (2,1))(block4_1)

    block4_2 = Conv2D(512, (3,1), (1,1), padding = 'same')(block4)
    block4_2 = BatchNormalization()(block4_2)
    block4_2 = Activation('relu')(block4_2)
    block4_2 = MaxPooling2D((2,1), (2,1))(block4_2)

    block4_3 = Conv2D(512, (5,1), (1,1), padding = 'same')(block4)
    block4_3 = BatchNormalization()(block4_3)
    block4_3 = Activation('relu')(block4_3)
    block4_3 = MaxPooling2D((2,1), (2,1))(block4_3)

    block5 = Add()([block4_1, block4_2, block4_3])

    block5_shape = tf.shape(block5)
    block5_reshaped = tf.reshape(block5, (block5_shape[0], block5_shape[1], -1))

    tspfe_output = TSPFE_layer(block5_reshaped, block5_shape[1], block5_shape[2])

    gap_output = GlobalAveragePooling2D()(tspfe_output)

    output = Dense(nb_classes, activation='softmax')(gap_output)

    return Model(inputs=input, outputs=output)

def channel_attention(input_tensor, ratio=8):
    channel = input_tensor.shape[-1]
    #b, _, _, channel = input_tensor.shape
    shared_dense_one = Dense(channel // ratio, activation='relu',kernel_initializer='he_normal', use_bias=False, bias_initializer='zeros')
    shared_dense_two = Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=False, bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_tensor)
    #avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    #max_pool = tf.reduce_max(input_tensor, axis=[1, 2], keepdims=True)
    max_pool = GlobalMaxPooling2D()(input_tensor)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return Multiply()([input_tensor, cbam_feature])

def spatial_attention(input_tensor):
    #x1 = tf.reduce_mean(input_tensor, axis = -1)
    #x1 = tf.expand_dims(x1, axis = -1)

    #x2 = tf.reduce_max(input_tensor, axis= -1)
    #x2 = tf.expand_dims(x2, axis = -1)

    #concat = Concatenate()([x1,x2])
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_tensor)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_tensor)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid')(concat)#, kernel_initializer='he_normal', use_bias=False)(concat)

    return Multiply()([input_tensor, cbam_feature])

def cbam_block(input_tensor, ratio=8):
    attention_feature = channel_attention(input_tensor, ratio)
    attention_feature = spatial_attention(attention_feature)
    return attention_feature

def MSTCN(nb_classes, Chans = 64, Samples = 250):
    input = Input(shape = (1, Samples, Chans))

    mastcn1 = Conv2D(8, (1,3), padding = 'same')(input)
    mastcn1 = BatchNormalization()(mastcn1)

    mastcn2 = Conv2D(8, (1,5), padding = 'same')(input)
    mastcn2 = BatchNormalization()(mastcn2)

    mastcn3 = Conv2D(8, (1,7), padding = 'same')(input)
    mastcn3 = BatchNormalization()(mastcn3)

    mastcn = Add()([mastcn1, mastcn2, mastcn3])

    spat_conv = Conv2D(48, (Samples,1), padding = 'same')(mastcn)
    spat_conv = BatchNormalization()(spat_conv)
    spat_conv = Activation('elu')(spat_conv)
    spat_conv = AveragePooling2D((1,64))(spat_conv)
    spat_conv = Dropout(0.5)(spat_conv)

    cbam_output = cbam_block(spat_conv)

    flat = Flatten()(cbam_output)

    output = Dense(nb_classes, activation='softmax')(flat)

    return Model(inputs=input, outputs=output)

def se_block(input, ratio=8):
    # attention block
    b, f, c, s = input.shape # b = batch, f = filterbank, c = channels, s = number of timepoints 
    pool = GlobalAveragePooling2D()(input)
    reshape = Reshape(f)(pool)
    dense = Dense(c // ratio, activation = 'relu')(reshape)
    dense = Dense(c, activation = 'sigmoid')(dense)
    weights = Reshape(f, 1, 1)(dense)
    out = Multiply()([input, weights])
    return out

def FMI_EEGNet(nb_classes, input_tensor, Chans = 64, Samples = 250, dropoutRate = 0.5, norm_rate = 0.25, F1 = 8, F2 = 16):
    kernLength = Samples//2
    dropoutType = Dropout
    D = 2
    #input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input_tensor)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1) # 
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    #softmax      = Activation('softmax', name = 'softmax')(dense) # change to softmax for more than 2 classes
    
    return dense

def FMI_EEG(nb_classes, Chans = 64, Samples = 250, filterbanks = 5):
    ''''
    Fine motor imager EEG model
    uses multiple input branches and EEGNet for one branch
    each branch uses one filterBank -- default 5 filterBanks
    '''
    input1 = Input(shape=(filterbanks, Chans, Samples))

    se = se_block(input1)

    #filterbanks_array = [se[:, i, :, :] for i in range(filterbanks)]
    filterbanks_array = [Lambda(lambda x: x[:, i, :, :, :])(se) for i in range(filterbanks)]
    #fb1_eegnet = FMI_EEGNet(Chans = 64, Samples = Samples, input_tensor=filterbanks_array[0])
    #fb2_eegnet = FMI_EEGNet(Chans = 64, Samples = Samples, input_tensor=filterbanks_array[1])
    #fb3_eegnet = FMI_EEGNet(Chans = 64, Samples = Samples, input_tensor=filterbanks_array[2])
    #fb4_eegnet = FMI_EEGNet(Chans = 64, Samples = Samples, input_tensor=filterbanks_array[3])
    #fb5_eegnet = FMI_EEGNet(Chans = 64, Samples = Samples, input_tensor=filterbanks_array[4])

    fb_eegnets = [FMI_EEGNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples, input_tensor=filterbanks_array[i]) for i in range(filterbanks)]
    
    #concat = Concatenate()([fb1_eegnet, fb2_eegnet, fb3_eegnet, fb4_eegnet, fb5_eegnet])
    concat = Concatenate()(fb_eegnets)
    flat = Flatten()(concat)

    dense = Dense(nb_classes)(flat)

    softmax = Activation('softmax', name = 'softmax')(dense)

    return Model(inputs = input1, outputs = softmax)