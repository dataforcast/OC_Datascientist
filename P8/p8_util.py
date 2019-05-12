
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import random
import numpy as np
import functools
import os
import shutil

import matplotlib.pyplot as plt

import tensorflow as tf
import adanet
from adanet.examples import simple_dnn

from six.moves import range
import keras

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout

from sklearn import preprocessing


import p5_util
import p8_util_config


import NNAdaNetBuilder
#import p8_util_config

RANDOM_SEED = 42
LOG_DIR = './tmp/models'

_NUM_LAYERS_KEY = "num_layers"
FEATURES_KEY = 'images'
    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def create_nn_builder(param_feature_shape, output_dir, layer_num=None):
    '''Creates an NNAdaNetBuilder object from configuration defined into file 
    p8_util_config.
    This allows to show 
    Input :
        * param_feature_shape : shape of the features. This parameter is unknowned 
        from configuration prior to dataset read.
        * output_dir : Directory where logs are dumped.
        * layer_num : number of layers used to build (deep) neural network. When 
        None value, then value from confugration file is used to create 
        NNAdaNetBuilder object. 
        Otherwise, dictionaries from configuration file are updated with this value.
        
    Output :
        * Object of type NNAdaNetBuilder
    
    '''
    if layer_num is None :
        if p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_type'] == 'CNN' :
            layer_num = p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_layer_config']['conv_layer_num']
        elif p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_type'] == 'RNN' :
            layer_num = p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_layer_config']['rnn_layer_num']
        elif p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_type'] == 'DNN' :
            layer_num = p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_layer_config']['dnn_layer_num']
    else :
        pass

    print("\n Number of convolutional layers= {}".format(layer_num))
    oNNAdaNetBuilder = NNAdaNetBuilder.NNAdaNetBuilder(p8_util_config.dict_adanet_config, num_layers=layer_num)
    oNNAdaNetBuilder.feature_shape = param_feature_shape
    oNNAdaNetBuilder.output_dir = output_dir

    #---------------------------------------------------------------------------
    # Dictionaries from configuration file are updated with this number of layers.
    #---------------------------------------------------------------------------
    if p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_type'] == 'CNN' :
        p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_layer_config']['conv_layer_num'] = layer_num
    elif p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_type'] == 'RNN' :
        p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_layer_config']['rnn_layer_num']=layer_num
    elif p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_type'] == 'DNN' :
        p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_layer_config']['dnn_layer_num']=layer_num

    print("\nMax steps= {} / Number of EPOCH={}".format(p8_util_config.MAX_STEPS,p8_util_config.NUM_EPOCHS))
    return oNNAdaNetBuilder
        
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_tf_head(tuple_dimension, nClasses, nn_type='CNN', feature_shape=None) :
    #FEATURES_KEY = feature_key
    list_dimension = [dimension for dimension in tuple_dimension]
    
    if feature_shape is None :
        if(len(tuple_dimension) > 2) :
            channel = tuple_dimension[2]
        else :
            channel = 1
        w_size = tuple_dimension[0]
        h_size = tuple_dimension[1]
        feature_shape = [w_size, h_size, channel]
    else :
        pass
    print("\n*** get_tf_head() : feature shape= {}".format(feature_shape))
    my_feature_columns = [tf.feature_column.numeric_column(FEATURES_KEY\
                                                , shape=feature_shape)]
    print("\n*** get_tf_head() : feature columns= {}".format(my_feature_columns))
    # Some `Estimators` use feature columns for understanding their input features.
    # We will average the losses in each mini-batch when computing gradients.
    if nn_type == 'RNN' :
        loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE
    else :
        loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE
    
    # A `Head` instance defines the loss function and metrics for `Estimators`.
    # Tells Tensorfow how to compute loss function and metrics
    tf_head = tf.contrib.estimator.multi_class_head(nClasses\
                                                  , loss_reduction=loss_reduction)
    return my_feature_columns, loss_reduction, tf_head    
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#   
#-------------------------------------------------------------------------------
def my_model_fn( features, labels, mode, params ): 
    '''This function implements training, evaluation and prediction.
    It also implements the predictor model.
    It is designed in the context of a customized Estimator.

    This function is invoked form Estimator's train, predict and evaluate methods.
        features : batch of features provided from input function.
        labels : batch labels provided from input function.
        mode : provided by input function, mode discriminate train, evaluation and prediction steps.
        params : parameters used in this function, passed to Estimator by higher level call.

    '''
    #-----------------------------------------------------------------------------
    # Get from parameters object that is used form Adanet to build NN sub-networks.
    #-----------------------------------------------------------------------------
    net_builder = params['net_builder']
    nn_type = net_builder._nn_type
    
    feature_shape = net_builder.feature_shape
    logits_dimension = net_builder.nb_class
    
    input_layer = tf.feature_column.input_layer(features=features\
    , feature_columns=net_builder.feature_columns)
    is_training = False

    with tf.name_scope(nn_type):
        print("\n*** my_model_fn() : input_layer shape= {} / Labels shape= {}"\
        .format(input_layer.shape, labels.shape))

        if mode == tf.estimator.ModeKeys.TRAIN :
            is_training = True

        #-----------------------------------------------------------------------
        # Predictions are computed for a batch
        #-----------------------------------------------------------------------
        if nn_type == 'CNN' or  nn_type == 'CNNBase' :
            _, logits = net_builder._build_cnn_subnetwork(input_layer, features\
                                                        , logits_dimension\
                                                        , is_training)
        elif nn_type == 'RNN' : 
            rnn_cell_type = net_builder._dict_rnn_layer_config['rnn_cell_type']
            _, logits = net_builder._build_rnn_subnetwork(input_layer, features\
                                                        , logits_dimension\
                                                        , is_training\
                                                        , rnn_cell_type = rnn_cell_type)
        elif nn_type == 'DNN' :
            _, logits = net_builder._build_dnn_subnetwork(input_layer, features\
                                                        , logits_dimension\
                                                        , is_training)
        else : 
            print("\n*** ERROR : Network type= {} NOT YET SUPPORTED!".format(nn_type))
            return None
        # Returns the index from logits for which logits has the maximum value.
        
        print("\n*** my_model_fn() : logits shape= {} / labels shape= {}"\
        .format(logits.shape, labels.shape))

        #-----------------------------------------------------------------------
        # Accuracy is computed
        #-----------------------------------------------------------------------
        is_accuracy_with_tf = True
        
        if False :
            if is_accuracy_with_tf :
                accuracy_op = tf.metrics.accuracy(labels=tf.argmax(labels,1)\
                                            , predictions=tf.argmax(logits,1)\
                                            , name='accuracy')
            else :
                # predicted_classes is a vector of indexes containing largest value in 
                # output_logitstimesteps, num_inputtimesteps, num_inputtimesteps, 
                # num_inputtimesteps, num_input
                predicted_classes = tf.argmax(logits, 1)
                
                print("\n*** my_model_fn() : predicted_classes= {}".format(predicted_classes))
                print("\n*** my_model_fn() : predicted_classes shape= {}".format(predicted_classes.shape))

                # y_true is a vector of indexes containing largest value in y
                y_true = tf.argmax(labels, 1)

                # A new node is inserted into graph : correct_prediction
                # Following will result as an array of boolean values; True if indexes matches, False otherwise.
                correct_prediction = tf.equal(predicted_classes, y_true, name='correct_pred')

                accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='train_accuracy')                                       


        #-----------------------------------------------------------------------
        # Loss is computed
        #-----------------------------------------------------------------------
        if p8_util_config.IS_LABEL_ENCODED is True:
            labels = tf.one_hot(labels, depth=3)
            
        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
            tf.summary.scalar('Loss', loss)

        if mode == tf.estimator.ModeKeys.TRAIN :
            with tf.name_scope('Train'):
                #---------------------------------------------------------------
                # Gradient descent is computed 
                #---------------------------------------------------------------
                optimizer = net_builder.optimizer
                train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
                tf_label_arg_max = tf.argmax(labels,1)
                accuracy, accuracy_op = tf.metrics.accuracy(labels=tf_label_arg_max\
                            , predictions=tf.argmax(logits,1)\
                            , name=nn_type+'Train_accuracy')
                            
                tf.summary.scalar(nn_type+'Train_Accuracy', accuracy)
                #print("\n*** my_model_fn() : Accuracy= {}".format(accuracy_op, accuracy))
                if False :
                    #tf.summary.scalar('train_accuracy', accuracy[1])
                    if is_accuracy_with_tf :
                        accuracy = train_op[1]
                    else : 
                        accuracy = train_op
                    tf.summary.scalar('Accuracy', accuracy)
                
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        elif mode ==  tf.estimator.ModeKeys.EVAL :
            with tf.name_scope('Eval'):
                # Compute accuracy from tf metrics package. It compares true values (labels) against
                # predicted one (predicted_classes)
                accuracy, accuracy_op = tf.metrics.accuracy(labels=tf.argmax(labels,1)\
                            , predictions=tf.argmax(logits,1)\
                            , name=nn_type+'Eval_accuracy')
                tf.summary.scalar(nn_type+'_Eval_accuracy', accuracy)
                if False :
                    if is_accuracy_with_tf :
                        accuracy = accuracy_op[1]
                    else :  
                        tf.summary.scalar(nn_type+'_Eval_accuracy', accuracy)

                    predicted_classes = tf.argmax(logits, 1)
                    

                    # y_true is a vector of indexes containing largest value in y
                    y_true = tf.argmax(labels, 1)

                    
                    
                    if True :
                        acc, acc_op = tf.metrics.accuracy(y_true, predicted_classes)
                    else :

                        # A new node is inserted into graph : correct_prediction
                        # Following will result as an array of boolean values; True if indexes matches, False otherwise.
                        correct_prediction = tf.equal(predicted_classes, y_true, name='correct_pred')

                        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='eval_accuracy')                                       
                metrics = {nn_type+'_Eval_accuracy': (accuracy, accuracy_op)}

                
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
            
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': predicted_classes[:, tf.newaxis],
                'probabilities': tf.nn.softmax(logits),
                'logits': logits,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        else :
            #print("\n*** ERROR : my_model_fn() : mode= {} is unknwoned!".format(mode))
            pass
    return None
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#   
#-------------------------------------------------------------------------------
def load_data_mnist():
    '''Defaut path to where MNIST dataset is located (after download) :
    ~/.keras/datasets/mnist.npz
    '''
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #print(y_train.shape, y_train.min(), y_train.max())
    #---------------------------------------------------------------------------
    # Make data separation between valid and train data.
    #---------------------------------------------------------------------------
    x_valid_num = x_test.shape[0]//2
    part = x_train.shape[0]-x_valid_num

    x_valid = x_train[part:,:]
    y_valid = y_train[part:]
    x_train = x_train[:part,:]
    y_train = y_train[:part]
    
    return x_train, x_valid, x_test, y_train, y_valid, y_test
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def batch_coloredimage_serial_reshape(x_batch):
    '''Reshape a batch of images allowing an image to be processed as serial data 
    for feefing RNN network.
    
    Each image with initial shape = [w,h,c]
    into a batch of images, each image with retruned shape=[h*c,w]

    This will allow for any image to be serialized as  h*c raws of w columns.
    
    Each raw sized as h*c will feed a deployed cell from RNN network while each 
    cell from RNN will be deployed as w parts.


             +--------------+        +---+----------+
             | B plan       |        |   |          |
          +--------------+  |        | R |          |
          | G plan       |  |        |   |          |
       +--------------+  |  |  ==>   |___|          |
       | R plan       |  |--+        |   |          |
       |              |  |           | G |          |
       |              |--+           |   |          |
       |              |              |___|          |
       +--------------+              |   |          |
                                     | B |          |
                                     |   |          |
                                     +---+----------+
                                       |          |
                                       |          |
                                       |          +---> CELL #W
                                       |
                                       +----> CELL #1                                                                              
    '''
    batch_size = x_batch.shape[0]
    
    w = x_batch.shape[1]
    h = x_batch.shape[2]
    c = x_batch.shape[3]

    #-----------------------------------------------------------------------
    # [batch_size, w, h*c]  --> [batch_size, w, h, c] 
    #-----------------------------------------------------------------------
    x_batch = x_batch.reshape(batch_size,w,-1)

    #-----------------------------------------------------------------------
    # [batch_size, w, h*c]  --> [batch_size, h*c, w ]
    #-----------------------------------------------------------------------
    x_batch = np.transpose(x_batch, [0,2,1])

    return x_batch
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#    
#-------------------------------------------------------------------------------
def load_dataset(filename, dataset_type='P7', is_label_encoded=False) :
    '''Load dataset from file name given as function parameter.
    '''
    is_label_encoded = p8_util_config.IS_LABEL_ENCODED
    if dataset_type == 'P7' :
        (x_train,x_test, y_train, y_test) = p5_util.object_load(filename)
        
        number = x_train.shape[0]
        if p8_util_config.NN_TYPE == 'RNN' :
            x_train = batch_coloredimage_serial_reshape(x_train)
            x_test  = batch_coloredimage_serial_reshape(x_test)
        else :
            pass
        
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        
        if is_label_encoded :
            #print("\n*** y_train= {} / y_test= {}".format(y_train.shape, y_test.shape))
            y_train=array_label_encode_from_index(y_train)
            y_test=array_label_encode_from_index(y_test)
            #print("\n*** y_train= {}".format(y_train.shape))
            nClasses = max(len(np.unique(y_train)), len(np.unique(y_test)))
        else :
            nClasses = y_train.shape[1]
    elif dataset_type == 'MNIST' :
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_data_mnist()
        nClasses = max(len(np.unique(y_train)), len(np.unique(y_test)))
        if True:
            y_train=array_label_encode_binary(y_train)
            y_test=array_label_encode_binary(y_test)
            y_valid=array_label_encode_binary(y_valid)
            nClasses = y_train.shape[1]
    else :
        pass
        
    w_size = x_train.shape[1]
    h_size = x_train.shape[2]        

    tuple_dimension = (x_train.shape,x_test.shape,y_train.shape,y_test.shape)
    
    #print("Dimensions= {}".format(tuple_dimension))
    #print("Number of classes= "+str(nClasses))
    if dataset_type == 'MNIST' :
        return x_train, x_valid, x_test, y_train, y_valid, y_test,  nClasses \
        ,tuple_dimension[0][1:]        

    return x_train, x_test, y_train, y_test, nClasses,tuple_dimension[0][1:]
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Estimator configuration.
#-------------------------------------------------------------------------------
def make_config(model_name, output_dir=None, is_restored=False):
    '''Reset output directory.
    Returns a TF configuration object for feeding Esmimator.
    '''
    if output_dir is None : 
        output_dir=LOG_DIR

    outdir = os.path.join(output_dir, model_name)
    
    print("\n*** make_config() : output dir= {}".format(outdir))
    
    if is_restored is False :
        shutil.rmtree(outdir, ignore_errors = True)
    else :
        pass    
    return tf.estimator.RunConfig(
        save_checkpoints_steps=10,
        save_summary_steps=10,
        tf_random_seed=RANDOM_SEED,
        model_dir=outdir),outdir
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
# 
#-------------------------------------------------------------------------------
def array_label_encode_from_index(y):
    '''Label encoder from index array. 
    '''
    array_label_encode = np.array([np.where(y[i]==1)[0][0] for i in range(0,len(y),1)])
    return array_label_encode
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# 
#-------------------------------------------------------------------------------
def array_label_encode_binary(y):
    '''Label encoder from value into binary values.
    '''

    lb = preprocessing.LabelBinarizer()
    y_enc = lb.fit_transform(y)
    return y_enc
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# 
#-------------------------------------------------------------------------------
def preprocess_color_image(image, label):
  """Preprocesses an image for an `Estimator`."""
  print("\n*** preprocess_color_image() : label shape= {}".format(label.shape))
  features = {FEATURES_KEY: image}
  return features, label

def preprocess_image(image, label):
  """Preprocesses an image for an `Estimator`."""
  # First let's scale the pixel values to be between 0 and 1.

  #image = image / 255.
  # Next we reshape the image so that we can apply a 2D convolution to it.
  print("\n*** preprocess_image() : image shape= {}".format(image.shape))
  #image = tf.reshape(image, [224, 224, 3])
  # Finally the features need to be supplied as a dictionary.
  features = {FEATURES_KEY: image}
  print("\n*** preprocess_image() : features= {} / label shape= {}".format(features, label.shape))
  return features, label
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# 
#-------------------------------------------------------------------------------
def generator(images, labels):
  """Returns a generator that returns image-label pairs."""

  def _gen():
    for image, label in zip(images, labels):
      yield image, label
    print("\n*** generator() : labels shape= {} / label values= {}".format(labels.shape, labels[10]))

  return _gen
  
def generator_deprecated(images, labels):
  """Returns a generator that returns image-label pairs."""

  def _gen():
    '''yield key word will return a generator.
    Such object is an iterator that iterates over elements once only.
    '''
    is_label_encoded = p8_util_config.IS_LABEL_ENCODED
    
    
    #random_index = random.randint(0,len(labels))
    #print("\n*** generator() : labels shape= {} / label values= {}".format(labels.shape, labels[random_index]))

    for image, label in zip(images, labels):
      #yield image, label

      #-------------------------------------------------------------------------
      # NB : label shape has to be compliant with shape defined in  _input_fn()
      # Otherwise, an error will occure when checking shapes issued from iterator 
      # from Dataset package because of shapes.
      #-------------------------------------------------------------------------
      if is_label_encoded :
          yield image, np.array(label).reshape(1)
      else :
          yield image, label
  return _gen
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def input_fn(partition, x, y, input_fn_param):
  """Generate an input_fn for the Estimator."""

  def _input_fn():
    
    num_epochs = input_fn_param['num_epochs']
    batch_size = input_fn_param['batch_size']
    feature_shape = input_fn_param['feature_shape']
    
    if 1 == len(y.shape):
        label_shape = [1]
    else : 
        label_shape = [y.shape[1]]
    
    

    #---------------------------------------------------------------------------
    # Defining shapes with None as first value allows the generator to 
    # adapt itself when batch does not fit expected size.
    # Otherwise an error value may be raized such as 
    # ValueError: `generator` yielded an element of shape () where an element of shape (1,) was expected.
    #---------------------------------------------------------------------------

    print("\n*** input_fn() : feature_shape= {} / label_shape= {}"\
    .format(feature_shape, label_shape))
    
    training=False
    
    if False :
        dataset = tf.data.Dataset.from_generator(
            generator(x, y), (tf.float32, tf.int32), (feature_shape, label_shape))
    else :        
        #dataset = tf.data.Dataset.from_generator(generator(x, y), (tf.float32, tf.int32), ((224, 224, 3), ()))
        dataset = tf.data.Dataset.from_generator(generator(x, y), (tf.float32, tf.int32), (feature_shape, ()))
        
        
    if partition == "train":
        training = True
        dataset = dataset.shuffle(10 * batch_size, seed=RANDOM_SEED).repeat(num_epochs)
    else:
        print("\n*** input_fn : TEST / feature_shape= {}".format(feature_shape))

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    if False :
        dataset = dataset.map(preprocess_image).batch(batch_size)
    else :
        dataset = dataset.map(preprocess_color_image).batch(batch_size)
    
    iterator = dataset.make_one_shot_iterator()

    print("\n***_input_fn() : Label shape ={}".format(label_shape))

    features, labels = iterator.get_next()
    
    return features, labels

  return _input_fn    
#-------------------------------------------------------------------------------


def input_fn_deprecated(partition, x, y, num_epochs, batch_size=None, feature_shape = [224,224,3]):
  """Generate an input_fn for the Estimator."""

  def _input_fn():
    #---------------------------------------------------------------------------
    # Defining shapes with None as first value allows the generator to 
    # adapt itself when batch does not fit expected size.
    # Otherwise an error value may be raized such as 
    # ValueError: `generator` yielded an element of shape () where an element of shape (1,) was expected.
    #---------------------------------------------------------------------------
    label_shape=[1]
    if label_shape is None :
        label_shape=[3]

    print("\n*** input_fn() : feature_shape= {} / Label shape= {}"\
    .format(feature_shape, label_shape))
    
    training=False

    if partition == "train":
        training = True
        dataset = tf.data.Dataset.from_generator(
            generator(x, y), (tf.float32, tf.int32), (feature_shape, label_shape))
        
        dataset = dataset.shuffle(10 * batch_size, seed=RANDOM_SEED).repeat(num_epochs)
    else:
        print("feature_shape= {}".format(feature_shape))
        dataset = tf.data.Dataset.from_generator(
            generator(x, y), (tf.float32, tf.int32), (feature_shape, label_shape))          

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    if training:
        #-----------------------------------------------------------------------
        # Each EPOCH is shuffled. Then shuffle applies before EPOCH
        #-----------------------------------------------------------------------
        dataset = dataset.shuffle(10 * batch_size, seed=RANDOM_SEED).repeat(num_epochs)

    dataset = dataset.map(preprocess_image).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()
    
    #print("\n***_input_fn() : Label={}".format(labels))
    return features, labels

  return _input_fn    
#-------------------------------------------------------------------------------



  

