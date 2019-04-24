
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
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


import p5_util

import NNAdaNetBuilder

# The random seed to use.
RANDOM_SEED = 42

LOG_DIR = './tmp/models'

_NUM_LAYERS_KEY = "num_layers"
FEATURES_KEY = 'images'

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
    feature_columns = net_builder.feature_columns
    logits_dimension = net_builder.nb_class
    input_layer = tf.feature_column.input_layer(features=features, feature_columns=feature_columns)
    is_training = False

    if mode == tf.estimator.ModeKeys.TRAIN :
        is_training = True
        
    _, logits = net_builder._build_cnn_subnetwork(input_layer, features\
                                                           , logits_dimension, is_training)
    predicted_classes = tf.argmax(logits, 1)

    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes
    , name='accuracy')
    #print("\n*** INFO : accuracy= {}".format(accuracy))
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN :
        optimizer = net_builder.optimizer
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        tf.summary.scalar('train_accuracy', accuracy[1])
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    elif mode ==  tf.estimator.ModeKeys.EVAL :
        # Compute accuracy from tf metrics package. It compares thruth values (labels) against
        # predicted one (predicted_classes)
        metrics = {'eval_accuracy': accuracy}
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
def load_dataset(filename) :
    '''Load dataset from file name given as function parameter.
    '''
    (x_train,x_test, y_train, y_test) = p5_util.object_load(filename)
    if True :
        y_train=array_label_encode_from_index(y_train)
        y_test=array_label_encode_from_index(y_test)


    w_size = x_train.shape[1]
    h_size = x_train.shape[2]        

    y_train.shape, y_train.min(), y_train.max()
    nClasses = max(len(np.unique(y_train)), len(np.unique(y_test)))
    tuple_dimension = (x_train.shape,x_test.shape,y_train.shape,y_test.shape)
    #print("Dimensions= {}".format(tuple_dimension))
    #print("Number of classes= "+str(nClasses))
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
    
    if is_restored is False :
        shutil.rmtree(outdir, ignore_errors = True)
    else :
        pass    
    return tf.estimator.RunConfig(
        save_checkpoints_steps=5,
        save_summary_steps=5,
        tf_random_seed=RANDOM_SEED,
        model_dir=outdir)
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
def preprocess_image(image, label):
  """Preprocesses an image for an `Estimator`."""
  # First let's scale the pixel values to be between 0 and 1.

  #image = image / 255.
  # Next we reshape the image so that we can apply a 2D convolution to it.
  image = tf.reshape(image, [224, 224, 3])
  # Finally the features need to be supplied as a dictionary.
  features = {FEATURES_KEY: image}
  return features, label
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# 
#-------------------------------------------------------------------------------
def generator(images, labels):
  """Returns a generator that returns image-label pairs."""

  def _gen():
    '''yield key word will return a generator.
    Such object is an iterator that iterates over elements once only.
    '''
    #labels = tf.reshape(labels, [-1,1,1,1], name=None)
    #labels = labels.reshape([-1,1,1,1])
    #print("\n*** generator() : labels shape= {} / label values= {}".format(labels.shape, labels[0]))
    for image, label in zip(images, labels):
      #yield image, label

      #-------------------------------------------------------------------------
      # NB : label shape has to be compliant with shape defined in  _input_fn()
      # Otherwise, an error will occure when checking shapes issued from iterator 
      # from Dataset package because of shapes.
      #-------------------------------------------------------------------------
      yield image, np.array(label).reshape(1)
  return _gen
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def input_fn(partition, x, y, num_epochs, batch_size=None, tuple_dimension=None):
  """Generate an input_fn for the Estimator."""

  def _input_fn():
    #---------------------------------------------------------------------------
    # Defining shapes with None as first value allows the generator to 
    # adapt itself when batch does not fit expected size.
    # Otherwise an error value may be raized such as 
    # ValueError: `generator` yielded an element of shape () where an element of shape (1,) was expected.
    #---------------------------------------------------------------------------
    feature_shape = [224,224,3]
    label_shape = [1]
    training=False
    if partition == "train":
        training = True
        #dataset = tf.data.Dataset.from_generator(
        #    generator(x, y), (tf.float32, tf.int32), ((224, 224, 3), (1,)))
          
        dataset = tf.data.Dataset.from_generator(
            generator(x, y), (tf.float32, tf.int32), (feature_shape, label_shape))          
        dataset = dataset.shuffle(10 * batch_size, seed=RANDOM_SEED).repeat(num_epochs)
    else:
        dataset = tf.data.Dataset.from_generator(
            generator(x, y), (tf.float32, tf.int32), (feature_shape, label_shape))          
      #dataset = tf.data.Dataset.from_generator(
      #    generator(x, y), (tf.float32, tf.int32), ((224, 224,3), (1,)))

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


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_tf_head(feature_key, tuple_dimension, nClasses) :
  #FEATURES_KEY = feature_key
  list_dimension = [dimension for dimension in tuple_dimension]
  w_size = tuple_dimension[0]
  h_size = tuple_dimension[1]
  if(len(tuple_dimension) > 2) :
    channel = tuple_dimension[2]
  else :
    channel = 1

  #hidden_units=[w_size*h_size*channel, 512, nClasses]
  my_feature_columns = [tf.feature_column.numeric_column(FEATURES_KEY\
                                                    , shape=[w_size, h_size, channel])]

  # Some `Estimators` use feature columns for understanding their input features.
  # We will average the losses in each mini-batch when computing gradients.
  loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE

  # A `Head` instance defines the loss function and metrics for `Estimators`.
  # Tells Tensorfow how to compute loss function and metrics
  tf_head = tf.contrib.estimator.multi_class_head(nClasses\
                                                  , loss_reduction=loss_reduction)
  return my_feature_columns, loss_reduction, tf_head    
#-------------------------------------------------------------------------------

  

