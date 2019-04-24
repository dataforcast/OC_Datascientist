
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
class MyGenerator(adanet.subnetwork.Generator):
    """Generates a two NN subnetworks at each iteration.

    The first NN has an identical shape to the most recently added subnetwork
    in `previous_ensemble`. The second has the same shape plus one more dense
    layer on top. This is similar to the adaptive network presented in Figure 2 of
    [Cortes et al. ICML 2017](https://arxiv.org/abs/1607.01097), without the
    connections to hidden layers of networks from previous iterations.
    """
    #---------------------------------------------------------------------------
    # 
    #---------------------------------------------------------------------------
    def __init__(self, dict_adanet_config):
        """Initializes a NN `Generator`.

        Args:
          nn_type : type of neural network; may be DNN, CNN, RNN, mixte of them. 
          feature_columns: An iterable containing all the feature columns used by
            DNN models. All items in the set should be instances of classes derived
            from `FeatureColumn`.
          optimizer: An `Optimizer` instance for training both the subnetwork and
            the mixture weights.
          layer_size: Number of nodes in each hidden layer of the subnetwork
            candidates. Note that this parameter is ignored in a DNN with no hidden
            layers.
          initial_num_layers: Minimum number of layers for each DNN subnetwork. At
            iteration 0, the subnetworks will be `initial_num_layers` deep.
            Subnetworks at subsequent iterations will be at least as deep.
          learn_mixture_weights: Whether to solve a learning problem to find the
            best mixture weights, or use their default value according to the
            mixture weight type. When `False`, the subnetworks will return a no_op
            for the mixture weight train op.
          dropout: The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out
            10% of input units.
          seed: A random seed.
          nb_nn_candidate : number of NN candidates generated for each iteration.
        Returns:
          An instance of `Generator`.

        Raises:
          ValueError: If feature_columns is empty.
          ValueError: If layer_size < 1.
          ValueError: If initial_num_layers < 0.
        """
        feature_columns = dict_adanet_config['adanet_feature_columns']
        layer_size = dict_adanet_config['adanet_nn_layer_config']['nn_dense_unit_size']
        #layer_size = dict_adanet_config['adanet_layer_size']
        initial_num_layers = dict_adanet_config['adanet_initial_num_layers']
        dict_nn_layer_config = dict_adanet_config['adanet_nn_layer_config']
        
        if not feature_columns:
          raise ValueError("feature_columns must not be empty")

        if layer_size < 1:
          raise ValueError("layer_size must be >= 1")

        if initial_num_layers < 0:
          raise ValueError("initial_num_layers must be >= 0")
        
        nn_type      = dict_nn_layer_config['nn_type']
        nn_candidate = dict_adanet_config['adanet_nn_candidate']
        
        self._initial_num_layers = initial_num_layers
        self._nn_type = nn_type
        self._nb_nn_candidate = nn_candidate 
        self._dict_adanet_config = dict_adanet_config.copy()
        self._nn_builder_fn = functools.partial(_NNAdaNetBuilder,dict_adanet_config)
  

        print("*** INFO : MyGenerator : instantiation DONE!")
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    # 
    #---------------------------------------------------------------------------
    def generate_candidates(self, previous_ensemble, iteration_number,
                              previous_ensemble_reports, all_reports):
            """See `adanet.subnetwork.Generator`.

            Generates 2 NN, the second one having one layer more then the fisrt one
            and the first NN having the same number of layers then the previous NN.
            """
            #-------------------------------------------------------------------
            # Candidates are instantiated
            # First one has 0 layer.
            # Number of layers is incremented from perevious subnetwork ensemble, 
            # if exists.
            # Oherwise, number of layers is the number of initial layer.
            #-------------------------------------------------------------------
            print("\n*** +++ generate_candidates() : Initial Layer(s)= {}\n".format(self._initial_num_layers))
            print("\n*** +++ generate_candidates() : previous_ensemble= {}\n".format(previous_ensemble))
            if previous_ensemble:
                num_layers = tf.contrib.util.constant_value(
                previous_ensemble.weighted_subnetworks[-1].subnetwork
                .persisted_tensors[self._nn_type])
            else : 
                num_layers = self._initial_num_layers
                self._start_time = time.time()
                print("\n*** +++ generate_candidates() : Layer(s) (1)= {}\n".format(num_layers))
            
            if False :
                list_nn_candidate = [self._nn_builder_fn(num_layers=num_layers+new_layer) \
                                     for new_layer in range(0, self._nb_nn_candidate)]
                return list_nn_candidate
            else :
                print("\n*** +++ generate_candidates() : Layer(s) (2)= {}\n".format(num_layers))
                # Returns a list of instanciated classes that implement 
                # subnetworks candidates.
                # self._cnn_convlayer  = num_layers + 1
                dict_adanet_config = self._dict_adanet_config.copy()
                dict_adanet_config['adanet_num_layers']= num_layers
                dict_adanet_config_p = dict_adanet_config.copy()
                dict_adanet_config_p['adanet_num_layers']= num_layers+1
                return [
                    self._nn_builder_fn(num_layers=num_layers),
                    self._nn_builder_fn(num_layers=num_layers + 1),]
#-------------------------------------------------------------------------------    


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
class MyGenerator_deprecated(adanet.subnetwork.Generator):
    """Generates a two NN subnetworks at each iteration.

    The first NN has an identical shape to the most recently added subnetwork
    in `previous_ensemble`. The second has the same shape plus one more dense
    layer on top. This is similar to the adaptive network presented in Figure 2 of
    [Cortes et al. ICML 2017](https://arxiv.org/abs/1607.01097), without the
    connections to hidden layers of networks from previous iterations.
    """
    #---------------------------------------------------------------------------
    # 
    #---------------------------------------------------------------------------
    def __init__(self,
               nb_class,
               feature_columns,
               optimizer,
               layer_size=32,
               initial_num_layers=0,
               learn_mixture_weights=False,
               dropout=0.,
               seed=None,
               nn_type='DNN',
               nb_nn_candidate = 2,
               cnn_layer_config=None,
               is_cnn_batch_norm=False,
               initializer=None):
        """Initializes a NN `Generator`.

        Args:
          nn_type : type of neural network; may be DNN, CNN, RNN, mixte of them. 
          feature_columns: An iterable containing all the feature columns used by
            DNN models. All items in the set should be instances of classes derived
            from `FeatureColumn`.
          optimizer: An `Optimizer` instance for training both the subnetwork and
            the mixture weights.
          layer_size: Number of nodes in each hidden layer of the subnetwork
            candidates. Note that this parameter is ignored in a DNN with no hidden
            layers.
          initial_num_layers: Minimum number of layers for each DNN subnetwork. At
            iteration 0, the subnetworks will be `initial_num_layers` deep.
            Subnetworks at subsequent iterations will be at least as deep.
          learn_mixture_weights: Whether to solve a learning problem to find the
            best mixture weights, or use their default value according to the
            mixture weight type. When `False`, the subnetworks will return a no_op
            for the mixture weight train op.
          dropout: The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out
            10% of input units.
          seed: A random seed.
          nb_nn_candidate : number of NN candidates generated for each iteration.
        Returns:
          An instance of `Generator`.

        Raises:
          ValueError: If feature_columns is empty.
          ValueError: If layer_size < 1.
          ValueError: If initial_num_layers < 0.
        """

        if not feature_columns:
          raise ValueError("feature_columns must not be empty")

        if layer_size < 1:
          raise ValueError("layer_size must be >= 1")

        if initial_num_layers < 0:
          raise ValueError("initial_num_layers must be >= 0")

        self._initial_num_layers = initial_num_layers
        self._nn_type = nn_type
        self._nb_nn_candidate = nb_nn_candidate 
        self._nn_builder_fn = functools.partial(
            _NNAdaNetBuilder,
            nn_type = self._nn_type,
            nb_class = nb_class,
            feature_columns=feature_columns,
            optimizer=optimizer,
            layer_size=layer_size,
            learn_mixture_weights=learn_mixture_weights,
            dropout=dropout,
            seed=seed,
            cnn_layer_config=cnn_layer_config,
            is_cnn_batch_norm = is_cnn_batch_norm,
            initializer = initializer)
  

        print("*** INFO : MyGenerator : instantiation DONE!")
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    # 
    #---------------------------------------------------------------------------
    def generate_candidates(self, previous_ensemble, iteration_number,
                              previous_ensemble_reports, all_reports):
            """See `adanet.subnetwork.Generator`.

            Generates 2 NN, the second one having one layer more then the fisrt one
            and the first NN having the same number of layers then the previous NN.
            """
            #-------------------------------------------------------------------
            # Candidates are instantiated
            # First one has 0 layer.
            # Number of layers is incremented from perevious subnetwork ensemble, 
            # if exists.
            # Oherwise, number of layers is the number of initial layer.
            #-------------------------------------------------------------------
            if previous_ensemble:
                num_layers = tf.contrib.util.constant_value(
                previous_ensemble.weighted_subnetworks[-1].subnetwork
                .persisted_tensors[self._nn_type])
            else : 
                num_layers = self._initial_num_layers
                self._start_time = time.time()
            if previous_ensemble:
                #print("\n*** +++ generate_candidates() : Layer= {}\n"\
                #.format(previous_ensemble.weighted_subnetworks[-1].subnetwork.persisted_tensors[self._nn_type]))
                pass
            else :
                #print("\n*** +++ generate_candidates() : Layer= {}\n".format(num_layers))
                pass
            if False :
                list_nn_candidate = [self._nn_builder_fn(num_layers=num_layers+new_layer) \
                                     for new_layer in range(0, self._nb_nn_candidate)]
                return list_nn_candidate
            else :
                # Returns a list of instanciated classes that implement 
                # subnetworks candidates.
                # self._cnn_convlayer  = num_layers + 1
                return [
                    self._nn_builder_fn(num_layers=num_layers),
                    self._nn_builder_fn(num_layers=num_layers + 1),]
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

  

