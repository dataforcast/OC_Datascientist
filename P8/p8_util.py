
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

# The random seed to use.
RANDOM_SEED = 42

LOG_DIR = './tmp/models'

_NUM_LAYERS_KEY = "num_layers"
FEATURES_KEY = 'images'

#-------------------------------------------------------------------------------
# Estimator configuration.
#-------------------------------------------------------------------------------
def make_config(model_name, output_dir=None):
    '''Reset output directory.
    Returns a TF configuration object for feeding Esmimator.
    '''
    if output_dir is None : 
        output_dir=LOG_DIR

    outdir = os.path.join(output_dir, model_name)
    shutil.rmtree(outdir, ignore_errors = True)
        
    return tf.estimator.RunConfig(
        save_checkpoints_steps=100000,
        save_summary_steps=100000,
        tf_random_seed=RANDOM_SEED,
        model_dir=outdir)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Estimator configuration.
#-------------------------------------------------------------------------------
class _NNAdaNetBuilder(adanet.subnetwork.Builder):
    """Builds a DNN subnetwork for AdaNet."""

    def __init__(self, nn_type, nb_class,feature_columns, optimizer, layer_size, num_layers,
               learn_mixture_weights, dropout, seed):
        """Initializes a `_DNNBuilder`.

        Args:
          feature_columns: An iterable containing all the feature columns used by
            the model. All items in the set should be instances of classes derived
            from `FeatureColumn`.
          optimizer: An `Optimizer` instance for training both the subnetwork and
            the mixture weights.
          layer_size: The number of nodes to output at each hidden layer.
          num_layers: The number of hidden layers.
          learn_mixture_weights: Whether to solve a learning problem to find the
            best mixture weights, or use their default value according to the
            mixture weight type. When `False`, the subnetworks will return a no_op
            for the mixture weight train op.
          dropout: The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out
            10% of input units.
          seed: A random seed.

        Returns:
          An instance of `_DNNBuilder`.
        """

        self._feature_columns = feature_columns
        self._optimizer = optimizer
        self._layer_size = layer_size
        self._num_layers = num_layers
        self._learn_mixture_weights = learn_mixture_weights
        self._dropout = dropout
        self._seed = seed
        self._nn_type = nn_type
        self._nb_class = nb_class
        self._cnn_rate = 0.5
        self._cnn_seed = 10
        self._cnn_cnnlayer = 0
        self._cnn_denselayer = 0
        if self._nn_type == 'CNN' :
            # Number of CNN layers
            self._cnn_cnnlayer = self._num_layers  

            # Number of dense layers      
            self._cnn_denselayer = 1
            
            # Number of units in CNN dense layer.
            self._cnn_layersize = layer_size
        else : 
            pass
        
        print("\n*** _NNAdaNetBuilder : Classes={}".format(self._nb_class))
        
        

    #----------------------------------------------------------------------------
    #
    #----------------------------------------------------------------------------
    def _build_cnn_subnetwork_keras(self, input_layer, features\
        , logits_dimension, is_training) :
    
        kernel_initializer = tf.keras.initializers.he_normal(seed=self._seed)


        features_key = list(features.keys())[0]
        w = features[features_key].shape[1]
        h = features[features_key].shape[2]
        c = features[features_key].shape[3]
        last_layer = input_layer
        
        print("\n*** _build_cnn_subnetwork_keras() : logits_dimension= {}".format(logits_dimension))
        print("*** _build_cnn_subnetwork_keras() : features= {}".format(list(features.keys())))
        print("*** _build_cnn_subnetwork_keras() : width={} / Heigh={} / Channel={}".format(w, h,c))
        print("*** _build_cnn_subnetwork_keras() : Features keys={}".format(features_key))
        print("*** _build_cnn_subnetwork_keras() : Input shape={}".format(last_layer.shape))
        print("*** _build_cnn_subnetwork_keras() : features shape= {}\n".format(features[features_key].shape))
        if True :
            for layer in range(self._cnn_cnnlayer) : 
                images = list(features.values())[0]
                last_layer = keras.layers.Conv2D(32, (3, 3), input_shape=(w, h, c), padding='same', activation='relu')(images)
                last_layer = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(last_layer)
                last_layer = keras.layers.Dropout(self._cnn_rate, seed = self._cnn_seed)(last_layer)
                last_layer = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(last_layer)  
                                  
                last_layer = keras.layers.Flatten()(last_layer)        
            
                #for layer in range(self._cnn_denselayer) : 
                last_layer = keras.layers.Dense(100, activation='relu')(last_layer)

        # Ajout de la dernière couche fully-connected qui permet de classifier
        logits_ = keras.layers.Dense(units=self._nb_class , activation='softmax')(last_layer)
        #model.add(Dense(layer_logits))
        print("*** _build_cnn_subnetwork_keras() : Layers (CNN, Dense)= ({},{}) Done!".format(self._cnn_cnnlayer, self._cnn_denselayer))

        return last_layer, logits_
   #----------------------------------------------------------------------------


    #----------------------------------------------------------------------------
    #
    #----------------------------------------------------------------------------
    def _build_dnn_subnetwork(self, input_layer, features, logits_dimension, is_training) :
        last_layer = input_layer

        
        for i_ in range(self._num_layers):
            print("\n**** *** _build_dnn_subnetwork : Layer= {} / Layers= {}".format(i_, self._num_layers))
            last_layer = tf.layers.dense(
                last_layer,
                units=self._layer_size,
                activation=tf.nn.relu,
                kernel_initializer=tf.glorot_uniform_initializer(seed=self._seed))
            
            last_layer = tf.layers.dropout(
                last_layer, rate=self._dropout, seed=self._seed, training=is_training)
        
        print("**** *** _build_dnn_subnetwork : Layers= {}\n".format(self._num_layers))
        logits = tf.layers.dense(
            last_layer,
            units=logits_dimension,
            kernel_initializer=tf.glorot_uniform_initializer(seed=self._seed))
        
        
        
        return last_layer, logits
    #---------------------------------------------------------------------------
            
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _build_cnn_subnetwork(self, input_layer, features\
    , logits_dimension, is_training) :
    
        features_key = list(features.keys())[0]
        w = features[features_key].shape[1]
        h = features[features_key].shape[2]
        c = features[features_key].shape[3]
        last_layer = input_layer

        print("*** _build_cnn_subnetwork() : width={} / Heigh={} / Channel={}".format(w, h,c))
        print("*** _build_cnn_subnetwork() : CNN layer size={}".format(self._cnn_layersize))
        if self._cnn_cnnlayer > 0 : 
            last_layer =  features['images']       
            for layer in range(self._cnn_cnnlayer) :     
            
                last_layer = tf.layers.conv2d(last_layer, filters=64,
                                      kernel_size=(3,3) , strides=1,
                                      padding='same', activation=tf.nn.relu)
                                      
                last_layer = tf.layers.conv2d(last_layer, filters=64,
                                      kernel_size=(3,3), strides=1,
                                      padding='same', activation=tf.nn.relu)
                
                pool_size = (2, 2)
                last_layer = tf.layers.max_pooling2d(inputs=last_layer, pool_size= pool_size , strides=2)
                last_layer = tf.layers.dropout(inputs=last_layer, rate=self._dropout)


                                  
            last_layer = tf.layers.conv2d(last_layer, filters=128,
                                  kernel_size=(3,3), strides=1,
                                  padding='same', activation=tf.nn.relu)

            pool_size = (2, 2)
            last_layer = tf.layers.max_pooling2d(inputs=last_layer, pool_size= pool_size , strides=2)
            last_layer = tf.layers.dropout(inputs=last_layer, rate=self._dropout)
        
            last_layer = tf.contrib.layers.flatten(last_layer)
            #print("\n*** *** Last layer shape= {}".format(last_layer))
            last_layer = tf.layers.dense(inputs=last_layer, units=self._cnn_layersize, activation=tf.nn.relu)
            last_layer = tf.layers.dropout(inputs=last_layer, rate=self._dropout, training=is_training)
        
        # Logits Layer
        logits = tf.layers.dense(inputs=last_layer, units=self._nb_class)
        return last_layer,logits
   #----------------------------------------------------------------------------
    
    #----------------------------------------------------------------------------
    #
    #----------------------------------------------------------------------------
    def build_subnetwork(self,
                       features,
                       logits_dimension,
                       training,
                       iteration_step,
                       summary,
                       previous_ensemble=None):
        """See `adanet.subnetwork.Builder`."""
        input_layer \
        = tf.feature_column.input_layer(features=features\
                                    , feature_columns=self._feature_columns)
        
        print("\n\n*** build_subnetwork() : features shape= {}".format(features['images'].shape))
        
        if self._nn_type == 'DNN' :
            last_layer, logits \
            = self._build_dnn_subnetwork(input_layer, features\
            , logits_dimension, training)
            
        elif self._nn_type == 'CNN' :
            last_layer, logits \
            = self._build_cnn_subnetwork(input_layer, features\
            , logits_dimension, training)
        else :
            print("\n*** ERROR : NN type={} no yet supported!".format(self._nn_type))
            return None
        
        # Approximate the Rademacher complexity of this subnetwork as the square-
        # root of its depth.
        complexity = tf.sqrt(tf.to_float(self._num_layers))
        with tf.name_scope(""):
            summary.scalar("complexity", complexity)
            summary.scalar("num_layers", self._num_layers)

        if False :
            persisted_tensors = {_NUM_LAYERS_KEY: tf.constant(self._num_layers)}
        else :
            if self._nn_type == 'CNN' :
                persisted_tensors = {self._nn_type: tf.constant(self._cnn_cnnlayer)}
            else : 
                persisted_tensors = {self._nn_type: tf.constant(self._num_layers)}
            
        return adanet.Subnetwork(
            last_layer=last_layer,
            logits=logits,
            complexity=complexity,
            persisted_tensors=persisted_tensors)

    def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
        """See `adanet.subnetwork.Builder`."""

        # NOTE: The `adanet.Estimator` increments the global step.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return self._optimizer.minimize(loss=loss, var_list=var_list)

    def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
        """See `adanet.subnetwork.Builder`."""

        if not self._learn_mixture_weights:
            return tf.no_op("mixture_weights_train_op")

        # NOTE: The `adanet.Estimator` increments the global step.
        return self._optimizer.minimize(loss=loss, var_list=var_list)
    #----------------------------------------------------------------------------

    @property
    def name(self):
        """See `adanet.subnetwork.Builder`."""
        num_layers = 0
        if self._nn_type == 'CNN' :
            num_layers = self._cnn_cnnlayer
        else : 
            num_layers = self._num_layers

        if False :
            if num_layers == 0:
                # No hidden layers is a linear model.
                return "linear"
            else : 
                return "{}_layer_{}".format(self._nn_type, num_layers)
        else :                 
            return "{}_layer_{}".format(self._nn_type, num_layers)
        #return f'cnn_{self._n_convs}'
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
               nb_nn_candidate = 2):
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
            seed=seed)
  

        print("*** INFO : MyGenerator : instantiation DONE!")

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

            if False :
                list_nn_candidate = [self._nn_builder_fn(num_layers=num_layers+new_layer) \
                                     for new_layer in range(0, self._nb_nn_candidate)]
                return list_nn_candidate
            else :
                # Returns a list of instanciated classes that implement 
                # subnetworks candidates.


                print("\n*** generate_candidates : Iteration= {} /Layers={}"\
                .format(iteration_number, num_layers))
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
def preprocess_image_deprecated( image, label,tuple_dimension= (224,224,3), kind='other', is_standard=False):
    """Preprocesses an image for an `Estimator`."""
    
    list_dimension = [dimension for dimension in tuple_dimension]
    
    print("\n*** preprocess_image() : list_dimension={}".format(list_dimension))
    print("\n*** preprocess_image() : image shape={}".format(image.shape))
    if is_standard is True :
        image = image / 255.
    else :
        pass
    
    # First let's scale the pixel values to be between 0 and 1.
    if kind == 'other' :
        image = tf.reshape(image,list_dimension)
    else :
        image = tf.cast(image, tf.float32)  #
        # Next we reshape the image so that we can apply a 2D convolution to it.
        image = tf.reshape(image, list_dimension)

    
    # Finally the features need to be supplied as a dictionary.
    features = {FEATURES_KEY: image}
    return features, label
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# 
#-------------------------------------------------------------------------------
def input_fn_deprecated(partition, x, y, tuple_dimension=None,training=False, batch_size=1\
, kind='other'):
    
    def input_fn_():
        print("\n*** input_fn() : Y shape={} ".format(y.shape))
        try :
            label_dimension = y.shape[1]
        except IndexError:
            label_dimension=1
        if partition == "train":
            if kind == 'other' :
                dataset = tf.data.Dataset.from_generator(
                generator(x, y), (tf.float32, tf.int32), (tuple_dimension, label_dimension))
            else :
                dataset = tf.data.Dataset.from_generator(
                generator(x, y), (tf.float32, tf.int32), (tuple_dimension, ()))
        elif partition == "predict":
            if kind == 'other' :
                dataset = tf.data.Dataset.from_generator(
                generator(x[:10], y[:10]), (tf.float32, tf.int32)\
                , (tuple_dimension, label_dimension))
            else :
                dataset = tf.data.Dataset.from_generator(
                generator(x[:10], y[:10]), (tf.float32, tf.int32)\
                , (tuple_dimension, ()))
        else:
            if kind == 'other' :
                dataset = tf.data.Dataset.from_generator(
                generator(x, y), (tf.float32, tf.int32), (tuple_dimension, label_dimension))
            else :
                dataset = tf.data.Dataset.from_generator(
                generator(x, y), (tf.float32, tf.int32), (tuple_dimension, ()))

        # We call repeat after shuffling, rather than before, to prevent separate
        # epochs from blending together.
        if training:
            dataset = dataset.shuffle(10 * batch_size, seed=RANDOM_SEED).repeat()

        dataset = dataset.map(preprocess_image).batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        print("\n*** input_fn() : labels shape= {} / Y shape={} / Dimension={}".format(labels.shape, y.shape,tuple_dimension))
        return features, labels
    return input_fn_
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
    print("\n*** generator() : labels shape= {} / label values= {}".format(labels.shape, labels[0]))
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
def input_fn(partition, x, y, training=False, batch_size=None, tuple_dimension=None):
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
    if partition == "train":
        #dataset = tf.data.Dataset.from_generator(
        #    generator(x, y), (tf.float32, tf.int32), ((224, 224, 3), (1,)))
          
      dataset = tf.data.Dataset.from_generator(
          generator(x, y), (tf.float32, tf.int32), (feature_shape, label_shape))          
    else:
        dataset = tf.data.Dataset.from_generator(
            generator(x, y), (tf.float32, tf.int32), (feature_shape, label_shape))          
      #dataset = tf.data.Dataset.from_generator(
      #    generator(x, y), (tf.float32, tf.int32), ((224, 224,3), (1,)))

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    if training:
      dataset = dataset.shuffle(10 * batch_size, seed=RANDOM_SEED).repeat()

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

  
