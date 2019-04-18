
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

# The random seed to use.
RANDOM_SEED = 42

LOG_DIR = './tmp/models'

_NUM_LAYERS_KEY = "num_layers"
FEATURES_KEY = 'images'

#-------------------------------------------------------------------------------
#   
#-------------------------------------------------------------------------------
def my_model_fn( features, labels, mode, params ): 
    '''This function implements training, evalaution and prediction.
    It also implements the predictor model.
    It is designed in the context of a customized Estimator.

    This function is invoked form Estimator's train, predict and evaluate methods.
        features : batch of features provided from input function.
        labels : batch labels provided from input function.
        mode : provided by input function.
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
# Estimator configuration.
#-------------------------------------------------------------------------------
class _NNAdaNetBuilder(adanet.subnetwork.Builder):
    """Builds a DNN subnetwork for AdaNet."""

    def __init__(self, dict_adanet_config, num_layers):
        """Initializes a `_DNNBuilder`.

        Args:
          dict_adanet_config : dictionary for configuration of AdaNet algo.
          num_layers: The number of hidden layers.


        Returns:
          An instance of `_DNNBuilder`.
        """
                
        self._feature_columns = dict_adanet_config['adanet_feature_columns']
        self._learn_mixture_weights = dict_adanet_config['adanet_is_learn_mixture_weights']

        #---------------------------------------------------
        # Hyper parameters for NN Builder
        #---------------------------------------------------
        dict_nn_layer_config= dict_adanet_config['adanet_nn_layer_config']
        
        self._nn_type    = dict_nn_layer_config['nn_type']
        self._optimizer  = dict_nn_layer_config['nn_optimizer']
        self._layer_size = dict_nn_layer_config['nn_dense_unit_size']
        #self._num_layers = dict_nn_layer_config['nn_dense_layer_num']
        self._num_layers = num_layers
        
        self._dropout    = dict_nn_layer_config['nn_dropout_rate']
        self._seed       = dict_nn_layer_config['nn_seed']
        self._nb_class   = dict_nn_layer_config['nn_logit_dimension']
        
        # When value is None, then HE normal initializer is used as default. 
        self._layer_initializer_name = dict_nn_layer_config['nn_initializer_name']
        # Batch normaization activation
        self._is_nn_batch_norm = dict_nn_layer_config['nn_batch_norm']
        
        #---------------------------------------------------
        # Hyper parameters for CNN network
        #---------------------------------------------------
        if self._nn_type == 'CNN' or self._nn_type == 'CNNBase':
            dict_cnn_layer_config = dict_nn_layer_config['nn_layer_config']
            self._cnn_seed = self._seed

            self._conv_kernel_size = dict_cnn_layer_config['conv_kernel_size']
            
            # Number of CNN CONCOLUTIONAL layers  
            #self._cnn_convlayer = dict_cnn_layer_config['conv_layer_num']
            self._cnn_convlayer =num_layers

            # Number of dense layers      
            self._cnn_denselayer =  dict_nn_layer_config['nn_dense_layer_num']
            
            # Number of units in CNN dense layer.
            self._cnn_layersize = self._layer_size
            
            # Fixed CNN layers configuration 
            self._dict_cnn_layer_config=dict_cnn_layer_config
            
            # Batch normaization activation
            self._is_cnn_batch_norm = self._is_nn_batch_norm
        else : 
            pass

        self._start_time = 0
        
        print("\n*** _NNAdaNetBuilder : NN Type={}".format(self._nn_type))
        

        
    #---------------------------------------------------------------------------
    #   Properties
    #---------------------------------------------------------------------------
    def _get_feature_columns(self) :
       return self._feature_columns
    def _set_feature_columns(self, feature_columns) :
        print("\n*** ERROR : assignement is forbidden for this parameter!")

    def _get_nb_class(self) :
       return self._nb_class
    def _set_nb_class(self, nb_class) :
        print("\n*** ERROR : assignement is forbidden for this parameter!")

    def _get_optimizer(self) :
       return self._optimizer
    def _set_optimizer(self, optimizer) :
        print("\n*** ERROR : assignement is forbidden for this parameter!")

    feature_columns = property(_get_feature_columns,_set_feature_columns)
    nb_class = property(_get_nb_class,_set_nb_class)
    optimizer = property(_get_optimizer, _set_optimizer)


    #----------------------------------------------------------------------------
    #
    #----------------------------------------------------------------------------
    def _build_dnn_subnetwork(self, input_layer, features, logits_dimension, is_training) :
        last_layer = input_layer

        
        for i_ in range(self._num_layers):
            #print("\n**** *** _build_dnn_subnetwork : Layer= {} / Layers= {}".format(i_, self._num_layers))
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
    def _cnn_bacth_norm(self, layer, is_training) :
        if self._is_cnn_batch_norm is True :                      
            layer = tf.layers.batch_normalization(layer, training=is_training)
        return layer
    #---------------------------------------------------------------------------
        
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _get_layer_initializer(self) :
        if self._layer_initializer_name == 'xavier' :
            layer_initializer = tf.contrib.layers.xavier_initializer
        else : 
            layer_initializer = tf.keras.initializers.he_normal
        return layer_initializer
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _get_conv_activation_fn(self, conv_activation_name) :
        conv_activation_fn = None
        if conv_activation_name == 'relu' :
            conv_activation_fn =tf.nn.relu
        else : 
            print("\n*** ERROR : activation function name= {} Unknown!".format(conv_activation_name))
        return conv_activation_fn
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _build_cnn_baseline_subnetwork(self, input_layer, features
    , logits_dimension, is_training) :
    
        features_key = list(features.keys())[0]
        w = features[features_key].shape[1]
        h = features[features_key].shape[2]
        c = features[features_key].shape[3]
        last_layer = input_layer

        tuple_conv_kernel_size = self._dict_cnn_layer_config['conv_kernel_size']
        conv_filters           = self._dict_cnn_layer_config['conv_filters']
        conv_strides           = self._dict_cnn_layer_config['conv_strides']
        conv_padding_name      = self._dict_cnn_layer_config['conv_padding_name']
        conv_activation_name   = self._dict_cnn_layer_config['conv_activation_name']
        
        layer_initializer = self._get_layer_initializer()
        conv_activation_fn = self._get_conv_activation_fn(conv_activation_name)
        
        #-----------------------------------------------------------------------                
        # Convolutional Layers
        #-----------------------------------------------------------------------                
        if self._cnn_convlayer > 0 : 
            last_layer =  features['images']       
            for layer in range(self._cnn_convlayer) :     
                last_layer = self._cnn_bacth_norm(last_layer, is_training)
                last_layer = tf.layers.conv2d(last_layer
                                        , filters=conv_filters
                                        , kernel_size=tuple_conv_kernel_size
                                        , strides=conv_strides
                                        , padding=conv_padding_name
                                        , activation=conv_activation_fn
                                        , kernel_initializer=layer_initializer())
                pool_size = (2, 2)
                last_layer = tf.layers.max_pooling2d(inputs=last_layer, pool_size= pool_size , strides=2)


        #-----------------------------------------------------------------------                
        # Dense Layer(s)
        #-----------------------------------------------------------------------
        last_layer = tf.contrib.layers.flatten(last_layer)      
        
                  
        for layer in range(self._cnn_denselayer) :     
            last_layer = self._cnn_bacth_norm(last_layer, is_training)
            last_layer = tf.layers.dense( inputs=last_layer
                                        , units=self._cnn_layersize 
                                        , activation=conv_activation_fn
                                        , kernel_initializer=layer_initializer())
            
            last_layer = tf.layers.dropout(   inputs=last_layer
                                            , rate=self._dropout
                                            , training=is_training)

        #-----------------------------------------------------------------------                
        # Logits Layer
        #-----------------------------------------------------------------------                
        logits = tf.layers.dense( inputs=last_layer
                                , units=self._nb_class
                                , kernel_initializer=layer_initializer())

        return last_layer,logits
    #---------------------------------------------------------------------------
        

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _build_cnn_subnetwork(self, input_layer, features\
    , logits_dimension, is_training) :
    
        if self._nn_type == 'CNNBase' :
            return self._build_cnn_baseline_subnetwork(input_layer, features\
                                            , logits_dimension, is_training)
        else :
            pass
            
        features_key = list(features.keys())[0]
        w = features[features_key].shape[1]
        h = features[features_key].shape[2]
        c = features[features_key].shape[3]
        last_layer = input_layer
        
        layer_initializer = self._get_layer_initializer()
        
        #print("\n*** _build_cnn_subnetwork() : width={} / Heigh={} / Channel={}".format(w, h,c))
        #print("*** _build_cnn_subnetwork() : CNN layer size={} / CNN layer= {}\n".format(self._cnn_layersize, self._cnn_convlayer))
        if self._cnn_convlayer > 0 : 
            last_layer =  features['images']       
            for layer in range(self._cnn_convlayer) :     
                last_layer = self._cnn_bacth_norm(last_layer, is_training)
                last_layer = tf.layers.conv2d(last_layer, filters=64,
                                      kernel_size=(3,3) , strides=1,
                                      padding='same', activation=tf.nn.relu,
                                      kernel_initializer=layer_initializer())
                last_layer = self._cnn_bacth_norm(last_layer, is_training)
                last_layer = tf.layers.conv2d(last_layer, filters=64,
                                      kernel_size=(3,3), strides=1,
                                      padding='same', activation=tf.nn.relu,
                                      kernel_initializer=layer_initializer())
                pool_size = (2, 2)
                last_layer = tf.layers.max_pooling2d(inputs=last_layer, pool_size= pool_size , strides=2)
                
            last_layer = tf.contrib.layers.flatten(last_layer)
            last_layer = self._cnn_bacth_norm(last_layer, is_training)
            
            last_layer = tf.layers.dense(inputs=last_layer
            , units=self._cnn_layersize
            , activation=tf.nn.relu, kernel_initializer=layer_initializer())
            last_layer = tf.layers.dropout(inputs=last_layer
            , rate=self._dropout, training=is_training)



            # Process fixed CNN layers configuration 
            if self._cnn_layer_config is not None :
                list_cnn_layer_filter = list(self._cnn_layer_config.values())[0]
                for cnn_layer_filter in list_cnn_layer_filter :
                
                    
                    last_layer = self._cnn_bacth_norm(last_layer, is_training)
                    last_layer = tf.layers.conv2d(last_layer, filters=cnn_layer_filter,
                                          kernel_size=(3,3), strides=1,
                                          padding='same', activation=tf.nn.relu,
                                          kernel_initializer=layer_initializer())
                    pool_size = (2, 2)
                    last_layer = tf.layers.max_pooling2d(inputs=last_layer, pool_size= pool_size , strides=2)

        
            last_layer = tf.contrib.layers.flatten(last_layer)

            for _ in range(self._cnn_denselayer) :
                last_layer = self._cnn_bacth_norm(last_layer, is_training)
                
                last_layer = tf.layers.dense(inputs=last_layer
                , units=self._cnn_layersize
                , activation=tf.nn.relu, kernel_initializer=layer_initializer())
                
                last_layer = tf.layers.dropout(inputs=last_layer
                , rate=self._dropout, training=is_training)
                     
            
            
            if False :
                #print("\n*** *** Last layer shape= {}".format(last_layer))
                last_layer = self._cnn_bacth_norm(last_layer, is_training)
                last_layer = tf.layers.dense(inputs=last_layer, units=self._cnn_layersize
                , activation=tf.nn.relu, kernel_initializer=layer_initializer())

                last_layer = tf.layers.dropout(inputs=last_layer
                , rate=self._dropout
                , training=is_training)
        
        # Logits Layer
        logits = tf.layers.dense(inputs=last_layer, units=self._nb_class
        ,kernel_initializer=layer_initializer())
        
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
        
        #print("\n\n*** build_subnetwork() : features shape= {}".format(features['images'].shape))
        print("\n\n*** build_subnetwork() : NN type= {}".format(self._nn_type))
        
        if self._nn_type == 'DNN' :
            last_layer, logits \
            = self._build_dnn_subnetwork(input_layer, features\
            , logits_dimension, training)
            complexity = tf.sqrt(tf.to_float(self._num_layers))
            
        elif self._nn_type == 'CNN' :
            last_layer, logits \
            = self._build_cnn_subnetwork(input_layer, features\
            , logits_dimension, training)
            complexity = tf.sqrt(tf.to_float(self._cnn_convlayer))

        elif self._nn_type == 'CNNBase' :
            last_layer, logits \
            = self._build_cnn_baseline_subnetwork(input_layer, features\
            , logits_dimension, training)
            complexity = tf.sqrt(tf.to_float(self._cnn_convlayer))

            
        else :
            print("\n*** ERROR : NN type={} no yet supported!".format(self._nn_type))
            return None
        
        # Approximate the Rademacher complexity of this subnetwork as the square-
        # root of its depth.
        with tf.name_scope(""):
            summary.scalar("complexity", complexity)
            summary.scalar("num_layers", self._num_layers)
            summary.scalar("cnn_num_layers", self._cnn_convlayer)

        if False :
            persisted_tensors = {_NUM_LAYERS_KEY: tf.constant(self._num_layers)}
        else :
            if self._nn_type == 'CNN' :
                persisted_tensors = {self._nn_type: tf.constant(self._cnn_convlayer)}
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
        if self._nn_type == 'CNN' or self._nn_type == 'CNNBase':
            num_layers = self._cnn_convlayer
        else : 
            num_layers = self._num_layers

        if True :
            if num_layers == 0:
                # No hidden layers is a linear model.
                return "{}_linear".format(self._nn_type)
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

  

