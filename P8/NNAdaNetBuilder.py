import tensorflow as tf
import adanet

#-------------------------------------------------------------------------------
# Estimator configuration.
#-------------------------------------------------------------------------------
class NNAdaNetBuilder(adanet.subnetwork.Builder) :
    '''Builds a NN subnetwork for AdaNet.'''

    def __init__(self, dict_adanet_config, num_layers):
        """Initializes a `_DNNBuilder`.

        Args:
          dict_adanet_config : dictionary for configuration of AdaNet algo.
          num_layers: The number of hidden layers.


        Returns:
          An instance of `NNAdaNetBuilder`.
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
        self._num_layers = num_layers
        
        self._dropout    = dict_nn_layer_config['nn_dropout_rate']
        self._seed       = dict_nn_layer_config['nn_seed']
        self._nb_class   = dict_nn_layer_config['nn_logit_dimension']
        
        # When value is None, then HE normal initializer is used as default. 
        self._layer_initializer_name = dict_nn_layer_config['nn_initializer_name']
        
        # Batch normalization activation
        self._is_nn_batch_norm = dict_nn_layer_config['nn_batch_norm']
        
        #---------------------------------------------------
        # Hyper parameters for CNN network
        #---------------------------------------------------
        if self._nn_type == 'CNN' or self._nn_type == 'CNNBase':
            dict_cnn_layer_config = dict_nn_layer_config['nn_layer_config']
            self._cnn_seed = self._seed

            self._conv_kernel_size = dict_cnn_layer_config['conv_kernel_size']
            
            #-------------------------------------------------------------------  
            # Number of CNN CONVOLUTIONAL layers
            # In case of AdaNet, then number of layers is provided while 
            # constructor is invoked.
            # In case of baseline calibration, then number of conv. layers 
            # is extracted from a CNN dictionary and provided while invoking 
            # constructor.
            #-------------------------------------------------------------------
            if dict_cnn_layer_config['conv_layer_num'] is None :
                #---------------------------------------------------------------
                # Conv layers will growth along with CNN candidates created from 
                # NNGenerator
                #---------------------------------------------------------------
                self._cnn_convlayer = num_layers
            else :
                #---------------------------------------------------------------
                # Conv layers size is fixed
                #---------------------------------------------------------------
                self._cnn_convlayer = dict_cnn_layer_config['conv_layer_num']            

            # Number of dense layers      
            if dict_nn_layer_config['nn_dense_layer_num'] is None :            
                #---------------------------------------------------------------
                # Dense layers will growth along with candidates created from 
                # NNGenerator
                #---------------------------------------------------------------
                self._cnn_denselayer =  num_layers
            else :
                #---------------------------------------------------------------
                # Dense layers size is fixed
                #---------------------------------------------------------------
                self._cnn_denselayer = dict_nn_layer_config['nn_dense_layer_num']            
            
            # Number of units in CNN dense layer.
            self._cnn_layersize = self._layer_size
            
            # Fixed CNN layers configuration 
            self._dict_cnn_layer_config=dict_cnn_layer_config
            
            # Batch normaization activation
            self._is_cnn_batch_norm = self._is_nn_batch_norm
        else : 
            pass

        self._start_time = 0
        
        print("\n*** NNAdaNetBuilder : NN Type={}".format(self._nn_type))
        

        
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
    def show_cnn(self) :
        print('\n')
        print("CNN seed             : ............................ {}".format(self._cnn_seed))
        print("Conv. Kernel size    : ............................ {}".format(self._conv_kernel_size))
        print("Conv. layers         : ............................ {}".format(self._cnn_convlayer))
        print("Dense layers         : ............................ {}".format(self._cnn_denselayer))
        print("Units in conv. layers: ............................ {}".format(self._cnn_layersize))
        print("CNN bacth norm.      : ............................ {}".format(self._is_cnn_batch_norm))
        print("Features map size    : ............................ {}".format(self._dict_cnn_layer_config['feature_map_size']))
        #print("Conv layers          : ............................ {}".format(self._dict_cnn_layer_config['conv_layer_num']))
        print("Conv filters         : ............................ {}".format(self._dict_cnn_layer_config['conv_filters']))
        print("Strides              : ............................ {}".format(self._dict_cnn_layer_config['conv_strides']))
        print("Padding              : ............................ {}".format(self._dict_cnn_layer_config['conv_padding_name']))
        print("Activation function  : ............................ {}".format(self._dict_cnn_layer_config['conv_activation_name']))


    
    #----------------------------------------------------------------------------
    #
    #----------------------------------------------------------------------------
    def show(self) :
        print("\n")
        print("NN type              : ............................ {}".format(self._nn_type))
        print("Units in dense layer : ............................ {}".format(self._layer_size))
        print("Number of layers     : ............................ {}".format(self._num_layers))
        print("Dropout rate         : ............................ {}".format(self._dropout))
        print("Seed value           : ............................ {}".format(self._seed))
        print("Nb of classes (logit): ............................ {}".format(self._nb_class))
        print("Weights initializer  : ............................ {}".format(self._layer_initializer_name))
        print("Batch normalization  : ............................ {}".format(self._is_nn_batch_norm))
        if self._nn_type == 'CNNBase' or self._nn_type == 'CNN' :
            self.show_cnn()
    #----------------------------------------------------------------------------
        
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
    def nn_layertype_alternate(self) :
        conv_layer = 0
        dense_layer = 0
        if self._cnn_convlayer == 0 : 
            conv_layer = self._cnn_convlayer
            dense_layer = self._cnn_denselayer
        else :
            if self._cnn_convlayer %2 == 0 : 
                conv_layer = self._cnn_convlayer
                dense_layer = 0
            else :
                conv_layer = 0
                dense_layer = self._cnn_denselayer
        return conv_layer, dense_layer
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
        # Build alternatively conv and dense layers 
        #----------------------------------------------------------------------- 
        cnn_conv_layer  = 0   
        conv_layer, dense_layer = self.nn_layertype_alternate()        
        
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
        # Flat Layer
        #-----------------------------------------------------------------------
        last_layer = tf.contrib.layers.flatten(last_layer)
                  
        #-----------------------------------------------------------------------                
        # Dense Layer(s)
        #-----------------------------------------------------------------------
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
            
        elif self._nn_type == 'CNN' or self._nn_type == 'CNNBase' :
            last_layer, logits \
            = self._build_cnn_baseline_subnetwork(input_layer, features\
            , logits_dimension, training)

            #-------------------------------------------------------------------
            # TBD : checking complexity considering Con layers and dense layers.
            #-------------------------------------------------------------------
            # Approximate the Rademacher complexity of this subnetwork as the square-
            # root of its depth.
            complexity = tf.sqrt(tf.to_float(self._cnn_convlayer+self._cnn_denselayer))
            
        else :
            print("\n*** ERROR : NN type={} no yet supported!".format(self._nn_type))
            return None
        

        if self._nn_type == 'CNN' or self._nn_type == 'CNNBase':
            if self._dict_cnn_layer_config['conv_layer_num'] is None :
                #---------------------------------------------------------------
                # Number of conv layers will growth with NNGenerator.
                #---------------------------------------------------------------
                persisted_tensors = {self._nn_type: tf.constant(self._cnn_convlayer)}
            else :
                #---------------------------------------------------------------
                # Number of dense layers will growth with NNGenerator.
                #---------------------------------------------------------------
                persisted_tensors = {self._nn_type: tf.constant(self._cnn_denselayer)}
                
        else : 
            persisted_tensors = {self._nn_type: tf.constant(self._num_layers)}
            
        with tf.name_scope(""):
            if complexity is not None :
                summary.scalar("Complexity", complexity)
            summary.scalar("Layers", tf.constant(self._num_layers))
            if self._nn_type == 'CNN' or self._nn_type == 'CNNBase':
                summary.scalar("Conv_layers", tf.constant(self._cnn_convlayer))
                summary.scalar("Dense_layers", tf.constant(self._num_layers))
                        

        return adanet.Subnetwork(
            last_layer=last_layer,
            logits=logits,
            complexity=complexity,
            shared=persisted_tensors)

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
            if self._dict_cnn_layer_config['conv_layer_num'] is None :
                num_layers = self._cnn_convlayer
            else :
                num_layers = self._num_layers
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

