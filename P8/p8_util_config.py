'''This is a global configuration file for parameters involved into ADANET 
networks.
'''
import tensorflow as tf
import p8_util


#-------------------------------------------------------------------------------
# They are 414 data 
# Batch size is 138
# Then there is 414/138 = 3 steps for covering all data.
# Then 1  EPOCH contains 3 steps.
# If NUM EPOCH is 20, then data will be processed 20 times, each time within 3 steps.
#-------------------------------------------------------------------------------
NUM_EPOCHS = 6
TRAIN_STEPS = 600
BATCH_SIZE = 138//4
MAX_STEPS = TRAIN_STEPS

LEARNING_RATE = 1.e-3
#NN_TYPE = 'CNN'
#NN_TYPE = 'DNN'
NN_TYPE = 'RNN'
NN_TYPE = 'CNNBase'
NN_TYPE = 'RNN'
NN_TYPE = 'GRU'
NB_CLASS = 3
DENSE_UNIT_SIZE = 10

#-------------------------------------------------------------------------------
# When None, then dense layer will growth with number of layers 
# provided from NNGenerator
# Otherwise, CNN is built at each Adanet iteration with same number of dense 
#layers.
#-------------------------------------------------------------------------------
DENSE_NUM_LAYERS = 1
#DENSE_NUM_LAYERS = None
#-------------------------------------------------------------------------------

IS_BATCH_NORM = True
DROPOUT_RATE = 0.0

#-------------------------------------------------------------------------------
# When CONV_NUM_LAYERS value is None, then conv. layers will growth with number 
# of layers provided from NNGenerator.
# Otherwise, CNN is built at each Adanet iteration with same number of conv. 
# layers.
# For CNN baseline, value has to be >0 and NN type fixed to CNNBase.
#-------------------------------------------------------------------------------
#CONV_NUM_LAYERS  = None
CONV_NUM_LAYERS  = 2
#-------------------------------------------------------------------------------

CONV_KERNEL_SIZE=(5,5)
CONV_FILTERS = 32
CONV_STRIDES =1
CONV_PADDING_NAME ='same'
CONV_ACTIVATION_NAME = 'relu'

#NN_NUM_LAYERS   = DENSE_NUM_LAYERS+CONV_NUM_LAYERS


# The random seed to use.
RANDOM_SEED = 42
SEED = RANDOM_SEED

INITIALIZER_NAME = 'xavier'
#-------------------------------------------------------------------------------
# In case of CNN network
#-------------------------------------------------------------------------------
OPTIMIZER=tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)

#-------------------------------------------------------------------------------
# RNN Network
#-------------------------------------------------------------------------------
RNN_ACTIVATION_NAME = 'relu'
RNN_HIDDEN_UNITS = 128 
RNN_NUM_LAYERS = 1
RNN_TIMESTEPS = 224 
if NN_TYPE == 'RNN' :
    OPTIMIZER=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    INITIALIZER_NAME = 'truncated_normal'

#-------------------------------------------------------------------------------
# Adanet hyper-parameters
#-------------------------------------------------------------------------------
ADANET_INITIAL_NUM_LAYERS = 0
ADANET_NN_CANDIDATE = 2
ADANET_LAMBDA = 0.005
ADANET_TRAIN_STEPS_PER_CANDIDATE = TRAIN_STEPS  #@param {type:"integer"}
ADANET_ITERATIONS = 10  #@param {type:"integer"}

#-------------------------------------------------------------------------------
# Every ADANET_TRAIN_STEPS_PER_CANDIDATE then a new candidate will be generated
# Max number of ADANET iterations is 30//3 = 3
# Then AdaNet will build 3 subnetwoks :
# Step 1 : 1 Dense layer (Linear)
# Step 2 : 1 Conv layer + 1 Dense layer 
# Step 3 : 2 Conv layers + 1 dense layer
#-------------------------------------------------------------------------------
ADANET_MAX_ITERATION_STEPS=TRAIN_STEPS//ADANET_ITERATIONS

#optimizer = tf.keras.optimizers.SGD(lr=LEARNING_RATE)
#optimizer = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE)
#optimizer = keras.optimizers.SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True)

#-------------------------------------------------------------------------------
# To be changed depending of feautures
#-------------------------------------------------------------------------------
tuple_dimension = (224,224,3)
#tuple_dimension = (28,28)
my_feature_columns, loss_reduction, tf_head = p8_util.get_tf_head("images",tuple_dimension, NB_CLASS, nn_type=NN_TYPE)
feature_columns = my_feature_columns



is_learn_mixture_weights = False
#---------------------------------------------------
# Hyper parameters for RNN network
#---------------------------------------------------
dict_rnn_layer_config={ 'rnn_layer_num':RNN_NUM_LAYERS
                       ,'rnn_hidden_units' : RNN_HIDDEN_UNITS
                       ,'rnn_activation_name' : RNN_ACTIVATION_NAME
                       ,'rnn_timesteps' : RNN_TIMESTEPS
                    }

#---------------------------------------------------
# Hyper parameters for CNN network
#---------------------------------------------------
dict_cnn_layer_config={ 'feature_map_size':[64,]
                      ,'conv_kernel_size':CONV_KERNEL_SIZE
                      ,'conv_layer_num':CONV_NUM_LAYERS
                      ,'conv_filters' : CONV_FILTERS
                      ,'conv_strides' :  CONV_STRIDES
                      ,'conv_padding_name' : CONV_PADDING_NAME
                      ,'conv_activation_name' : CONV_ACTIVATION_NAME
                      }

#---------------------------------------------------
# Hyper parameters for NN Builder
#---------------------------------------------------
dict_nn_layer_config = {  'nn_type':NN_TYPE
                        , 'nn_dropout_rate': DROPOUT_RATE
                        , 'nn_batch_norm' : IS_BATCH_NORM
                        , 'nn_dense_layer_num' : DENSE_NUM_LAYERS
                        , 'nn_dense_unit_size':DENSE_UNIT_SIZE
                        , 'nn_logit_dimension':NB_CLASS
                        , 'nn_optimizer':OPTIMIZER
                        , 'nn_seed' : SEED
                        , 'nn_initializer_name' : INITIALIZER_NAME
                        , 'nn_layer_config':dict_cnn_layer_config
                       }
if NN_TYPE == 'RNN' :
    dict_nn_layer_config['nn_layer_config'] = dict_rnn_layer_config

#---------------------------------------------------
# AdaNet configuration
#---------------------------------------------------
dict_adanet_config = {  'adanet_feature_columns': feature_columns
                      , 'adanet_tf_head' : tf_head
                      , 'adanet_lambda' : ADANET_LAMBDA
                      , 'adanet_is_learn_mixture_weights': True
                      , 'adanet_initial_num_layers': ADANET_INITIAL_NUM_LAYERS
                      , 'adanet_num_layers': None
                      , 'adanet_nn_candidate' : ADANET_NN_CANDIDATE
                      , 'adanet_nn_layer_config' : dict_nn_layer_config}

