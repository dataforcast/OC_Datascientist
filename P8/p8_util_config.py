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
TRAIN_STEPS = 9
BATCH_SIZE = 138
NUM_EPOCHS = 1
MAX_STEPS = NUM_EPOCHS*TRAIN_STEPS





LEARNING_RATE = 1.e-4
#NN_TYPE = 'CNN'
NN_TYPE = 'CNNBase'
nb_class = 3
DENSE_UNIT_SIZE = 10

CONV_NUM_LAYERS  = 2
DENSE_NUM_LAYERS = 1
IS_BATCH_NORM = True

DROPOUT_RATE = 0.0
CONV_KERNEL_SIZE=(3,3)
CONV_FILTERS = 32
CONV_STRIDES =1
CONV_PADDING_NAME ='same'
CONV_ACTIVATION_NAME = 'relu'

NN_NUM_LAYERS   = DENSE_NUM_LAYERS+CONV_NUM_LAYERS


SEED=p8_util.RANDOM_SEED
INITIALIZER_NAME = 'xavier'


OPTIMIZER=tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)

ADANET_INITIAL_NUM_LAYERS = 0
ADANET_NN_CANDIDATE = 2
ADANET_LAMBDA = 0.02
ADANET_TRAIN_STEPS_PER_CANDIDATE = TRAIN_STEPS  #@param {type:"integer"}
ADANET_ITERATIONS = 3  #@param {type:"integer"}

#-------------------------------------------------------------------------------
# Every ADANET_TRAIN_STEPS_PER_CANDIDATE then a new candidate will be generated
# Max number of ADANET iterations is 30//3 = 3
# Then AdaNet will build 3 subnetwoks :
# Step 1 : 1 Dense layer (Linear)
# Step 2 : 1 Conv layer + 1 Dense layer 
# Step 3 : 2 Conv layers + 1 dense layer
#-------------------------------------------------------------------------------
ADANET_MAX_ITERATION_STEPS=MAX_STEPS//ADANET_ITERATIONS

#optimizer = tf.keras.optimizers.SGD(lr=LEARNING_RATE)
#optimizer = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE)
#optimizer = keras.optimizers.SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True)
tuple_dimension = (224,224,3)
my_feature_columns, loss_reduction, tf_head = p8_util.get_tf_head("images",tuple_dimension, nb_class)
feature_columns = my_feature_columns
is_learn_mixture_weights = False

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
                        , 'nn_logit_dimension':nb_class
                        , 'nn_optimizer':OPTIMIZER
                        , 'nn_seed' : SEED
                        , 'nn_initializer_name' : INITIALIZER_NAME
                        , 'nn_layer_config':dict_cnn_layer_config
                       }
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

