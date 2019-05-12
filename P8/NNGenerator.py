import time
import tensorflow as tf
import adanet

import functools

import p8_util_config
import NNAdaNetBuilder
from NNAdaNetBuilder import NNAdaNetBuilder

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
class NNGenerator(adanet.subnetwork.Generator):
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
        
        print("\n*** NNGenerator() : feature_columns= {}".format(feature_columns))
        
        layer_size = dict_adanet_config['adanet_nn_layer_config']['nn_dense_unit_size']
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
        self._nn_builder_fn = functools.partial(NNAdaNetBuilder, dict_adanet_config)
  

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
            #print("\n*** +++ generate_candidates() : Initial Layer(s)= {}\n".format(self._initial_num_layers))
            #print("\n*** +++ generate_candidates() : previous_ensemble= {}\n".format(previous_ensemble))
            if previous_ensemble:
                num_layers = tf.contrib.util.constant_value(
                previous_ensemble.weighted_subnetworks[-1].subnetwork
                .shared[self._nn_type])
                
            else : 
                num_layers = self._initial_num_layers
                self._start_time = time.time()
            
            if False :
                list_nn_candidate = [self._nn_builder_fn(num_layers=num_layers+new_layer) \
                                     for new_layer in range(0, self._nb_nn_candidate)]
                return list_nn_candidate
            else :
                # Returns a list of instanciated classes that implement 
                # subnetworks candidates.
                print("\n*** NNGenerator : layers= ({},{})".format(num_layers, num_layers+1))
                if False :
                    return [
                        self._nn_builder_fn(num_layers=num_layers),
                        self._nn_builder_fn(num_layers=num_layers + 1),]
                else :
                    adanet_feature_shape = p8_util_config.dict_adanet_config['adanet_feature_shape']
                    nn1 = self._nn_builder_fn(num_layers=num_layers)
                    nn1.feature_shape = adanet_feature_shape
                    
                    nn2 = self._nn_builder_fn(num_layers=num_layers + 1)
                    nn2.feature_shape = adanet_feature_shape
                    return [nn1, nn2]
                
#-------------------------------------------------------------------------------    

