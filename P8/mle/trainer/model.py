#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import shutil

import p8_util
import p8_util_config

import NNGenerator

IS_DEBUG=False
RANDOM_SEED=42

FEATURES_KEY = 'images'
# Define your feature columns

FEATURE_SHAPE= p8_util_config.dict_adanet_config['adanet_feature_shape']

INPUT_COLUMNS = [
    tf.feature_column.numeric_column(str(i)) for i in range(0,FEATURE_SHAPE[0],1)
]

#-------------------------------------------------------------------------------
# Create your serving input function so that your trained model will be able to 
# serve predictions
#-------------------------------------------------------------------------------
def serving_input_fn():
    feature_placeholders = {
        column.name: tf.placeholder(tf.float32, [None]) for column in INPUT_COLUMNS
    }
    features = feature_placeholders
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def train_and_evaluate(args):
    subnetwork_generator=NNGenerator.NNGenerator(p8_util_config.dict_adanet_config)
    adanet_estimator, output_dir_log = p8_util.build_adanet_estimator(subnetwork_generator)
    result, _= p8_util.train_and_evaluate(adanet_estimator)
    return  result, _

#-------------------------------------------------------------------------------
    
