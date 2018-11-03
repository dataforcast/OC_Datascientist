import pandas as pd
import numpy as np
import time

from scipy import sparse

import p3_util
import p5_util

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def model_load(file_name=None):
   """ Load dumped object handled by file_name.
   
   If file_name is None, then default file name is used.

   Input : 
      * file_name : file handling dumped object
   Output :
      * object : loaded dumped object.
   """
   if file_name is None :
      file_name = "./data/_oP5_SegmentClassifier.dump"
   else:
      pass

   return p5_util.object_load(file_name)
#-------------------------------------------------------------------------------

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def _print_stat_rows(title,rows_before,rows_after):
   """Print percentage of rows that have been processed.
   """
   self.strprint(str(title)+" : Percent of processed rows = %1.2F"\
    %(np.abs(rows_before-rows_after)*100/rows_before))
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def print_stat_rows(title,rows_before,rows_after):
   """Print percentage of rows that have been processed.
   """
   _print_stat_rows(title,rows_before,rows_after)
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def model_dump( my_object, file_name):
   if file_name is None:
      file_name = self._path_to_model
      
   p5_util.object_dump(my_object,file_name)
#----------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
class P5_SegmentClassifier() :
   ''' 
   This class implements a segment classifier model.
    
      
   Instructions flow :
   -------------------
      +-->__init__()
      |
      .
      .
      // Building step
      .
      .
      |
      +-->data_load(fileName)
      |
      +-->data_clean(fileName)
      |
      +-->data_transform()
      |   |
      |   +-->data_transform_rfm()
      |   |
      |   +-->data_transform_timeFeature()
      |   |  |
      |   |  +-->p5_util.time_list_feature_build()
      |   |     |
      |   |     +-->p5_util.time_feature_encoded_new()
      |   |
      |   |
      |   +-->data_transform_nlp()
      |
      +-->clusters_build()
      |
      +-->classifier_build()
      |
      +-->model_dump()
      |
      .
      .
      // Exploitation step
      .
      .
      |      
      +-->order_process()
      |   |
      |   +-->createCustomerID()
      |   |
      |   +-->create_customer_df_invoice_line()
      |   |
      |   +-->get_customer_marketSegment()
      |       |
      |       +-->data_transform()
      |       |
      |       +-->df_customers_features_build()
      |       |
      |       +-->_classifier_model.predict()
      |
      .
      .
      // Tools and utilities
      .
      .
      |      
      +-->print
      |      
      +-->strprint
      |      
      +-->get_customer_marketSegment
      |      
      +-->get_listCustomer_out_sample
      |      
      +-->getStockCodeList
      |      
      +-->getUnitPriceList
      |      
      +-->get_customer_history_df_invoice_line
      |      
      +-->getDescriptionList
      |      
      +-->is_customer_out_sample
      
      
      
   '''
   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def __init__(self, path_to_model=None) :


      #-------------------------------------------------------------------------
      # Debug attributes
      #-------------------------------------------------------------------------
      self.dbg_df = None
      self._is_verbose = True
      


      #-------------------------------------------------------------------------
      # Data-model parameters
      #-------------------------------------------------------------------------
      if None is path_to_model:
         path_to_model = "./data/_oP5_SegmentClassifier.dump"
      self._path_to_model = path_to_model
      self._df_invoice_original = pd.DataFrame()
      self._df_invoice_line = pd.DataFrame()
      self._total_outliers = False
      self._df_invoice_ref = pd.DataFrame()
      self._list_quant_feature = list()
      self._list_feature_to_drop = ['InvoiceNo','StockCode','InvoiceDate'\
      ,'CustomerID','Country','Total','RFM']
      self._arr_sample_customerID = None
      self._df_invoice_line_out_sample = pd.DataFrame()
      
      #-------------------------------------------------------------------------
      # RFM features
      #-------------------------------------------------------------------------
      self._is_rfm_encode = False
      self._encoder_rfm = None
      self._df_customers_rfm = pd.DataFrame()
      self._df_RFM_quantiles = None
      self._day_now = None
      self._is_transform_rfm = True
      self.df_customers_rfm_fileName = './data/df_customers_rfm.dump'

      #-------------------------------------------------------------------------
      # Time features
      #-------------------------------------------------------------------------
      self._is_transform_timeFeature = True
      self._list_new_feature = ['month','day','dow','pod']
      self._pca_timeFeature = None
      self._std_scaler_timeFeature = None
      self._df_customers_timeFeature_fileName \
      = './data/df_customers_timeFeature_pca.dump'
      self._dict_timeFeature_encoder = None
      self._df_customers_timeFeature = pd.DataFrame()
      
      #-------------------------------------------------------------------------
      # NLP features
      #-------------------------------------------------------------------------
      self._is_transform_nlp = True
      self._vectorizer_nlp = None
      self._matrix_weights_nlp = None
      self._df_customers_pca_nlp = pd.DataFrame()
      self._df_w_nlp = pd.DataFrame()
      self._df_customers_nlp_fileName = './data/df_customers_nlp_pca.dump'
      self._pca_nlp = None
      self._nlp_pca_ndim = 250
      
      #-------------------------------------------------------------------------
      # All features
      #-------------------------------------------------------------------------
      self._df_customers_fileName = './data/df_customers.dump'
      self._df_customers = pd.DataFrame()

      #-------------------------------------------------------------------------
      # Classifier
      #-------------------------------------------------------------------------
      self._y_clusters = None
      self._dict_classifier_param = dict()
      self._classifier_name = None
      self._classifier_model = None


      self.strprint("P5_SegmentClassifier : init done!")
   #----------------------------------------------------------------------------
   
   #----------------------------------------------------------------------------
   # 
   #----------------------------------------------------------------------------
   def strprint(self, mystr):
      """Encapsulation of print function.
      If flag is_verbose is fixed to True, then print takes place.
      
      Input :
         * mystr : string to be printed.

      Output : none
      
      """
      if self.is_verbose is True:
         print(mystr)
      else:
         pass
      return
   #----------------------------------------------------------------------------
   
   #----------------------------------------------------------------------------
   # Properties
   #----------------------------------------------------------------------------
   def _get_path_to_model(self) :
      return self._path_to_model
      
   def _set_path_to_model(self, path_to_model) :
      self._path_to_model = path_to_model
   #----------------------------------------------------------------------------      
   def _get_encoder_rfm(self) :
      return self._encoder_rfm
      
   def _set_encoder_rfm(self, rfm_encoder) :
      self._encoder_rfm = rfm_encoder
   #----------------------------------------------------------------------------
   def _get_vectorizer(self) :
      return self._vectorizer
      
   def _set_vectorizer(self, vectorizer) :
      self._vectorizer = vectorizer
   #----------------------------------------------------------------------------      
   def _get_total_outliers(self):
      return self._total_outliers
      
   def _set_total_outliers(self, total_outliers):
      self._total_outliers = total_outliers
   #----------------------------------------------------------------------------
   def _get_y_clusters(self):
      return self._y_clusters
      
   def _set_y_clusters(self, y_clusters):
      self._y_clusters = y_clusters.copy()
   #----------------------------------------------------------------------------      
   def _get_dict_classifier_param(self):
      return self._dict_classifier_param
      
   def _set_dict_classifier_param(self, dict_classifier_param):
      self._dict_classifier_param = dict_classifier_param.copy()

   #----------------------------------------------------------------------------      
   def _get_classifier_name(self):
      return self._classifier_name

   def _set_classifier_name(self, classifier_name):
      self._classifier_name = classifier_name
   #----------------------------------------------------------------------------      
   def _get_predictor_model(self):
      return self._predictor_model
      
   def _set_predictor_model(self, predictor_model):
      self._predictor_model = predictor_model
      
   #----------------------------------------------------------------------------      
   def _get_std_scaler_timeFeature(self):
      return self._std_scaler_timeFeature
      
   def _set_std_scaler_timeFeature(self, std_scaler_timeFeature):
      self._std_scaler_timeFeature = std_scaler_timeFeature
   #----------------------------------------------------------------------------      
   def _get_list_quant_feature(self):
      return self._list_quant_feature

   def _set_list_quant_feature(self, list_quant_feature):
      self._list_quant_feature = list_quant_feature.copy()
      
   #----------------------------------------------------------------------------      
   def _get_df_invoice_line(self):
      return self._df_invoice_line.copy()

   def _set_df_invoice_line(self, df_invoice_line):
      self._df_invoice_line = df_invoice_line.copy()

   #----------------------------------------------------------------------------      
   def _get_df_customers_timeFeature(self):
      return self._df_customers_timeFeature

   def _set_df_customers_timeFeature(self, df_customers_timeFeature):
      self._df_customers_timeFeature = df_customers_timeFeature.copy()

   #----------------------------------------------------------------------------
   def _get_nlp_pca_ndim(self):
      return self._nlp_pca_ndim
      
   def _set_nlp_pca_ndim(self, nlp_pca_ndim):
      self._nlp_pca_ndim = nlp_pca_ndim
      
   #----------------------------------------------------------------------------      
   def get_df_RFM_quantiles(self):
      if self._df_RFM_quantiles is not None : 
         return self._df_RFM_quantiles.copy()
      else:
         return None

   def set_df_RFM_quantiles(self, df_RFM_quantiles):
      if df_RFM_quantiles is not None:
         self._df_RFM_quantiles = df_RFM_quantiles.copy()
      else:
         self._df_RFM_quantiles = None
   #----------------------------------------------------------------------------         
   def _get_is_transform_rfm(self):
      return self._is_transform_rfm
      
   def _set_is_transform_rfm(self, is_transform_rfm):
      self._is_transform_rfm = is_transform_rfm
   #----------------------------------------------------------------------------      
   def _get_is_transform_nlp(self):
      return self._is_transform_nlp
      
   def _set_is_transform_nlp(self, is_transform_nlp):
      self._is_transform_nlp = is_transform_nlp
   #----------------------------------------------------------------------------      
   def _get_is_transform_timeFeature(self):
      return self._is_transform_timeFeature

   def _set_is_transform_timeFeature(self, is_transform_timeFeature):
      self._is_transform_timeFeature = is_transform_timeFeature
   #----------------------------------------------------------------------------
   def _get_is_verbose(self):
      return self._is_verbose

   def _set_is_verbose(self, is_verbose):
      self._is_verbose = is_verbose
   #----------------------------------------------------------------------------
   
   path_to_model = property(_get_path_to_model, _set_path_to_model)
   rfm_encoder = property(_get_encoder_rfm, _set_encoder_rfm)
   vectorizer = property(_get_vectorizer, _set_vectorizer)   
   total_outliers = property(_get_total_outliers, _set_total_outliers)
   dict_classifier_param = property(_get_dict_classifier_param\
   , _set_dict_classifier_param)
   classifier_name = property(_get_classifier_name, _set_classifier_name)
   predictor_model = property(_get_predictor_model, _set_predictor_model)
   std_scaler_timeFeature = property(_get_std_scaler_timeFeature\
   , _set_std_scaler_timeFeature)
   list_quant_feature = property(_get_list_quant_feature\
   , _set_list_quant_feature)
   df_invoice_line = property(_get_df_invoice_line, _set_df_invoice_line)
   df_customers_timeFeature = property(_get_df_customers_timeFeature\
   , _set_df_customers_timeFeature)
   nlp_pca_ndim = property(_get_nlp_pca_ndim , _set_nlp_pca_ndim)
   df_RFM_quantiles = property(get_df_RFM_quantiles, set_df_RFM_quantiles)
   
   is_transform_nlp = property(_get_is_transform_nlp, _set_is_transform_nlp)
   is_transform_rfm = property(_get_is_transform_rfm, _set_is_transform_rfm)
   is_transform_timeFeature = property(_get_is_transform_timeFeature\
   , _set_is_transform_timeFeature)
   is_verbose = property(_get_is_verbose, _set_is_verbose)
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def copy(self, other_object):
      """Copy all attributes from a given object P5_SegmentClassifier object 
      into self.
      
      Input : 
         * other_object : P5_SegmentClassifier object to be copied
      Output : none
         
      """
      #-------------------------------------------------------------------------
      # Debug parameters
      #-------------------------------------------------------------------------
      self._path_to_model = other_object.path_to_model
      
      #-------------------------------------------------------------------------
      # Data-model parameters
      #-------------------------------------------------------------------------
      self.is_verbose = other_object.is_verbose

      self._df_invoice_line = other_object._df_invoice_line.copy()
      self._total_outliers = other_object._total_outliers
      self._df_invoice_ref = other_object._df_invoice_ref.copy()
      self._list_quant_feature = other_object._list_quant_feature.copy()
      self._list_feature_to_drop = other_object._list_feature_to_drop.copy()
      self._df_invoice_original = other_object._df_invoice_original.copy()
      if other_object._arr_sample_customerID is not None:
         self._arr_sample_customerID = other_object._arr_sample_customerID.copy()
      else :
         self._arr_sample_customerID = None
         
      self._df_invoice_line_out_sample \
      = other_object._df_invoice_line_out_sample.copy()
      
      #-------------------------------------------------------------------------
      # RFM features
      #-------------------------------------------------------------------------
      self._is_rfm_encode = other_object._is_rfm_encode
      self._encoder_rfm = other_object._encoder_rfm
      self._df_customers_rfm = other_object._df_customers_rfm.copy()
      self.df_customers_rfm_fileName = other_object.df_customers_rfm_fileName
      self.df_RFM_quantiles = other_object.df_RFM_quantiles
      self._day_now = other_object._day_now
      self._is_transform_rfm = other_object._is_transform_rfm

      #-------------------------------------------------------------------------
      # Time features
      #-------------------------------------------------------------------------
      self._list_new_feature = other_object._list_new_feature
      self._pca_timeFeature = other_object._pca_timeFeature
      self._std_scaler_timeFeature = other_object._std_scaler_timeFeature
      
      self._df_customers_timeFeature_fileName \
      = other_object._df_customers_timeFeature_fileName
      
      if other_object._dict_timeFeature_encoder is not None:
         self._dict_timeFeature_encoder \
         = other_object._dict_timeFeature_encoder.copy()
      else:
         self._dict_timeFeature_encoder = other_object._dict_timeFeature_encoder
      
      if other_object._df_customers_timeFeature is not None:
         self._df_customers_timeFeature \
         = other_object._df_customers_timeFeature.copy()
      else:
         self._df_customers_timeFeature = other_object._df_customers_timeFeature
         
      self._is_transform_timeFeature = other_object._is_transform_timeFeature
      
      #-------------------------------------------------------------------------
      # NLP features
      #-------------------------------------------------------------------------
      self._vectorizer_nlp = other_object._vectorizer_nlp 
      self._matrix_weights_nlp = other_object._matrix_weights_nlp
      self._df_customers_nlp_fileName = other_object._df_customers_nlp_fileName
      self._pca_nlp = other_object._pca_nlp
      self._df_customers_pca_nlp = other_object._df_customers_pca_nlp.copy()
      self._nlp_pca_ndim = other_object._nlp_pca_ndim
      self._is_transform_nlp = other_object._is_transform_nlp
      
      #-------------------------------------------------------------------------
      # All features
      #-------------------------------------------------------------------------
      self._df_customers_fileName = other_object._df_customers_fileName
      self._df_customers = other_object._df_customers.copy()

      #-------------------------------------------------------------------------
      # Classifier
      #-------------------------------------------------------------------------
      if other_object._y_clusters is not None:
         self._y_clusters = other_object._y_clusters.copy()
      else:
         self._y_clusters = other_object._y_clusters

      self._dict_classifier_param = other_object._dict_classifier_param.copy()
      self._classifier_name = other_object._classifier_name
      self._classifier_model = other_object._classifier_model
      
   #----------------------------------------------------------------------------
   
   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def print(self):
      """ Displays object attributs"""
      is_verbose_save = self.is_verbose
      self.is_verbose = True
      self.strprint("\n    Debug parameters ")
      self.strprint("Verbose  ................: "+str(is_verbose_save))   

      self.strprint("\n    Data-model parameters ")

      self.strprint("Path  ...................: "+str(self._path_to_model))   

      self.strprint("Original invoice lines : ..........................: "\
      +str(self._df_invoice_original.shape))   

      self.strprint("Out of sampling data ..............................: "\
      +str(self._df_invoice_line_out_sample.shape))   

      self.strprint("Invoice lines : ...................: "\
      +str(self._df_invoice_line.shape))   

      self.strprint("Total outliers flag  ..............................: "\
      +str(self._total_outliers))   

      self.strprint("List of quantitatives features  ...................: "\
      +str(self._list_quant_feature))   

      self.strprint("List of features to be droped .....................: "\
      +str(self._list_feature_to_drop))   

      #-------------------------------------------------------------------------
      # RFM features
      #-------------------------------------------------------------------------
      self.strprint("\n     RFM features         ")
     
      self.strprint("RFM features ......................................: "\
      +str( self._df_customers_rfm.shape))   

      self.strprint("RFM transformation ................................: "\
      +str( self.is_transform_rfm))   

      self.strprint("RFM features file name ............................: "\
      +str(self.df_customers_rfm_fileName))   

      self.strprint("Most recent day ...................................: "\
      +str(self._day_now))   

      df_RFM_quantiles = self.df_RFM_quantiles
      df_RFM_quantiles['neg_recency'] = sorted( df_RFM_quantiles['neg_recency'])
      self.strprint("RFM quantile scheme ...............................: "\
      +str(df_RFM_quantiles))   
   
      self.strprint("RFM encoding flag .................................: "\
      +str(self._is_rfm_encode))   

      self.strprint("RFM encoder .......................................: "\
      +str(self._encoder_rfm))   

      #-------------------------------------------------------------------------
      # Time features
      #-------------------------------------------------------------------------
      self.strprint("\n     Time features         ")   

      self.strprint("Time features transformation ......................: "\
      +str( self.is_transform_timeFeature))   

      self.strprint("Time features file name ...........................: "\
      +str(self._df_customers_timeFeature_fileName))   

      self.strprint("Time new features list ............................: "\
      +str(self._list_new_feature))   

      self.strprint("Time features PCA reducer .........................: "\
      +str(self._pca_timeFeature))   

      self.strprint("Time features standard scaler .....................: "\
      +str(self._std_scaler_timeFeature))   

      self.strprint("Time features encoders ............................: "\
      +str(self._dict_timeFeature_encoder))   

      self.strprint("Time features dataframe ...........................: "\
      +str(self._df_customers_timeFeature.shape))   

      
      #-------------------------------------------------------------------------
      # NLP features
      #-------------------------------------------------------------------------
      self.strprint("\n     NLP features         ")

      self.strprint("NLP transformations...............................: "\
      +str( self.is_transform_nlp))   

      self.strprint("NLP features file name ...........................: "\
      +str(self._df_customers_nlp_fileName))   
      
      self.strprint("NLP features vectorizer ..........................: "\
      +str(self._vectorizer_nlp))   
      
      self.strprint("NLP features PCA reducer .........................: "\
      +str(self._pca_nlp))   
      
      self.strprint("NLP dataframe ....................................: "\
      +str(self._df_customers_pca_nlp.shape))   
      
      if self._matrix_weights_nlp is not None:
         self.strprint("NLP features matrix weight .......................: "\
         +str(self._matrix_weights_nlp.shape))   
      else:
         self.strprint("NLP features matrix weight .......................: "\
         +str(self._matrix_weights_nlp))   

      self.strprint("NLP PCA dimension ...................................: "\
      +str(self._nlp_pca_ndim))   

      #-------------------------------------------------------------------------
      # All features
      #-------------------------------------------------------------------------
      self.strprint("\n      ALL features         ")
      self.strprint("All features file name ...........................: "\
      +str(self._df_customers_fileName))   
      self.strprint("Customers dataframe ..............................: "\
      +str(self._df_customers.shape))   
      

      #-------------------------------------------------------------------------
      # Classifier
      #-------------------------------------------------------------------------
      self.strprint("\n        Classifier        ")
      if self._y_clusters is not None:
         self.strprint("Clusters vector ...........................: "\
         +str(np.unique(self._y_clusters)))
      else:
         self.strprint("Clusters vector ...........................: "\
         +str(self._y_clusters))   
      
      self.strprint("Classifier name ...........................: "\
      +str(self._classifier_name))   

      self.strprint("Classifier model ..........................: "\
      +str(self._classifier_model))   

      self.strprint("Classifier hyper-parameters ...............: "\
      +str(self._dict_classifier_param))   

      self.is_verbose = is_verbose_save
   #----------------------------------------------------------------------------
   
   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def _feature_country_process(self):
      """Remove raws with countries other then 'United Kingdom' then 
      remove Country feature.
      Input : none
      Output : none
      """
      if 'Country' not in self._df_invoice_line.columns:
         return

      list_countries_keep = ['United Kingdom']
      rows_before = self._df_invoice_line.shape[0]
      
      df_invoice_line_new = pd.DataFrame()
      for country in list_countries_keep : 
         df_invoice_line_new = df_invoice_line_new.append(\
         self._df_invoice_line[self._df_invoice_line['Country']==country]\
         , ignore_index=True)

      self.df_invoice_line = df_invoice_line_new
      del(df_invoice_line_new)
      
      rows_after = self._df_invoice_line.shape[0]      
      _print_stat_rows("Countries filtering : ",rows_before, rows_after)

      
      #-------------------------------------------------------------------------
      # Due to the fact only one country is used, then this feature is dropped
      #-------------------------------------------------------------------------
      list_col_to_keep = [col for col in self._df_invoice_line.columns \
      if col not in 'Country']
      
      self._df_invoice_line = self._df_invoice_line[list_col_to_keep]      

      return
   #----------------------------------------------------------------------------
   
   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def data_transform_timeFeature(self):
      """ Builds features issued from InvoiceDate.
      A dataframe is built per new feature and dumped into a file.
      Each one of the dataframe have encoded features issues from InvoiceDate.
      Files names are under format : ./data/df_customers_<new_feature>.dump
      
      When this method is called during building data_model step, then 
      dataframe handling new time features is dumped into a file.

      Input : none
      
      Output :
         * dataframe with RFM encoded values per customeris dumped into file 
         ./data/df_customers_timeFeature_pca.dump
      
      """
      #-------------------------------------------------------------------------
      # All new features are built into separate dataframes 
      # and each of them are dumped into a separate file.
      #-------------------------------------------------------------------------
      self.strprint("self.df_invoice_line : "+str(self.df_invoice_line.shape))
      
      self._dict_timeFeature_encoder, df_customers_timeFeature \
      = p5_util.time_list_feature_build(self.df_invoice_line\
      , self._list_new_feature, dict_encoder = self._dict_timeFeature_encoder\
      ,is_verbose=self.is_verbose)
      
      #-------------------------------------------------------------------------
      # New time features are aggregated into a single dataframe.
      # Values are scaled.
      #-------------------------------------------------------------------------
      df_customers_timeFeature, self._std_scaler_timeFeature \
      = p5_util.time_list_feature_restore(self._list_new_feature \
      , std_scale = self._std_scaler_timeFeature\
      , df_timeFeature = df_customers_timeFeature, is_verbose = self.is_verbose)

      self.strprint("df_customers_timeFeature : "+str(df_customers_timeFeature.shape))
   
      #-------------------------------------------------------------------------
      # Dimension reduction thanks to PCA
      #-------------------------------------------------------------------------
      n_dim=30
      root_name = 'time_pca_'
      # Column CustomerID is used into df_pca_reduce
      df_customers_timeFeature['CustomerID'] = df_customers_timeFeature.index
      
      df_customers_timeFeature, pca_timeFeature \
      = p5_util.df_pca_reduce(df_customers_timeFeature, n_dim, root_name\
      , p_is_scale=False, pca = self._pca_timeFeature)

      self.strprint(df_customers_timeFeature.shape)
      
      if self._pca_timeFeature is None:
         #----------------------------------------------------------------------
         # Data-model is in built process with part of data-set.
         #----------------------------------------------------------------------
         self._pca_timeFeature = pca_timeFeature
         p5_util.object_dump(df_customers_timeFeature\
         , self._df_customers_timeFeature_fileName)
      else:
         #----------------------------------------------------------------------
         # Data-model is already built and this method is called 
         # for a customer classification.
         #----------------------------------------------------------------------
         self._df_customers_timeFeature = df_customers_timeFeature.copy()
      return
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def data_transform_rfm(self) :
      """ Builds for each customer, RFM scores and encode scores.

      When this method is called during building data_model step, then 
      dataframe handling new RFM features is dumped into a file.

      Input : none

      Output :
         * dataframe with RFM encoded values per customeris dumped into file 
         ./data/df_customers_rfm.dump
            
      """
      
      is_built_step = False
      if self._encoder_rfm is None:
         is_built_step = True  
      
      #-------------------------------------------------------------------------
      # RFM feature is built
      #-------------------------------------------------------------------------
      ser_invoice_date = self._df_invoice_line.InvoiceDate
      
      self.df_invoice_line, df_RFM, self.df_RFM_quantiles, self._day_now \
      = p5_util.p5_df_rfm_build(self.df_invoice_line, day_now = self._day_now\
      , df_RFM_threshold=self.df_RFM_quantiles)
      
      self._df_invoice_line.InvoiceDate = ser_invoice_date
      
      #-------------------------------------------------------------------------
      # RFM score is added to dataframe
      #-------------------------------------------------------------------------
      df_merged = pd.merge(self.df_invoice_line\
      , df_RFM[['CustomerID','RFM']], how='left', on=['CustomerID'])

      self._df_invoice_line \
      = pd.DataFrame(df_merged.values, index = self._df_invoice_line.index\
      , columns=df_merged.columns)
      

      #self._df_invoice_line \
      #= pd.concat([ self.df_invoice_line,df_RFM[['CustomerID','RFM']] ], axis=1\
      #,join='inner')
         
      
      #-------------------------------------------------------------------------
      # RFM encoding
      #-------------------------------------------------------------------------
      self._encoder_rfm, df_RFM_encoded \
      = p5_util.df_rfm_one_hot_encode(df_RFM,'RFM', encoder=self._encoder_rfm)

      #-------------------------------------------------------------------------
      # Encoded RFM features are renamed
      #-------------------------------------------------------------------------
      df_customers_rfm, list_col_unchanged \
      = p5_util.df_rename_columns(df_RFM_encoded, df_RFM_encoded.columns\
      , 'w_rfm_')
      
      self.strprint("df_customers_rfm =" +str(df_customers_rfm.shape))

      #-------------------------------------------------------------------------
      # dataframe with RFM encoded values per customer is dumped
      #-------------------------------------------------------------------------
      if is_built_step is True:
         p5_util.object_dump(df_customers_rfm, self.df_customers_rfm_fileName)
      else :
         self._df_customers_rfm = df_customers_rfm.copy()
      return
   #----------------------------------------------------------------------------


   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def data_transform_nlp(self):
      """Creates new features from Description feature thanks to NLTK, a 
      NLP package.
      NLP features are handled into a dataframe. A PCA reduction is applied on 
      this dataframe.
      Features from dataframe are renamed with root ane w_nlp.
      
      When this method is called during building data_model step, then 
      dataframe handling new NLP feature is dumped into a file.

      Input : none
      
      Output :
         * dataframe with RFM encoded values per customeris dumped into file 
         ./data/df_customers_nlp_pca.dump

      """
      df_invoice_line = None
      
      is_build_step = False

      if self._vectorizer_nlp is None:
         is_build_step = True
         
      list_no_words=['SET','PACK']

      df_invoice_line, csr_matrix_weights, self._vectorizer_nlp \
      = p5_util.nlp_process(self.df_invoice_line\
      , 'Description' , vectorizer= self._vectorizer_nlp\
      , list_no_words=list_no_words, is_verbose= self.is_verbose)
            
      if df_invoice_line is None:
         self.strprint("***ERROR : NLP process interrupted!")
         return
         
            
      #-------------------------------------------------------------------------
      # NLP weights are cumulated  (sumerized) per customer
      #-------------------------------------------------------------------------
      if csr_matrix_weights is None:
         csr_matrix_weights \
         = p5_util.object_load('./data/matrix_weights_NLP.dump')
      else:
         pass
         
      self.strprint("df_invoice_line : "+str(df_invoice_line.shape))
      
      self.dbg_df = df_invoice_line.copy()
      
      root_name = 'w_nlp_'
      self._df_w_nlp = p5_util.df_nlp_sum_per_customer(df_invoice_line\
      , csr_matrix_weights, root_name)

      del(csr_matrix_weights)
      
      #-------------------------------------------------------------------------
      # Dimension reduction thanks to PCA
      #-------------------------------------------------------------------------      
      self.strprint("self._df_w_nlp : "+str(self._df_w_nlp.shape))

      root_name_pca = 'nlp_pca_'
      n_dim = self._nlp_pca_ndim
      
      df_customers_pca_nlp, self._pca_nlp \
      = p5_util.df_pca_reduce(self._df_w_nlp, n_dim, root_name_pca\
      , p_is_scale=False,  pca=self._pca_nlp)
      
      self.strprint("df_customers_pca_nlp : " +str(df_customers_pca_nlp.shape))

      #-------------------------------------------------------------------------
      # Backup of NLP features per customer
      #-------------------------------------------------------------------------
      if is_build_step is True:
         p5_util.object_dump(df_customers_pca_nlp\
         , self._df_customers_nlp_fileName)
      else:
         self._df_customers_pca_nlp = df_customers_pca_nlp.copy()
      
      return
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def data_transform(self, df) :
      """Proceed to data transformation in order to deliver computable 
      data model.
      Data trasformation process : 
         --> RFM is computed and encoded, resulting in a dataframe per customer.
         --> New features issued from InvoiceDate feature are created.
         --> New features issued from Description feature are created.
      Input : 
         * df : dataframe containing data to be transformed
      Output : none
      """

      #-------------------------------------------------------------------------
      # Copy of given dataframe to be transformed
      #-------------------------------------------------------------------------
      self.df_invoice_line = df
      
      #-------------------------------------------------------------------------
      # Features issued from InvoiceDate are created
      #-------------------------------------------------------------------------
      if self.is_transform_timeFeature is True:
         self.strprint("\n*** Time features transformation ***")
         self.data_transform_timeFeature()

      #-------------------------------------------------------------------------
      # RFM is computed and encoded
      #-------------------------------------------------------------------------
      if self.is_transform_rfm is True:
         self.strprint("\n*** RFM transformation ***")
         self.data_transform_rfm()

      #-------------------------------------------------------------------------
      # NLP features issued from Description are created
      #-------------------------------------------------------------------------
      if self.is_transform_nlp is True:
         self.strprint("\n*** NLP transformation ***")
         self.data_transform_nlp()
      
      return self.df_invoice_line
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def df_customers_features_build(self):
      """Build dataframe df_customers from transformed data.
      
      Transformed data are issued from NLP, Time and RFM features. 
      See data_transform().
      
      These data are stored as dataframes attributes.
      
      """

      df_customers_rfm = self._df_customers_rfm.copy()
      df_customers_timeFeature = self._df_customers_timeFeature.copy()
      df_customers_nlp = self._df_customers_pca_nlp.copy()

      #-------------------------------------------------------------------------
      # Dataframe are aggregated; note that indexes are customerID.
      #-------------------------------------------------------------------------
      df_customers = pd.DataFrame()

      df_customers = pd.concat([df_customers,df_customers_rfm],  axis=1)

      df_customers = pd.concat([df_customers,df_customers_timeFeature]\
      , join='inner', axis=1)

      df_customers = pd.concat([df_customers,df_customers_nlp]\
      , join='inner', axis=1)
      
      self.strprint("All features : "+str(df_customers.shape))
      self._df_customers = df_customers.copy()
      return
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def df_customers_fileRead(self):
      """Build dataframe df_customers from transformed data.
      
      Transformed data are loaded from dumped files issued from NLP, 
      Time and RFM features. See data_transform()
      
      """
         
      #-------------------------------------------------------------------------
      # RFM features are restored
      #-------------------------------------------------------------------------
      df_customers_rfm \
      = p5_util.object_load(self.df_customers_rfm_fileName)
      self.strprint("RFM features : "+str(df_customers_rfm.shape))
            
      #-------------------------------------------------------------------------
      # Time features are restored
      #-------------------------------------------------------------------------
      df_customers_timeFeature \
      = p5_util.object_load(self._df_customers_timeFeature_fileName)
      self.strprint("Time features : "+str(df_customers_timeFeature.shape))
      
      #-------------------------------------------------------------------------
      # NLP features are restored
      #-------------------------------------------------------------------------
      df_customers_nlp = p5_util.object_load(self._df_customers_nlp_fileName)
      self.strprint("NLP features : "+str(df_customers_nlp.shape))

      if False:
         df_customers_rfm = self._df_customers_rfm.copy()
         df_customers_timeFeature = self._df_customers_timeFeature.copy()
         df_customers_nlp = self._df_customers_pca_nlp.copy()

      #-------------------------------------------------------------------------
      # Dataframe are aggregated; note that indexes are customerID.
      #-------------------------------------------------------------------------
      df_customers = pd.DataFrame()

      df_customers = pd.concat([df_customers,df_customers_rfm],  axis=1)

      df_customers = pd.concat([df_customers,df_customers_timeFeature]\
      , join='inner', axis=1)

      df_customers = pd.concat([df_customers,df_customers_nlp]\
      , join='inner', axis=1)

      self.strprint("All features : "+str(df_customers.shape))

      #----------------------------------------------------------------------
      # Dataframe is dumped into a file
      #----------------------------------------------------------------------
      p5_util.object_dump(df_customers, self._df_customers_fileName)
      if False:
         #----------------------------------------------------------------------
         # Dataframe is copied as an attribute
         #----------------------------------------------------------------------
         self._df_customers = df_customers.copy()
      
      return
   #----------------------------------------------------------------------------
   
   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def createCustomerID(self):
      """ Creates a new customer identifier from existing data-set.

      Input : none
      Output : 
         * customerID : new customer identifier      
      """

      customerID = self._df_invoice_original.CustomerID.max()
      customerID += 1
      return int(customerID)
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def list_feature_drop(self):
      """Drop from df_invoice_line dataframe features in list given as parameter.
      All elements from list are checked to be into dataframe columns.
      Input : 
         * list_feature_to_drop : list of features to be droped from dataframe.

      Output : none         
      """
      
      list_to_drop = list()
      list_not_in_df = list()
      
      #-------------------------------------------------------------------------
      # Columns are checked to be into df_invoice_line dataframe
      #-------------------------------------------------------------------------
      for col in self._list_feature_to_drop:
         if col in self.df_invoice_line.columns:
            list_to_drop.append(col)
         else:
            list_not_in_df.append(col)
      
      if 0 == len(list_to_drop):
         self.strprint("\n*** ERROR : no element in list belonging to dataframe!")
      else:
         if len(self._list_feature_to_drop) != len(list_to_drop):
            self.strprint("\n*** WARNING : followings features do not belong to \
            dataframe : {}".format(list_not_in_df))
         else:
            pass
         list_col_keep \
         = [col for col in self.df_invoice_line.columns \
         if col not in list_to_drop]
         s
         self.df_invoice_line = self.df_invoice_line[list_col_keep]
      return
   #----------------------------------------------------------------------------
   
   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def feature_description_nlp(self):
      """Process df_invoice_line.Description with NLTK package.
      """
      
      #-------------------------------------------------------------------------
      # Returned dataframe is aggregated with weights from self.vectorizer
      #-------------------------------------------------------------------------
      list_no_words=['SET','PACK']
      self.df_invoice_line, vectorizer, matrix_weights \
      = p5_util.nlp_process(self.df_invoice_line,'Description'\
      , vectorizer=self.vectorizer, list_no_words=list_no_words)

      #-------------------------------------------------------------------------
      # Each vectorized column 'x' is renamed w_nlp_i
      #-------------------------------------------------------------------------
      dict_matching_name = dict()
      for col in self.df_invoice_line.columns:
         if str(col).isdigit() is True:
            new_col_name = "w_nlp_"+str(col)
            dict_matching_name[col] = new_col_name
         
      self.df_invoice_line.rename(columns=dict_matching_name,inplace=True)
      #-------------------------------------------------------------------------
      # Description is droped from columns
      #-------------------------------------------------------------------------
      del(self.df_invoice_line['Description'])
      
   #----------------------------------------------------------------------------
   #----------------------------------------------------------------------------
   def feature_scale(self):
      """Standardize quantitatives features.
      Standardizer is stored as object attribute.
      It will be copied into P5_SegmentClassifier object.
      
      Input : none
      Output : none
      """
   
      #-------------------------------------------------------------------------
      # List of quantitative features to be standardized
      #-------------------------------------------------------------------------
      list_quant_feature = ['Quantity','UnitPrice']
      self._list_quant_feature = list_quant_feature.copy()

      #-------------------------------------------------------------------------
      # Standardization is applied over quantitative features in list.
      #-------------------------------------------------------------------------
      X_std = self.std_scale.transform(self.df_invoice_line[self.list_quant_feature])
      df_quant_std = pd.DataFrame(X_std, index=self.df_invoice_line.index)
      
      #-------------------------------------------------------------------------
      # Columns from standardized dataframe are renamed
      #-------------------------------------------------------------------------
      df_quant_std.rename(columns={0:'STD_Quantity',1:'STD_UnitPrice'}\
      ,inplace=True)

      #-------------------------------------------------------------------------
      # Standardized values dataframe is aggregated to df_invoice_line
      #-------------------------------------------------------------------------
      list_col_drop = ['Quantity','UnitPrice']
      list_col_keep = \
      [col for col in self.df_invoice_line.columns if col not in list_col_drop ]
      self.df_invoice_line = self.df_invoice_line[list_col_keep]

      self.df_invoice_line \
      = pd.concat([self.df_invoice_line,df_quant_std], axis=1)
      
      return
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def get_customer_marketSegment(self, df_invoice_line_customer):
      """ Returns market segment ID related to a customer thanks to 
      customer invoices lines given as parameter.
      
      Features transformations are applied on data included into invoice lines.
      Once done, a machine learning algorithm is invocated in order to predict
      customer market segment.
      
      Input : 
         * df_invoice_line_customer : customer invoices lines.
      Output : 
         * segmentID : market segment customer belongs to
      """
      #-------------------------------------------------------------------------
      # Building data model 
      #-------------------------------------------------------------------------
      self.data_transform(df_invoice_line_customer)

      #-------------------------------------------------------------------------
      # Customer features are built thanks to transformers.
      #-------------------------------------------------------------------------
      self.df_customers_features_build()
      
      #-------------------------------------------------------------------------
      # Customer market segment is predicted
      #-------------------------------------------------------------------------
      X_test = self._df_customers.values
      y_pred = self._classifier_model.predict(X_test)
      segmentID = y_pred[0]
      
      return segmentID
      
   #----------------------------------------------------------------------------
   
   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def order_process(self, customerID, list_stockCode, list_quantity\
   , orderDate=None):
      """ This function creates an invoice compounding invoices lines from 
      data given as parameters.

      Once done, this function computes market segment customer belongs to.
      
      If customerID is None, then a new customer identifier is created 
      before order process to take place.
      
      Input : 
         * customerID : customer identifier
         * list_stockCode : list of items identified with stock code.
         * list_quantity : list of items quantities.
         * orderDate : date of order. This date will apply to invoice.
      Output : 
         * segmentID : market segment customer has been predicted to belongs to.
         * customerID : customer identifier.
      
      """

      segmentID = -1

      #-------------------------------------------------------------------------
      # A new customer is created and inserted into data-set.
      #-------------------------------------------------------------------------
      if customerID is None:
         customerID = int(self.createCustomerID())
      else:
         pass
      
      #-------------------------------------------------------------------------
      # A new dataframe with new invoice lines are created.
      #-------------------------------------------------------------------------
      df_invoice_line = self.create_customer_df_invoice_line(customerID\
      , list_stockCode, list_quantity, orderDate)
      
      #-------------------------------------------------------------------------
      # Original dataframe is updated with customer invoices lines.
      #-------------------------------------------------------------------------
      print("order_process : shape before concat= "+str(self._df_invoice_original.shape))
      self._df_invoice_original \
      = pd.concat([self._df_invoice_original, df_invoice_line], axis=0)
      print("order_process : shape after concat= "+str(self._df_invoice_original.shape))
      
      #-------------------------------------------------------------------------
      # All invoices lines (including new one) related to customer is retrieved 
      # from original dataframe.
      #-------------------------------------------------------------------------
      df_invoice_line_customer \
      = self.get_customer_history_df_invoice_line(customerID)

      #-------------------------------------------------------------------------
      # When calling get_customer_marketSegment(), df_invoice_line_customer is
      # concatened to the original dataframe.
      #-------------------------------------------------------------------------      
      segmentID = self.get_customer_marketSegment(df_invoice_line_customer)
      
      return segmentID, customerID
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def data_process_deprecated(self, CustomerID, InvoiceDate, InvoiceNo, Description, Quantity\
   , UnitPrice ):
      """ Performs data processing in order data to feed prediction algorithm.
      Input : 
         * CustomerID 
         * InvoiceDate
         * InvoiceNo
         * Description
         * Quantity
         * UnitPrice
      """
      dict_invoice = {'InvoiceDate':InvoiceDate, 'Description':Description\
      , 'Quantity':Quantity, 'UnitPrice':UnitPrice}
      dict_invoice['CustomerID'] = CustomerID
      dict_invoice['InvoiceNo'] = InvoiceNo
      df_invoice_line \
      = pd.DataFrame(dict_invoice, columns=dict_invoice.keys(), index=[0])
   
      self.data_transform(df_invoice_line)

      #self.feature_rfm_encode()

      self.feature_scale()

      self.list_feature_drop()

      self.feature_description_nlp()
      return
   #----------------------------------------------------------------------------
   
   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def predict_segment(self, df_invoice_line=None):
      """ Return the segment identifier a customers is predicted to belongs to.
      Input : 
      
         * df_invoice_line : dataframe with at least one transaction performed 
         by customer. Dataframe format is identical to
      Output:
         * segment identifier value.
      """
      if df_invoice_line is not None:
         self.data_transform(df_invoice_line)    
         self.df_customers_features_build()     
      else:
         pass
      X_test = self._df_customers.values
      y_pred = self._classifier_model.predict(X_test)
      return y_pred[0]
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def getStockCodeList(self, list_description=None):
      """Returns list of stock codes from list of items descriptions.
      Input : 
         * list_description : list of items descriptions. If value is None, 
         then list of all stock codes is returned.
         
      Output : 
         * list_stockCode : list of stock codes issued from list of items.
      """
      list_stockCode = list()
      df = self._df_invoice_original
      
      if list_description is None:
         list_stockCode = list(df.StockCode.unique())
      else :
         for description in list_description:
            stockCode = df[df.Description==description].StockCode.unique()[0]
            list_stockCode.append(stockCode)
      return list_stockCode
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def getUnitPriceList(self, list_stockCode):
      """Returns list of imtes unit price from list of stock codes.
      Input : 
         * list_stockCode :list of stock codes.
      Output : 
         * list_unitPrice : list of unit prices issued from list of items.
      """
      df = self._df_invoice_original

      list_unitPrice = list()
      
      for stockCode in list_stockCode:
         unitPrice = df[df.StockCode==stockCode].UnitPrice.unique()[0]
         list_unitPrice.append(unitPrice)
      return list_unitPrice
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def getDescriptionList(self, list_stockCode=None):
      """Returns list of items descriptions from list of stock codes.
      Input : 
         * list_stockCode :list of stock codes. If value is None, then all
         descriptions from original data-set are returned.
      Output : 
         * list_description : items descriptions from given stock codes list
      """
      df = self._df_invoice_original

      list_description = list()
      if list_stockCode is None :
         list_description = list(df.Description.unique())
      else:
         for stockCode in list_stockCode:
            description = df[df.StockCode==stockCode].Description.unique()[0]
            list_description.append(description)
   
      return list_description
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def create_customer_df_invoice_line(self, customerID, list_stockCode\
   , list_quantity, invoiceDate):
      """Creates new dataframe with invoices lines issued from given 
      parameters.
      
      Once done, the new dataframe is aggregated with original one.

      Input :
         * customerID : customer identfier
         * list_stockCode : list of stock code for new invoices lines.
         * list_quantity : list of items quantities for new invoices lines.
         * invoiceDate : date on which invoice is supposed to be created.
         If None, then current date is used.
      Output :
         * df_invoice_line : dataframe containing new invoices lines.
      """
      
      dict_invoice = dict()

      dict_invoice['Quantity'] = list_quantity
      dict_invoice['StockCode'] = list_stockCode

      #------------------------------------------------------------------------
      # Build invoiceDate from local current time
      #------------------------------------------------------------------------
      if invoiceDate is None:
         time_struct = time.localtime()
         invoiceDate = str(time_struct.tm_year)+'-'+str(time_struct.tm_mon)\
         +'-'+str(time_struct.tm_mday)
         invoiceDate +=' '
         invoiceDate +=str(time_struct.tm_hour)+':'+str(time_struct.tm_min)\
         +':'+str(time_struct.tm_sec)
         invoiceDate = pd.Timestamp(invoiceDate)
      else:
         pass


      #------------------------------------------------------------------------
      # Lists initialization
      #------------------------------------------------------------------------
      list_customerID = list()
      list_invoiceNo  = list()
      list_invoiceDate = list()
      list_invoice_line_index = list()
      
      #------------------------------------------------------------------------
      # Increase Invoice number
      #------------------------------------------------------------------------
      invoiceNo = max(self._df_invoice_original.InvoiceNo)
      invoiceNo += 1

      #------------------------------------------------------------------------
      # Get latest invoice line index value
      #------------------------------------------------------------------------
      invoice_line_index = max(self._df_invoice_original.index)

      #------------------------------------------------------------------------
      # Build lists for CustomerID, InvoiceNo, InvoiceDate
      # A list of incremented indexes is built for new rows.
      #------------------------------------------------------------------------
      for quantity in list_quantity:
         list_customerID.append(customerID)
         list_invoiceNo.append(invoiceNo)
         list_invoiceDate.append(invoiceDate)
         invoice_line_index += 1
         list_invoice_line_index.append(invoice_line_index)   

      
      dict_invoice['CustomerID'] = list_customerID
      dict_invoice['InvoiceNo'] = list_invoiceNo
      dict_invoice['InvoiceDate'] = list_invoiceDate

      #------------------------------------------------------------------------
      # Get description list from list of stock codes.
      #------------------------------------------------------------------------
      list_description = self.getDescriptionList(list_stockCode)
      
      dict_invoice['Description'] = list_description

      #------------------------------------------------------------------------
      # Get unit price list from list of stock codes.
      #------------------------------------------------------------------------
      list_unitPrice = self.getUnitPriceList(list_stockCode)
      
      dict_invoice['UnitPrice'] = list_unitPrice

      #------------------------------------------------------------------------
      # Dataframe with new invoices lines is created.
      #------------------------------------------------------------------------
      df_invoice_line \
      = pd.DataFrame(dict_invoice, columns=dict_invoice.keys()\
      , index=list_invoice_line_index)
      
      return df_invoice_line
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def get_customer_history_df_invoice_line(self, customerID):
       """Returns a dataframe with all invoice lines from customerID 
       given as parameter.
       Input : 
           * customerID : customer Identifier
       Output :
           * dataframe with all invoice lines belonging to customerID
       """
       df_invoice_line \
       = self._df_invoice_original[self._df_invoice_original.CustomerID \
       == customerID]
       return df_invoice_line
   #----------------------------------------------------------------------------   
   
   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------   
   def get_listCustomer_out_sample(self, customerCount=10):
      """ Returns a list of customers that have been excluded of data sampling 
      used for building model.
      
      By default, 10 customers identifier is returned.
      If customerCount value is None, or <= 0, then list of all customers that 
      have been excluded of data sampling is returned.
      Input :
         * customerCount : number of customers identifier to be returned.
         

      Output :
         * list of customer identifiers.
      """
      
      if customerCount is None :
         listCustomer= list(self._df_invoice_line_out_sample.CustomerID.unique())
      else:
         if customerCount <= 0 :
            listCustomer \
            = list(self._df_invoice_line_out_sample.CustomerID.unique())
         else:
            listCustomer \
            = list(self._df_invoice_line_out_sample.CustomerID.unique()[:customerCount])
      return listCustomer
   #----------------------------------------------------------------------------   

   #----------------------------------------------------------------------------   
   #
   #----------------------------------------------------------------------------   
   def is_customer_out_sample(self, customerID):
      """Returns a True if a customer identifier does not belongs to dataframe
      used to build classifier model.
      """
      listCustomer = list(self._df_invoice_line_out_sample.CustomerID.unique())
      is_flag = customerID in listCustomer
      return is_flag
   #----------------------------------------------------------------------------   

   #----------------------------------------------------------------------------   
   #
   #----------------------------------------------------------------------------   
   def get_invoice_count(self):
      """Returns number of invoices from original data-set.
      """
      return self._df_invoice_original.InvoiceNo.unique().shape[0]
   #----------------------------------------------------------------------------   

   #----------------------------------------------------------------------------   
   #
   #----------------------------------------------------------------------------   
   def get_customer_count(self):
      """Returns number of customers from original data-set.
      """
      return self._df_invoice_original.CustomerID.unique().shape[0]
   #----------------------------------------------------------------------------   

   #----------------------------------------------------------------------------   
   #
   #----------------------------------------------------------------------------   
   def get_invl_count(self):
      """Returns number of invoice lines (number of rows) from original data-set.
      """
      return self._df_invoice_original.index.unique().shape[0]
   #----------------------------------------------------------------------------   

   #----------------------------------------------------------------------------   
   #
   #----------------------------------------------------------------------------   
   def json_all_builder(self, customer_count, invoice_count, invl_count ):
      """Returns a json sructure built from given parameters.
      
       {
       "_results": [
         { "customer_count": "645657", "invoice_count": "1234", "invl_count": "10}"
         ]
       }      
       
       
       
      """
      json_result = '{\n'
      json_result += '\t "_results":[\n'
      json_result += '\t\t{ "customer_count": "' + str(customer_count)
      json_result += ', "invoice_count": "' + str(invoice_count)
      json_result += ', "invl_count": "' + str(invl_count)
      json_result += '}\n'
      json_result += '\n\t]\n}'
      return json_result      
   #----------------------------------------------------------------------------   
   
   #----------------------------------------------------------------------------   
   #
   #----------------------------------------------------------------------------   
   def json_df_builder(self, df, marketID, RFM=None):
      """Returns JSON structure issued form dataframe content given as 
      parameter .
      
      JSON structure is formated as following : 
       {
       "_results": [
         { 
            "customerID": "12822"
          , "marketID": "2"
          , "invoice_count": "10"
          , "item_count": "100"
          , "invl_count": "1000"
          , "mean_unit_price": "13.9"
          , "incomes": "1345.45"
          , "old_date" : "2010-03-01 10:51:00"
          , "new_date" : "2011-12-01 02:35:21"
         }"
         ]
       }      
         
      """
      
      #-------------------------------------------------------------------------
      # Extract from dataframe content to be returned
      #-------------------------------------------------------------------------
      str_customerID = str(df.CustomerID.unique()[0])
      
      invoice_count = len(df.InvoiceNo.unique())
      item_count = df.Quantity.sum()
      invl_count = df.shape[0]
      
      ser_incomes = df.UnitPrice * df.Quantity
      incomes = ser_incomes.sum()
      str_incomes = "{0:1.2F}".format(incomes)
      
      mean_unit_price = incomes/item_count
      str_mean_unit_price = "{0:1.2F}".format(mean_unit_price)
      
      serInvoiceDate = df.InvoiceDate
      str_old_date = serInvoiceDate.map(str).min()
      str_new_date = serInvoiceDate.map(str).max()
      
      #-------------------------------------------------------------------------
      # Build JSON structure form content
      #-------------------------------------------------------------------------
      json_result = '{\n'
      json_result += '\t "_results":[\n'
      json_result += "{\n"
      json_result += "\t\t"+" \"customerID\":"+str_customerID+"\n"
      json_result += "\t\t"+",\"marketID\":"+str(marketID)+"\n"
      json_result += "\t\t"+",\"invoice_count\":"+str(invoice_count)+"\n"
      json_result += "\t\t"+",\"item_count\":"+str(item_count)+"\n"
      json_result += "\t\t"+",\"invl_count\":"+str(invl_count)+"\n"
      json_result += "\t\t"+",\"mean_unit_price\":"+str_mean_unit_price+"\n"
      json_result += "\t\t"+",\"incomes\":"+str_incomes+"\n"
      json_result += "\t\t"+",\"old_date\":"+str_old_date+"\n"
      json_result += "\t\t"+",\"new_date\":"+str_new_date+"\n"
      
      if RFM is not None:
         json_result += "\t\t"+",\"RFM\":"+RFM+"\n"
      else:
         pass

      json_result += "}\n"
      json_result += '\n\t]\n}'
      return json_result
   #----------------------------------------------------------------------------   

   #----------------------------------------------------------------------------   
   #
   #----------------------------------------------------------------------------   
   def json_market_builder(self, customerID, marketID) :
      """ Returns a market segment ID into a JSON structure.
      
      JSON structure is formated as following : 
       {
       "_results": [
         { "customerID": "645657", "marketID": "0"}"
         ]
       }      
      
      """
      json_result = '{\n'
      json_result += '\t "_results":[\n'
      json_result += '\t\t{ "customerID": "' + str(customerID)
      json_result += ', "marketID": "' + str(marketID)
      json_result += '}\n'
      json_result += '\n\t]\n}'
      return json_result      
      
   #----------------------------------------------------------------------------   

   #----------------------------------------------------------------------------   
   #
   #----------------------------------------------------------------------------   
   def json_error(self, errorMessage):
      json_result = '{\n'
      json_result += '\t "_results":[\n'
      json_result += '\t\t{ "ERROR": "' + str(errorMessage)
      json_result += '}\n'
      json_result += '\n\t]\n}'
      return json_result      
   
   #----------------------------------------------------------------------------   
   
   #----------------------------------------------------------------------------   
   #
   #----------------------------------------------------------------------------   
   def get_rfm(self, df):
      """Returns RFM score from dataframe given from parameter.
      RFM score is computed from local RFM matrix threshold.
      """
      df_tmp, df_RFM, df_RFM_threshold, day_now \
      = p5_util.p5_df_rfm_build(df, df_RFM_threshold=self.df_RFM_quantiles
                               ,day_now = self._day_now)
      RFM = df_RFM.RFM.iloc[0]
      return str(RFM)
   #----------------------------------------------------------------------------   

   #----------------------------------------------------------------------------   
   #
   #----------------------------------------------------------------------------   
   def get_order_lists(self, n_items, n_quantities):
      """This function is used for validation process.
      It returns a list of stockCode items and a list of quantities for 
      each item.
      """
      arr_stock_code = self._df_invoice_original.StockCode.unique()
      arr_stock_code = np.random.choice(arr_stock_code, n_items)
      list_stockCode = list(arr_stock_code)
      list_quantities = np.ones(arr_stock_code.shape[0])
      list_quantities *=n_quantities

      return list_stockCode, list_quantities
   #----------------------------------------------------------------------------   


#-------------------------------------------------------------------------------



