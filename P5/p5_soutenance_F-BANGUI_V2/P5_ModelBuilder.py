import pandas as pd
import numpy as np
import time
from scipy import sparse

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


import p3_util
import p5_util

import P5_SegmentClassifier

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
class P5_ModelBuilder() :
   ''' 
   This class aims to build a model for customer segmentation as per 
   P5 project description from Openclassrooms course of Data-Scientist training.
    
   Services provided by this class are :
      -> It acquires data from CSV file.
      -> It proceeds to data preparation : data cleaning and scaling.
      -> It builds data to be feed into a computable model.
      -> It builds computational models 
      -> It builds a deployable component, oP5_SegmentClassifier, 
      in charge of customer segmentation. Such class aims to be deployed for 
      production.
      
   Instructions flow :
   -------------------
      +-->__init__()
      |
      +-->data_load(fileName)
      |
      +-->data_clean(fileName)
      |
      +-->data_sampling(fileName)
      |
      +-->data_transform()
      |   |
      |   +-->data_transform_rfm()
      |   |  |
      |   |  +-->p5_util.p5_df_rfm_build()
      |   |  |
      |   |  +-->p5_util.df_rfm_one_hot_encode()
      |   |  |
      |   |  +-->p5_util.df_rename_columns()
      |   |
      |   +-->data_transform_timeFeature()
      |   |  |
      |   |  +-->p5_util.time_list_feature_build()
      |   |  |   |
      |   |  |   +-->p5_util.time_feature_encoded_new()
      |   |  |
      |   |  +-->p5_util.time_list_feature_restore()
      |   |  |
      |   |  +-->p5_util.df_pca_reduce()
      |   |
      |   +-->data_transform_nlp()
      |   |  |
      |   |  +-->p5_util.nlp_process()
      |   |  |
      |   |  +-->p5_util.df_rename_columns()
      |   |  |
      |   |  +-->df_nlp_sum_per_customer()
      |   |  |
      |   |  +-->p5_util.df_pca_reduce()
      |
      +-->clusters_build()
      |   |
      |   +-->df_customers_fileRead()
      |
      +-->classifier_build()
      |
      +-->model_dump()
   '''

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def __init__(self, path_to_data=None, n_clusters=6) :
      self._path_to_data = path_to_data
      self._df_invoice = pd.DataFrame()
      self._df_invoice_original = pd.DataFrame()
      self._df_invoice_sample = pd.DataFrame()
      self._df_invoice_line_out_sample = pd.DataFrame()
      self._country_list = ['United Kingdom']
      
      self._rfm_encoder = None
      self._vectorizer = None
      self._oP5_SegmentClassifier = None
      
      self._n_clusters = n_clusters
      self._y_clusters = None
      self._classifier_name = "RandomForests"
      self._dict_classifier_param = {'RandomForests':{'nb_forests':13}}
      self._classifier_model = None
      self._is_data_sampling = True
      self._std_scale = None
      self.list_quant_feature = list()
      self._is_rfm_encode = False
      self._nb_customers = 0
      self._nb_invoices = 0
      self._cluster_model_name = 'GMM'
      self._dict_cluster_model \
      = {'GMM':{'n_clusters':6, 'covariance_type':'diag'}\
      ,  'KMEANS':{'n_clusters':6}}
      self._df_customers_fileName = './data/df_customers.dump'

   #----------------------------------------------------------------------------
   
   #----------------------------------------------------------------------------
   # Properties
   #----------------------------------------------------------------------------
   def _get_path_to_data(self) :
      return self._path_to_data
   def _set_path_to_data(self, path_to_data) :
      self._path_to_data = path_to_data

   def _get_df_invoice(self) :
      return self._df_invoice
   def _set_df_invoice(self, df_invoice) :
      if df_invoice is not None:
         self._df_invoice = df_invoice.copy()
         del(df_invoice)
   
   def _get_df_invoice_sample(self) :
      return self._df_invoice_sample.copy()
   def _set_df_invoice_sample(self, df_invoice_sample) :
      self._df_invoice_sample = df_invoice_sample.copy()
      del(df_invoice_sample)
            
   def _get_classifier_name(self) :
      return self._classifier_name
   def _set_classifier_name(self, classifier_name) :
      self._classifier_name = classifier_name

   def _get_dict_classifier_param(self) :
      return self._dict_classifier_param
   def _set_dict_classifier_param(self, dict_classifier_param) :
      self._dict_classifier_param = dict_classifier_param.copy()

   def _get_is_data_sampling(self):
      return self._is_data_sampling
   def _set_is_data_sampling(self, is_data_sampling):
      self._is_data_sampling = is_data_sampling

   def _get_std_scale(self):
      return self._std_scale
   def _set_std_scale(self, std_scale):
      self._std_scale = std_scale

   def _get_list_quant_feature(self):
      return self._list_quant_feature
   def _set_list_quant_feature(self, list_quant_feature):
      self._list_quant_feature = list_quant_feature.copy()
      
   #----------------------------------------------------------------------------
   path_to_data = property(_get_path_to_data, _set_path_to_data)
   df_invoice = property(_get_df_invoice, _set_df_invoice)
   df_invoice_sample = property(_get_df_invoice_sample, _set_df_invoice_sample)
   classifier_name = property(_get_classifier_name, _set_classifier_name)
   dict_classifier_param = property(_get_dict_classifier_param\
   , _set_dict_classifier_param)
   is_data_sampling = property(_get_is_data_sampling, _set_is_data_sampling )
   std_scale = property(_get_std_scale, _set_std_scale)
   list_quant_feature = property(_get_list_quant_feature\
   , _set_list_quant_feature)
   #----------------------------------------------------------------------------
   


   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def print(self):
      print("Data file name : ...................... :"+str(self.path_to_data))

      if self._df_invoice is not None :
         print("Working data-frame infos ........... :"\
         +str(self._df_invoice.shape))
      else :
         print("Working data-frame infos ........... :"+self._df_invoice.shape)

      if self._df_invoice_original is not None :
         print("Original Data-frame infos .......... :"\
         +str(self._df_invoice_original.shape))
      else :
         print("Original Data-frame infos .......... :"\
         +self._df_invoice_original.shape)

      print("Out-of-sample data-frame infos ........ :"\
      +str(self._df_invoice_line_out_sample.shape))

      print("Number of customers.................... :"+str(self._nb_customers))
      print("Number of invoices .................... :"+str(self._nb_invoices))
      
      print("Data sampling tag ..................... :"\
      +str(self.is_data_sampling))
      print("Quantitative features.................. :{}"\
      .format(self._list_quant_feature))
      print("Standard scaler........................ :{}"\
      .format(self.std_scale))
      print("RFM encoding flag ..................... :"\
      +str(self._is_rfm_encode))

      print("RFM encoder ........................... :"+str(self._rfm_encoder))
      print("NLP Vectorizer ........................ :"+str(self._vectorizer))
      print("Cluster model name .................... :"\
      +str(self._cluster_model_name))
      print("Clusters hyper-parameters ............. :"\
      +str(self._dict_cluster_model))
      
      if self._y_clusters is not None:
         print("Clusters array......................... :"\
         +str(np.unique(self._y_clusters)))
      else:
         print("Clusters vector........................ :"\
         +str(self._y_clusters))

      print("Classifier ............................ :"+self.classifier_name)
      print("Classifier parameters ................. :"\
      +str(self._dict_classifier_param))
      
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def copy(self, other_object):
      """Copy my_object data into self"""
      self.path_to_data = other_object.path_to_data
      self.df_invoice = other_object.df_invoice
      self.df_invoice_sample = other_object.df_invoice_sample
      self._rfm_encoder = other_object._rfm_encoder
      self._vectorizer = other_object._vectorizer 
      self._oP5_SegmentClassifier = other_object._oP5_SegmentClassifier
      if other_object._y_clusters is not None:
         self._y_clusters = other_object._y_clusters.copy()
      else:
         self._y_clusters = None
      self.classifier_name = other_object.classifier_name
      self.dict_classifier_param = other_object.dict_classifier_param
      self._classifier_model = other_object._classifier_model
      self.is_data_sampling = other_object.is_data_sampling
      self.list_quant_feature = other_object.list_quant_feature
      self._is_rfm_encode = other_object._is_rfm_encode
      self._nb_customers = other_object._nb_customers
      self._nb_invoices = other_object._nb_invoices
      self._cluster_model_name = other_object._cluster_model_name
      self._dict_cluster_model = other_object._dict_cluster_model
      self._df_invoice_original = other_object._df_invoice_original.copy()
      self._df_invoice_line_out_sample \
      = other_object._df_invoice_line_out_sample.copy()
   #----------------------------------------------------------------------------


   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def data_read(self) :
      if self.path_to_data is None :
         print("\n***ERROR : no file name for loading data!")
         return
      t0 = time.time()
      self._df_invoice = pd.read_excel(self.path_to_data)
      t1 = time.time()
      print("Elapsed time = %1.2f" %(t1-t0))

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
      if 'Country' not in self.df_invoice.columns:
         return

      list_countries_keep = ['United Kingdom']
      rows_before = self.df_invoice.shape[0]
      
      df_invoice_new = pd.DataFrame()
      for country in list_countries_keep : 
         df_invoice_new = df_invoice_new.append(\
         self._df_invoice[self.df_invoice['Country']==country]\
         , ignore_index=True)

      self.df_invoice = df_invoice_new
      del(df_invoice_new)
      
      rows_after = self._df_invoice.shape[0]      
      P5_SegmentClassifier.print_stat_rows("Countries filtering : "\
      , rows_before, rows_after)

      
      #-------------------------------------------------------------------------
      # Due to the fact only one country is used, then this feature is dropped
      #-------------------------------------------------------------------------
      list_col_to_keep = [col for col in self._df_invoice.columns \
      if col not in 'Country']
      
      self._df_invoice = self._df_invoice[list_col_to_keep]      

      return
   #----------------------------------------------------------------------------



   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def data_clean(self) :
      """Cleand data in a such way : 
      * Rows with CustomerID values <=0 are dropped.
      * Rows with Quantity values <=0 are dropped.
      * Rows with UnitPrice values <=0 are dropped.
      * All countries except UK are filtered
      Input : none
      Output : 
         * dataframe issued from this process is backup into athe dumped file :
         ./data/df_invoice_line_clean
      """
      
      if self._df_invoice is None:
         print("\n***ERROR : no data loaded!")
         return
      #-------------------------------------------------------------------------
      # Drop rows with Nan values
      #-------------------------------------------------------------------------
      rows_before = self._df_invoice.shape[0]      
      self._df_invoice.dropna(axis=0, how='any', inplace=True)      
      rows_after = self._df_invoice.shape[0]
      P5_SegmentClassifier.print_stat_rows("NUll values : "\
      ,rows_before, rows_after)      
      
      #-------------------------------------------------------------------------
      # Drop rows feature 'Quantity' <=0
      #-------------------------------------------------------------------------
      rows_before = rows_after
      self._df_invoice = self._df_invoice[(self._df_invoice['Quantity']>0)]
      rows_after = self._df_invoice.shape[0]
      P5_SegmentClassifier.print_stat_rows("Quantity <= 0 : "\
      ,rows_before, rows_after)
      
      #-------------------------------------------------------------------------
      # Drop rows feature 'CustomerID' <=0
      #-------------------------------------------------------------------------
      rows_before = rows_after
      self._df_invoice = self._df_invoice[self._df_invoice['CustomerID'] > 0]
      rows_after = self._df_invoice.shape[0]
      P5_SegmentClassifier.print_stat_rows("CustomerID <= 0 : "\
      ,rows_before, rows_after)
      
      #-------------------------------------------------------------------------
      # Drop rows feature 'UnitPrice' <=0
      #-------------------------------------------------------------------------
      rows_before = rows_after
      self._df_invoice = self._df_invoice[self._df_invoice['UnitPrice']>0]
      rows_after = self._df_invoice.shape[0]
      P5_SegmentClassifier.print_stat_rows("UnitPrice <= 0 : "\
      ,rows_before, rows_after)

      #-------------------------------------------------------------------------
      # Removing duplicated rows
      #-------------------------------------------------------------------------
      list_col_dupl \
      = ['InvoiceNo','StockCode','InvoiceDate','Quantity','CustomerID']

      self._df_invoice.drop_duplicates(subset=list_col_dupl\
      , keep='first', inplace=True)

      #-------------------------------------------------------------------------
      # Countries are filtered
      #-------------------------------------------------------------------------
      self._feature_country_process()

      #-------------------------------------------------------------------------
      # Index is reset
      #-------------------------------------------------------------------------
      self._df_invoice = self._df_invoice.reset_index()
      del(self._df_invoice['index'])

      
      #-------------------------------------------------------------------------
      # CustomerID is converted as integer
      #-------------------------------------------------------------------------
      self._df_invoice.CustomerID = self._df_invoice.CustomerID.astype(int)

      #-------------------------------------------------------------------------
      # Number of customers, number of invoices
      #-------------------------------------------------------------------------
      self._compute_numbers()

      #-------------------------------------------------------------------------
      # Once cleaned, dataframe is backup
      #-------------------------------------------------------------------------
      file_name = './data/df_invoice_line_clean.dump'
      self.data_dump(file_name)
      return
   #----------------------------------------------------------------------------


   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def data_dump(self,file_name):
      """Dump data from df_invoice into a file which name is given as parameter. 
      Input : 
         * file_name : name of the file that handles dumped data
      output : none
      """
      if file_name is None:
         print("\n*** ERROR : no file name for dumping data!")
         return
      if 0 == len(file_name):
         print("\n*** ERROR : undefined file name for dumping data!")
         return
      p5_util.object_dump(self.df_invoice,file_name)
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def _compute_numbers(self):
      """Compute : 
         --> numbers of customers
         --> number of invoices
      Input : none
      Output : none
      """
      #-------------------------------------------------------------------------
      # Number of customers
      #-------------------------------------------------------------------------
      self._nb_customers = len(self._df_invoice.CustomerID.unique())

      #-------------------------------------------------------------------------
      # Number of invoices
      #-------------------------------------------------------------------------
      self._nb_invoices = len(self._df_invoice.InvoiceNo.unique())
      
      return
   #----------------------------------------------------------------------------
   
   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def data_load(self,file_name):
      """Load df_invoice data from dumped file which name is given as parameter. 
      Input : 
         * file_name : name of the file that handles dumped data
      output : none
      """
      t0 = time.time()
      if file_name is None:
         print("\n*** ERROR : no file name for loading data!")
         return
      if 0 == len(file_name):
         print("\n*** ERROR : undefined file name for loading data!")
         return

      self.df_invoice = p5_util.object_load(file_name)

      t1 = time.time()

      print("Elapsed time = %1.2f" %(t1-t0))

      self._compute_numbers()

      return
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def data_transform(self) :
      """Proceed to data transformation in order to deliver computable 
      data model.
      Input : none
      Output : none
      """
      self._oP5_SegmentClassifier = P5_SegmentClassifier.P5_SegmentClassifier()
      self.df_invoice = \
      self._oP5_SegmentClassifier.data_transform(self.df_invoice)
      
   #----------------------------------------------------------------------------
   
   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def data_sampling(self, ratio=0.1, select='random'):
      """Build a dataframe sample from the original dataframe.

      Sampling operation applies on CustomerID values.
      Original dataframe is backup.
      Out of sample data is dumped into a separate file. Data issued from this
      dataframe will be used into validation process.
      
      All operations for building data-model will apply on this sample.

      Input : 
         * ratio : percent of rows in the sampled dataframe
         * select : mode of selection for sampling.
         
      Output : none
                  
      """
      self._df_invoice_original = self._df_invoice.copy()
      self._df_invoice \
      = p5_util.df_sampling(self._df_invoice_original, 'CustomerID'\
      , ratio, mode=select)
      
      #-------------------------------------------------------------------------
      # Compute statistics about number of customers, number of invoice
      #-------------------------------------------------------------------------
      self._compute_numbers()
      
      # TBD : the following sequences has not been tested.

      #-------------------------------------------------------------------------
      # Get customerID from sampling
      #-------------------------------------------------------------------------
      arr_sample_customerID = np.unique(self._df_invoice.CustomerID.values)

      #-------------------------------------------------------------------------
      # Query original dataframe with customerID out of sample
      #-------------------------------------------------------------------------
      df_query = "CustomerID not in "+str(list(arr_sample_customerID))
      df_invoice_line_out_sample = self._df_invoice_original.query(df_query)

      #-------------------------------------------------------------------------
      # Dump out of sample customerID dataframe
      #-------------------------------------------------------------------------
      file_name = './data/df_invoice_line_out_sample_random.dump'
      p5_util.object_dump(df_invoice_line_out_sample, file_name)

   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def data_sampling_load(self, mode='random'):
      """Load sampled dataframe dumped into a file.
      Input : 
         * ratio : percent of rows in the sampled dataframe
         * select : mode of selection for sampling.
         
      Output : none
                  
      """
            
      if mode == 'random' :
         file_name = './data/df_invoice_line_sample_random.dump'
      else :
         print("*** ERROR : mode = "+str(mode)+ " is not supported!" )
         return

      self._df_invoice = p5_util.object_load(file_name)


      #-------------------------------------------------------------------------
      # Countries are filtered
      #-------------------------------------------------------------------------
      if 'Country' in self._df_invoice.columns:
         self.df_invoice \
         = self.df_invoice[self.df_invoice.Country == 'United Kingdom']
      self._df_invoice_original = self._df_invoice.copy()      

      #-------------------------------------------------------------------------
      # Compute statistics about number of customers, number of invoice
      #-------------------------------------------------------------------------
      self._compute_numbers()
      
      #-------------------------------------------------------------------------
      # Dataframe with out of sample customers is relaoded.
      #-------------------------------------------------------------------------
      file_name = './data/df_invoice_line_out_sample_random.dump'
      self._df_invoice_line_out_sample = p5_util.object_load(file_name)
      
   #----------------------------------------------------------------------------



   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def data_sampling_deprecated(self, ratio=0.1, select='random'):
      """Get a sample from the original dataframe.
      Sampling operation applies on CustomerID values.
      Original dataframe is backup.
      All operations for building data-model will apply on this sample.

      Input : 
         * ratio : percent of rows in the sampled dataframe
         * select : mode of selection for sampling.
         
      Output : none
                  
      """
      self._df_invoice_original = self._df_invoice.copy()
      self._df_invoice \
      = p5_util.df_sampling(self.df_invoice, 'CustomerID', ratio, mode=select)
      if self.df_invoice is not None :      
         self._compute_numbers()
      else:
         print("*** ERROR : p5_util.df_sampling() returned None for sample!")

      return
   #----------------------------------------------------------------------------


   #----------------------------------------------------------------------------
   #
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
      self.std_scale, X_quantitative_std = \
      p5_util.df_features_standardize(self.df_invoice, list_quant_feature)


      df_quant_std = pd.DataFrame(X_quantitative_std\
      , index=self.df_invoice.index)
      
      #-------------------------------------------------------------------------
      # Columns from standardized dataframe are renamed
      #-------------------------------------------------------------------------
      df_quant_std.rename(columns={0:'Quantity',1:'UnitPrice'},inplace=True)

      #-------------------------------------------------------------------------
      # Standardized values dataframe is aggregated to df_invoice
      #-------------------------------------------------------------------------
      list_col_drop = ['Quantity','UnitPrice']
      list_col_keep = \
      [col for col in self.df_invoice.columns if col not in list_col_drop ]
      self.df_invoice = self.df_invoice[list_col_keep]

      self.df_invoice = pd.concat([self.df_invoice,df_quant_std], axis=1)
      
      return
   #----------------------------------------------------------------------------
   
   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def list_feature_drop(self, list_feature_to_drop):
      """Drop from df_invoice dataframe features in list given as parameter.
      All elements from list are checked to be into dataframe columns.
      Input : 
         * list_feature_to_drop : list of features to be droped from dataframe.

      Output : none         
      """
      
      list_to_drop = list()
      list_not_in_df = list()
      #-------------------------------------------------------------------------
      # Columns are checked to be into df_invoice dataframe
      #-------------------------------------------------------------------------
      for col in list_feature_to_drop:
         if col in self.df_invoice.columns:
            list_to_drop.append(col)
         else:
            list_not_in_df.append(col)
      
      if 0 == len(list_to_drop):
         print("\n*** ERROR : no element in list belonging to dataframe!")
      else:
         if len(list_feature_to_drop) != len(list_to_drop):
            print("\n*** WARNING : followings features do not belong to \
            dataframe : {}".format(list_not_in_df))
         else:
            pass
         list_col_keep \
         = [col for col in self.df_invoice.columns if col not in list_to_drop]
         self.df_invoice = self.df_invoice[list_col_keep]
      return
   #----------------------------------------------------------------------------
   
   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def feature_rfm_encode(self):
      """ Apply one hot encoding over RFM feature.
      Each new column 'x' is renamed with the  'w_rfm_x'.
      
      Input : none
      
      Output : none
      """
      if self._is_rfm_encode is False:
         return

      self._rfm_encoder, df_encoded = \
      p5_util.df_rfm_one_hot_encode(self.df_invoice, 'RFM')
      
      #-------------------------------------------------------------------------
      # Encoded columns are renamed with root name = w_rfm_
      #-------------------------------------------------------------------------
      df_encoded, list_col_unchanged \
      = p5_util.df_rename_columns(df_encoded, "w_rfm_")

      #-------------------------------------------------------------------------
      # New features issue from encoded RFM are aggregated to data sample
      #-------------------------------------------------------------------------
      self.df_invoice = pd.concat([self.df_invoice, df_encoded] , axis=1)
      del(df_encoded)
      return
   #----------------------------------------------------------------------------
   
   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def feature_description_nlp(self):
      """Feature Description is vectorized using NLTK package.
      Each vectorized column 'x' is renamed w_nlp_i
      
      Input : none
      Output : none
      """
      list_no_words=['SET','PACK']
      self.df_invoice, self._vectorizer, matrix_weights = \
      p5_util.nlp_process(self.df_invoice, 'Description' \
      ,list_no_words=list_no_words)   
      

      #-------------------------------------------------------------------------
      # List of columns to be renaned is built
      #-------------------------------------------------------------------------
      list_colums = list()
      for col in self.df_invoice.columns:
       if str(col).isdigit() is True:
          list_colums.append(col)

      #-------------------------------------------------------------------------
      # Each vectorized column 'x' is renamed w_nlp_i
      #-------------------------------------------------------------------------
      self.df_invoice, list_col_unchanged \
      = p5_util.df_rename_columns(self.df_invoice, list_colums, 'w_nlp_')
      
      #-------------------------------------------------------------------------
      # Description is droped from columns
      #-------------------------------------------------------------------------
      del(self.df_invoice['Description'])
      return
   #----------------------------------------------------------------------------
   
   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def clusters_build(self):
      """Builds clusters from transformed data.
      Input : none
      Output : none
      """
      #-------------------------------------------------------------------------
      # Read all features dumped files, agregate them and dump them into a 
      # file.
      #-------------------------------------------------------------------------
      self._oP5_SegmentClassifier.df_customers_fileRead()
      
      #-------------------------------------------------------------------------
      #  Read df_customers dataframe from dumped file
      #-------------------------------------------------------------------------
      df_customers = p5_util.object_load(self._df_customers_fileName)
      X = df_customers.values
      print("df_customers : "+str(df_customers.shape))
      
      #-------------------------------------------------------------------------
      #  Get clustering model
      #-------------------------------------------------------------------------
      cluster_model_name = self._cluster_model_name
      dict_param_cluster = self._dict_cluster_model[cluster_model_name]
      n_clusters = dict_param_cluster['n_clusters']
      
      
      print("Clustering model : "+str(cluster_model_name))
      print("Clustering parameters : "+str(dict_param_cluster))
      
      
      #-------------------------------------------------------------------------
      #  Building clusters
      #-------------------------------------------------------------------------
      if cluster_model_name == 'GMM':
         covariance_type = dict_param_cluster['covariance_type']
         cluster_model \
         = GaussianMixture(n_clusters, covariance_type=covariance_type\
            , random_state=0).fit(X)
      elif cluster_model_name == 'KMEANS':
         cluster_model = KMeans(n_clusters = n_clusters) 
         cluster_model.fit(X) 
      else:
         print("\n*** ERROR : Unknown cluster model : "+str(cluster_model_name))

      self._y_clusters = cluster_model.predict(X)
      del(df_customers)

      return

   #----------------------------------------------------------------------------
   
   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def classifier_build(self):
      #-------------------------------------------------------------------------
      #  Read df_customers dataframe from dumped file
      #-------------------------------------------------------------------------
      df_customers = p5_util.object_load(self._df_customers_fileName)
      X = df_customers.values
      print("df_customers : "+str(df_customers.shape))
      if self.classifier_name == 'RandomForests' :
         csr_invoice_matrix = sparse.csr_matrix(X)
         dict_parameters = self.dict_classifier_param[self.classifier_name]
         nb_forests = dict_parameters['nb_forests']
         
         X_train, X_test, y_train, y_test \
         = train_test_split(csr_invoice_matrix, self._y_clusters, test_size=0.8)
         
         rfc = RandomForestClassifier(n_estimators=nb_forests)
         self._classifier_model = rfc.fit(X_train, y_train)

      else:
         print("\n***ERROR : Algorithm : "+self.classifier_name\
         +" not yet supported!")
         

   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def model_dump(self):
      """ Backup into a dumped file classifier model built from this object.
      Dumped classifier is accessed via file name : 
      --> ./data/_oP5_SegmentClassifier.dump
      """
      import P5_SegmentClassifier
      path_to_model = './data/_oP5_SegmentClassifier.dump'
      
      self._oP5_SegmentClassifier._classifier_model = self._classifier_model
      self._oP5_SegmentClassifier._classifier_name = self._classifier_name
      self._oP5_SegmentClassifier._cluster_model_name = self._cluster_model_name

      # TBD : _y_clusters as properties; then test over None value will 
      # be embedded into property getter and setter.
      if self._y_clusters is not None:
         self._oP5_SegmentClassifier._y_clusters = self._y_clusters.copy()
      else:
         self._oP5_SegmentClassifier._y_clusters = None
      
      if self._dict_classifier_param  is not None:
         self._oP5_SegmentClassifier._dict_classifier_param \
         = self._dict_classifier_param.copy()
      else:         
         self._oP5_SegmentClassifier._dict_classifier_param = None

      #-------------------------------------------------------------------------
      # Original dataframe is backup into _oP5_SegmentClassifier
      # While current dataframe is removed.
      #-------------------------------------------------------------------------
      list_col = [col for col in self._df_invoice_original.columns \
      if col not in ['Country']]
      
      
      file_name = './data/df_invoice_line_clean.dump'
      df_invoice_original = p5_util.object_load(file_name)
      print(df_invoice_original[list_col].columns)

      self._oP5_SegmentClassifier._df_invoice_original \
      = df_invoice_original[list_col].copy()
      print(self._oP5_SegmentClassifier._df_invoice_original.shape)
      
      
      #-------------------------------------------------------------------------
      # List of customers identifier from data issued from sampling is copied 
      # into _oP5_SegmentClassifier
      #-------------------------------------------------------------------------
      self._oP5_SegmentClassifier._arr_sample_customerID \
      = np.unique(self._df_invoice.CustomerID.values).copy()
      
      #-------------------------------------------------------------------------
      # Dataframe with customers out of sample is copied into 
      # _oP5_SegmentClassifier
      #-------------------------------------------------------------------------
      self._oP5_SegmentClassifier._df_invoice_line_out_sample \
      = self._df_invoice_line_out_sample.copy()
      
      #-------------------------------------------------------------------------
      # Dataframe containing data sample used for building classifier model
      # is cleaned.
      #-------------------------------------------------------------------------
      self._oP5_SegmentClassifier._df_invoice_line = pd.DataFrame()

      #-------------------------------------------------------------------------
      # Effective dump of classifier model
      #-------------------------------------------------------------------------
      P5_SegmentClassifier.model_dump(self._oP5_SegmentClassifier\
      , path_to_model)
      return
   #----------------------------------------------------------------------------
      
#-------------------------------------------------------------------------------
   
#-------------------------------------------------------------------------------

