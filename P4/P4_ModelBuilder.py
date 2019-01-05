from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model

from sklearn import metrics
from sklearn import dummy


from sklearn.model_selection import GridSearchCV



import pandas as pd
import numpy as np
import hashlib
import pickle
import time
import scipy
import pickle
import zlib

from p3_util import *
from p4_util import *

#from LinearDelayPredictor import *
import LinearDelayPredictor

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def dump_me(self, fileName=None) :
   """Dump this object into file ./data/oP4_ModelBuilder.dumped.
   It is used to save data attributes.
   Then reload them after changing any object methods.
   This save time when debugging with big data.
   """
   if fileName is None :
      fileName = "./data/oP4_ModelBuilder.dump"
   else :
      pass
   with open(fileName,"wb") as dumpedFile:
       oPickler = pickle.Pickler(dumpedFile)
       oPickler.dump(self)
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def reloadme(dumpFileName=None) :
    """
    """
    if dumpFileName is None :
        dumpFileName = "./data/oP4_ModelBuilder.dump"
    else :
        pass

    oP4_ModelBuilder = None
    try:
        with open(dumpFileName, 'rb') as (dataFile):
            oUnpickler = pickle.Unpickler(dataFile)
            oP4_ModelBuilder = oUnpickler.load()
    except FileNotFoundError as fileNotFoundError:
        print('\n*** ERROR : file not found : ' + dumpFileName\
        +" Error message= "+str(fileNotFoundError) )

    return oP4_ModelBuilder
#----------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
class P4_ModelBuilder() :
   ''' 
   This class aims to build data model for project P4 for Openclassrooms 
   course of datascientist training.
   Services provided by this class are :
      -> It builds computational data issues from CSV files.
      -> It proceeds to data preparation : data cleaning and scaling.
      -> It builds computational models from linear regression algorithms 
         issued fro scikitlearn library.
      -> It builds component in charge of flight delay predictions : LinearDelayPredictor
         LinearDelayPredictor class aims to be deployed for production.
      
   Instructions flow :
   -------------------
      main()
      |
      +-->load_and_clean(list_col_keep)
      |   |
      |   +--> < Boucle sur les fichiers a lire >
      |   |    |
      |   |    +--> < loading file into temp dataframe>
      |   |    |
      |   |    +--> < cleaning  temp dataframe regarding list given as parameter >
      |   |    |
      |   |    +--> < temp dataframe concatenation with working dataframe >
      |   |
      |   +--> _fract_whole_data()       
      |
      +-->data_build(list_ref, list_quant, list_target)
      |   |
      |   +--> climat_model_build()
      |   |
      |   +--> _list_store(list_ref, list_quant, list_target)
      |   |
      |   +--> _model_preprocessing()
      |   |    |
      |   |    +--> remove_outliers_delay()  
      |   |    |
      |   |    +--> split_delay()
      |   |    |
      |   |    +--> filter_carrier_model()  
      |   |    |
      |   |    +--> _build_week_of_month()  
      |   |    |
      |   |    +--> route() /**Construction des routes par hash */  
      |   |    |    |
      |   |    |    +--> <Store route dataframe into oLinearDelayPredictor >
      |   |    |
      |   |    +--> _user_test_split()
      |   |    |    |
      |   |    |    +--><Build and store user test dataframe >
      |   |    |
      |   |    +--> _convert_periodic_values()
      |   |     
      |   +--> _build_qualitative_list()
      |   |
      |   +--> _data_for_computation()
      |   |    |
      |   |    +--> < For any route>
      |   |    |    |
      |   |    |    +--><Data is structured for computation >
      |   |    |
      |   |    +--> < Low memory flag : drop unused columns from dataframe >
      |
      +--> _model_build()
      |   |
      |   +--> _model_best_hyper_parameters()
      |   |
      |   +--> _model_compute()
      |   |    |
      |   |    +--><Metrics, hyper-parameter, model structure are stored into model dictionary>
      |
      +--> predictor_dump()
      |
      +--> predictor_load()
   '''
   
   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def __init__(self, path_to_data=None, year=None) :
      path_to_dumped_data = "./data/"

      self._oLinearDelayPredictor = LinearDelayPredictor.LinearDelayPredictor(path_to_dumped_data)

      self._path_to_data = path_to_data
      self._str_year = str(year)
      self._dumpFile = 'oLinearDelayPredictor.dump'
      
      self._df = pd.DataFrame()
      self._col_quant_count = 0
      
      self._list_quantitative = dict()
      self._list_ref = None
      self._list_qualitative = list()
      self._list_target = None
      self._list_carrier_id = list()
      self._list_route_excluded = list()
      self._list_excluded = list()
      self._list_periodic_feature = list()
      
      self._is_route_in_model = False
      self._is_delay_outlier_removed = False
      self._percent_removed_outliers = 0
      self._is_carrier_model = False
      self._is_route_restricted = True

      self._dict_delay_splitted ={'neg':0,'pos':0}
      self._df_neg = pd.DataFrame()
      self._df_pos = pd.DataFrame()
      self._X_std = None
      self._y = None
      self._test_size = 0.3
      
      self._dict_route_y = dict()
      self._route_count = 0
      self._dict_route_data = dict()
      self._is_low_memory = False
      self._frac_test = 0.1
      self._modulo_month = 4
      self._fract_data = -1
      self._dict_climat = dict()
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   # Properties
   #----------------------------------------------------------------------------
   def _get_is_route_in_model(self) :
      return self._is_route_in_model
      
   def _set_is_route_in_model(self, is_route_in_model) :
      self._is_route_in_model = is_route_in_model
   #----------------------------------------------------------------------------
   def _get_is_delay_outlier_removed(self) :
      return self._is_delay_outlier_removed

   def _set_is_delay_outlier_removed(self, is_delay_outlier_removed) :
      self._is_delay_outlier_removed = is_delay_outlier_removed
   #----------------------------------------------------------------------------
   def _get_is_carrier_model(self) :
      return self._is_carrier_model

   def _get_list_carrier_id(self) :
      return self._list_carrier_id

   def _set_list_carrier_id(self, list_carrier_id) :
      if len(list_carrier_id) > 0 :
         self._is_carrier_model = True
         self._list_carrier_id = list_carrier_id.copy()
      else :
         self._is_carrier_model = False
         self._list_carrier_id = list()
      
   #----------------------------------------------------------------------------
   def _get_dict_model(self) :
      return self.oLinearDelayPredictor.dict_model      
   #----------------------------------------------------------------------------
   def _get_dict_delay_splitted(self) :
      return self._dict_delay_splitted

   def _set_dict_delay_splitted(self, dict_delay_splitted) :
      self._dict_delay_splitted = dict_delay_splitted.copy()      
   #----------------------------------------------------------------------------
   def _get_test_size(self) :
      return self._test_size

   def _set_test_size(self, test_size) :
      self._test_size = test_size
      
   #----------------------------------------------------------------------------
   def _get_model_name(self) :
      return self.oLinearDelayPredictor.model_name

   def _set_model_name(self, model_name) :
      self.oLinearDelayPredictor.model_name = model_name      
   #----------------------------------------------------------------------------
   def _get_route_count(self) :
      return self._route_count
   #----------------------------------------------------------------------------
   def _get_is_route_restricted(self) :
      return self._is_route_restricted

   def _set_is_route_restricted(self, is_route_restricted) :
      self._is_route_restricted = is_route_restricted      
   #----------------------------------------------------------------------------
   def _get_list_route(self) :
      if 'HROUTE' in self.oLinearDelayPredictor._df_route.columns :
         return (self.oLinearDelayPredictor._df_route.HROUTE.unique()).tolist()
      else :
         return None
   #----------------------------------------------------------------------------
   def _get_is_low_memory(self) :
      return self._is_low_memory

   def _set_is_low_memory(self, is_low_memory) :
      self._is_low_memory = is_low_memory      
   #----------------------------------------------------------------------------
   def _get_X_std(self) :
      return self.decompress(self._X_std)

   def _set_X_std(self, X_std) :
      self._X_std = self.compress(X_std) 
   #----------------------------------------------------------------------------
   def _get_y(self) :
      return self.decompress(self._y)

   def _set_y(self, y) :
      self._y = self.compress(y) 
   #----------------------------------------------------------------------------
   def _get_oLinearDelayPredictor(self) :
      return self._oLinearDelayPredictor
   #----------------------------------------------------------------------------
   def _linear_combination(self, my_vector) :
         self.oLinearDelayPredictor._linear_combination(my_vector)
         
   #----------------------------------------------------------------------------
   def _param_transform(self, flight_parameters) :
         self.oLinearDelayPredictor._param_transform(flight_parameters)
   #----------------------------------------------------------------------------
   def _forecast_delay(self, flight_parameters) :
      self.oLinearDelayPredictor._forecast_delay(flight_parameters)
   #----------------------------------------------------------------------------
   def _print_result(self) :
      self.oLinearDelayPredictor._print_result()
   #----------------------------------------------------------------------------
   def _get_year(self) :
      return self._str_year

   def _set_year(self, year) :
      self._str_year = str(year)
   #----------------------------------------------------------------------------
   def _get_dict_feature_processor(self):
      return self._oLinearDelayPredictor._dict_feature_processor.copy()

   def _set_dict_feature_processor(self,dict_feature_processor):
      self._oLinearDelayPredictor._dict_feature_processor = dict_feature_processor.copy()
   #----------------------------------------------------------------------------
   def _get_fract_data(self) :
    return self._fract_data      

   def _set_fract_data(self, fract_data) :
    self._fract_data =   fract_data   
   #----------------------------------------------------------------------------
   def _get_dict_climat(self) :
    return self._dict_climat
    
   def _set_dict_climat(self, dict_climat) :
    self._dict_climat = dict_climat
    
   #----------------------------------------------------------------------------
   def _get_frac_test(self) :
    return self._frac_test
    
   def _set_frac_test(self, frac_test) :
    self._frac_test = frac_test
   
   #----------------------------------------------------------------------------
    
   is_route_in_model = property(_get_is_route_in_model, _set_is_route_in_model)

   is_delay_outlier_removed = property(_get_is_delay_outlier_removed\
   , _set_is_delay_outlier_removed)

   is_delay_outlier_removed = property(_get_is_delay_outlier_removed\
   , _set_is_delay_outlier_removed)
   
   is_carrier_model = property(_get_is_carrier_model)

   dict_model = property(_get_dict_model)

   dict_delay_splitted = property(_get_dict_delay_splitted, _set_dict_delay_splitted)

   list_carrier_id = property(_get_list_carrier_id, _set_list_carrier_id)

   test_size = property(_get_test_size, _set_test_size)

   model_name = property(_get_model_name, _set_model_name)
   
   route_count = property(_get_route_count)

   is_route_restricted = property(_get_is_route_restricted, _set_is_route_restricted)

   list_route = property(_get_list_route)
   
   is_low_memory = property(_get_is_low_memory, _set_is_low_memory)

   X_std = property(_get_X_std, _set_X_std)

   y = property(_get_y, _set_y)
   
   oLinearDelayPredictor = property(_get_oLinearDelayPredictor)
   
   linear_combination = property(_linear_combination)

   param_transform = property(_param_transform)

   forecast_delay = property(_forecast_delay)

   year = property(_get_year, _set_year)
   
   dict_feature_processor = property(_get_dict_feature_processor, _set_dict_feature_processor)
   
   fract_data = property(_get_fract_data, _set_fract_data)
   
   dict_climat = property(_get_dict_climat, _set_dict_climat)
   
   frac_test = property(_get_frac_test, _set_frac_test)
   #----------------------------------------------------------------------------

   #---------------------------------------------------------------------------
   # 
   #---------------------------------------------------------------------------
   def predictor_load(self) :
      '''Loads LinearDelayPredictor object from a dumped file.
      '''
      self._oLinearDelayPredictor = LinearDelayPredictor.LinearDelayPredictor.load_dumped()

   #---------------------------------------------------------------------------
   
   #---------------------------------------------------------------------------
   # 
   #----------------------------------------------------------------------------
   def predictor_dump(self):
      ''' Dump LinearDelayPredictor object into a dumped file.
      All required data in LinearDelayPredictor object issued from 
      P4_ModelBuilder are added in LinearDelayPredictor object.
      '''
      #-------------------------------------------------------------------------
      # Model for climat is added into oLinearDelayPredictor
      #-------------------------------------------------------------------------
      self.oLinearDelayPredictor._dict_climat = self._dict_climat.copy()

      self.oLinearDelayPredictor.dump(self._dumpFile)
   #---------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def dumpme(self) :
       """Dump this object into file ./data/oP4_ModelBuilder.dumped.
       It is used to save data attributes.
       Then reload them after changing any object methods.
       This save time when debugging with big data.
       """
       fileName = "./data/oP4_ModelBuilder.dump"
       with open(fileName,"wb") as dumpedFile:
           oPickler = pickle.Pickler(dumpedFile)
           oPickler.dump(self)
   #----------------------------------------------------------------------------


   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def copy(self, oP4_ModelBuilder):
      ''' Copy all attributes from object given as parameter into this object.
      '''
      self._df = oP4_ModelBuilder._df.copy()
      self._path_to_data = oP4_ModelBuilder._path_to_data

      self._list_quantitative = oP4_ModelBuilder._list_quantitative
      self._list_ref = oP4_ModelBuilder._list_ref
      self._list_qualitative = oP4_ModelBuilder._list_qualitative
      self._list_target = oP4_ModelBuilder._list_target
      self._col_quant_count = oP4_ModelBuilder._col_quant_count
      self._route_count = oP4_ModelBuilder._route_count

      self._is_route_in_model = oP4_ModelBuilder._is_route_in_model
      self._is_delay_outlier_removed = oP4_ModelBuilder._is_delay_outlier_removed
      self._is_carrier_model = oP4_ModelBuilder._is_carrier_model
      self._dict_delay_splitted = oP4_ModelBuilder._dict_delay_splitted
      self._test_size       = oP4_ModelBuilder._test_size

      if oP4_ModelBuilder._X_std is not None :
         self.X_std = oP4_ModelBuilder.X_std.copy()

      if oP4_ModelBuilder._y is not None :
         self.y = oP4_ModelBuilder.y.copy()

      #if len(oP4_ModelBuilder._dict_model) != 0 and oP4_ModelBuilder._dict_model is not None:
      #   self._dict_model = oP4_ModelBuilder._dict_model.copy()
         
      if len(oP4_ModelBuilder._dict_delay_splitted) != 0 and oP4_ModelBuilder._dict_delay_splitted is not None:
         self._dict_delay_splitted = oP4_ModelBuilder._dict_delay_splitted.copy()

      if len(oP4_ModelBuilder.list_carrier_id) != 0 and oP4_ModelBuilder.list_carrier_id is not None:
         self._list_carrier_id = oP4_ModelBuilder.list_carrier_id.copy()

      #self._encoder = oP4_ModelBuilder._encoder
      self.oLinearDelayPredictor.copy(oP4_ModelBuilder.oLinearDelayPredictor)
      
      if len(oP4_ModelBuilder._list_route_excluded) > 0 :
         self._list_route_excluded = oP4_ModelBuilder._list_route_excluded.copy()

      if len(oP4_ModelBuilder._list_excluded) > 0 :
         self._list_excluded = oP4_ModelBuilder._list_excluded.copy()

      if len(oP4_ModelBuilder._list_periodic_feature) > 0 :
         self._list_periodic_feature = oP4_ModelBuilder._list_periodic_feature.copy()


      #if oP4_ModelBuilder._std_scale is not None :
      #   self._std_scale = oP4_ModelBuilder._std_scale

      #if oP4_ModelBuilder._model_name is not None :
      #   self._model_name = oP4_ModelBuilder._model_name
      self._dict_route_data = oP4_ModelBuilder._dict_route_data.copy()
      self._is_route_restricted = oP4_ModelBuilder._is_route_restricted
      self._is_low_memory = oP4_ModelBuilder._is_low_memory
      self._str_year = oP4_ModelBuilder._str_year
      self.dict_feature_processor = oP4_ModelBuilder.dict_feature_processor
      self._modulo_month = oP4_ModelBuilder._modulo_month
      self._fract_data = oP4_ModelBuilder._fract_data
      self._dict_climat = oP4_ModelBuilder._dict_climat.copy()
      self.frac_test = oP4_ModelBuilder.frac_test
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def _print_result(self) :
      ''' Print all results issues from regression model optimizations.
      '''
      if self.list_route is None :
         return
      if 0 >=  len(self._dict_route_data) :
         return
         
      #for route in self.list_route :
      nb_route = len(self._dict_route_data)
      dict_score_name_mean = dict()
      for route in self._dict_route_data.keys() :
         #print("\n---- Route : "+str(route)+" ----------------------------------")
            
         dict_route_data = self._dict_route_data[route].copy()

         if 'result' not in dict_route_data.keys() :
            continue

         dict_model = dict_route_data['result']
         if dict_model is not None and len(dict_model) >0:
            for model_name in dict_model.keys() :        
               #print("\nModel name .............................. : {}".format(model_name) )
               if model_name in dict_score_name_mean.keys() :
                pass
               else :
                dict_score_name_mean[model_name] = 0.0
               dict_result = dict_model[model_name]
               for score_name in dict_result.keys() :
                  if score_name != 'model' :
                     #print(score_name, dict_result)
                     if dict_result[score_name] is None :
                        #print(score_name+" scoring .............................. : None")
                        pass
                     else :
                        if isinstance(dict_result[score_name],dict):
                           #print(score_name+" parameters name ...................... : {}".format(dict_result[score_name]))
                           pass
                        else :
                           #print(score_name+" scoring .............................. : %1.3F" %dict_result[score_name])
                           if 'MAE' == score_name :
                            dict_score_name_mean[model_name] +=dict_result[score_name]
                           else :
                            pass
                     
               #print("")      
         #print("\n-------------------------------------------------------------")
      for model_name in dict_score_name_mean.keys() :
        total_score = dict_score_name_mean[model_name]/nb_route
        print("\n*** "+model_name+" MAE for all routes : %1.3F" %total_score)  
   #----------------------------------------------------------------------------


   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def print(self) :
      '''Print attributes from ths object.
      '''
      print("\n-------------------------------------------------------------")
      print("Data set year : ......................... : "+str(self._str_year))
      print("Low memory flag : ....................... : "+str(self._is_low_memory))
      if self._df is None :
         print("*** WARNING : Data model size : dataframe is None!")
      else :         
         print("Data model dimensions : ................. :"+str(self._df.shape))
      #for col in self._df.columns :
      print("Route in model : ........................ : "+str(self._is_route_in_model))
      if 'HROUTE' in self._df.columns and self._is_route_in_model is True:
         print("Number of routes in model : ............. : "+str(self._route_count))
         print("Excluded features list related to route . : "+str(self._list_route_excluded))
         print("Restricted routes in model : ............ : "+str(self._is_route_restricted))
      print("Removing outliers : ..................... : "+str(self._is_delay_outlier_removed))
      print("Percent removed outliers : .............. : %1.2F" %self._percent_removed_outliers)
      print("Split delay config ...................... : "+str(self._dict_delay_splitted))
      print("Carrier in model : ...................... : "+str(self._is_carrier_model))
      if self._is_carrier_model is True :
         print("Carrier ID list ......................... : "+str(self._list_carrier_id))
      print("")
      print("Qualitative features list ............... : "+str(self._list_qualitative))
      print("Quantitative features list .............. : "+str(self._list_quantitative))
      print("Target features list .................... : "+str(self._list_target))
      print("Excluded features list .................. : "+str(self._list_excluded))
      print("Periodic features list .................. : "+str(self._list_periodic_feature))
      
      print("References features list ................ : "+str(self._list_ref))
      print("Number of columns for quantitative data . : "+str(self._col_quant_count))
      
      if self._X_std is not None :
         print("Data model dimensions for computation .. : "+str(self.X_std.shape))
      if self._y is not None :
         print("Target model dimensions for computation .. : "+str(self.y.shape))

      print("Test size ratio ......................... : "+str(self._test_size))
      print("Features processor ...................... : {}".format(self.dict_feature_processor))
      print("Modulo for loading file  ................ :  "+str(self._modulo_month))
      print("Fraction of data  ....................... :  "+str(self.fract_data))
      print("Fraction of test  ....................... :  "+str(self.frac_test))
      print("Climat model ............................ :  "+str(self._dict_climat))

      #self.oLinearDelayPredictor.print()
      self._print_result()
      print("\n------------------DataFrame------------------------------------")
      print(self._df.columns)
      print("\n-------------------------------------------------------------")
      
   #----------------------------------------------------------------------------      

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def _fract_whole_data(self) :
      """This method allows to use a part of data in this object.
      During test and validation process, it allows to increase speed processing.
      Fraction value is held in attribute _fract_data.
      When _fract_data value is -1, then the whole data is kept.
      """
      if self._fract_data == -1 :
         pass
      else :
         rows = self._df.shape[0]
         fract_rows = int(rows*self._fract_data)
         self._df = self._df.sample(fract_rows).copy()
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def _state_climate_build(self, state) :
      if state not in self.dict_climat.keys() :
         state_climat_value = 3
      else :
         state_climat_value = self.dict_climat[state]
      return state_climat_value
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def climat_model_build(self) :
      """Climat model integration to data model.

      Each state is assigned among 6 climate categories.

      Dataframe is enreached with 2 new columns : ORIGIN_STATE_ABR and 
      DEST_STATE_ABR
      """
      dict_climat = dict()
      list_state = self._df.ORIGIN_STATE_ABR.unique().tolist()
      # Climat tempéré océanique : 1
      dict_climat['WA']=1
      dict_climat['OR']=1

      # Climat tempéré continental sec : 2
      list_climate_2 = ['ID','MT','WY','UT','CO','ND','SD','NE','KS','OK']
      for state in list_climate_2 :
         dict_climat[state]=2

      # Climat tempéré continental pacifique : 3

      # Climat sub-tropical : 4
      list_climate_4 = ['VA','NC','GA','FL','AL','TN','MS','LA','AR']
      for state in list_climate_4 :
         dict_climat[state]= 4

      # Climat Aride : 5
      list_climate_5 = ['AZ','TX','NM','NV']
      for state in list_climate_5 :
         dict_climat[state]= 5

      # Climat tempéré méditéranéen : 6
      list_climate_6 = ['CA']
      for state in list_climate_6 :
         dict_climat[state]= 6

      for state in list_state :
         if state not in dict_climat.keys() :
            dict_climat[state] = 3

      self.dict_climat = dict_climat
      self._df['ORIGIN_CLIMAT'] = self._df.ORIGIN_STATE_ABR.apply(self._state_climate_build)
      self._df['DEST_CLIMAT'] = self._df.DEST_STATE_ABR.apply(self._state_climate_build)
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def _clean_value_from_list(self, list_for_cleaning):
       ''' Drop raws considering criterias applied on values for 
       each feature given as list in parameter.
       Criterias apply as following :
         --> If any value is not digit, raw containing this feature is deleted.
         --> All values are casted into integer type.'''
        
       if 0>= len(list_for_cleaning)  or list_for_cleaning is None :
         return

       for column_for_cleaning in list_for_cleaning :
           # mark as -1 fields that are not digit
           print("\n"+str(column_for_cleaning)+" : ...")
           #----------------------------------------------------------------------
           # Drop any rows with Nan values
           #----------------------------------------------------------------------
           print(self._df.shape)
           self._df[column_for_cleaning].dropna(axis=0, inplace=True)
           print(self._df.shape)


           #--------------------------------------------------------------------
           # Printing for debug
           #--------------------------------------------------------------------
           if column_for_cleaning != 'ARR_DELAY' \
           and column_for_cleaning != 'ORIGIN_AIRPORT_ID' \
           and column_for_cleaning != 'DEST_AIRPORT_ID':
               print(self._df[column_for_cleaning].unique())

           #--------------------------------------------------------------------
           # Mark none digit values in order raw holding this value to be deleted.
           #--------------------------------------------------------------------
           self._df[column_for_cleaning] = \
           self._df[column_for_cleaning].apply(p4_mark_none_digit)

           #--------------------------------------------------------------------
           # Printing for debug
           #--------------------------------------------------------------------
           if column_for_cleaning != 'ARR_DELAY' \
           and column_for_cleaning != 'ORIGIN_AIRPORT_ID' \
           and column_for_cleaning != 'DEST_AIRPORT_ID':
               print(self._df[column_for_cleaning].unique())

           #--------------------------------------------------------------------
           # All digit are casted into integer
           #--------------------------------------------------------------------
           try :
               self._df[column_for_cleaning] = \
               self._df[column_for_cleaning].apply(lambda x: int(x))
           except  ValueError as valueError:
               print("*** ERROR : column= "+str(column_for_cleaning)+" error= "+str(valueError))
           #--------------------------------------------------------------------
           # Printing for debug
           #--------------------------------------------------------------------
           if column_for_cleaning != 'ARR_DELAY' \
           and column_for_cleaning != 'ORIGIN_AIRPORT_ID' \
           and column_for_cleaning != 'DEST_AIRPORT_ID':
               print(self._df[column_for_cleaning].unique())

           if column_for_cleaning != 'ARR_DELAY' :
               #----------------------------------------------------------------
               # Excepted values from feature ARR_DELAY
               # raws with strictly negative value are deleted.
               #----------------------------------------------------------------
               self._df = self._df[self._df[column_for_cleaning]>=0] 


           #--------------------------------------------------------------------
           # Printing for debug
           #--------------------------------------------------------------------
           if column_for_cleaning != 'ARR_DELAY' \
           and column_for_cleaning != 'ORIGIN_AIRPORT_ID' \
           and column_for_cleaning != 'DEST_AIRPORT_ID':
               print(self._df[column_for_cleaning].unique())

           #--------------------------------------------------------------------
           # Months belongs to interval [1, 12]
           #--------------------------------------------------------------------
           if column_for_cleaning == 'MONTH' :
               self._df = self._df[self._df[column_for_cleaning]<=12] 

           print(column_for_cleaning+" : Done!\n")
   #-------------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def _clean_value_from_list_deprecated(self, list_for_cleaning):
       ''' Drop raws considering criterias applied on values for 
       each feature given as list in parameter.
       Criterias apply as following :
         --> If any value is not digit, raw containing this feature is deleted.
         --> All values are casted into integer type.
    '''
        
       if 0>= len(list_for_cleaning)  or list_for_cleaning is None :
         return

       for column_for_cleaning in list_for_cleaning :
           # mark as -1 fields that are not digit
           print("\n"+column_for_cleaning+" : ...")
           #--------------------------------------------------------------------
           # Printing for debug
           #--------------------------------------------------------------------
           if column_for_cleaning != 'ARR_DELAY' \
           and column_for_cleaning != 'ORIGIN_AIRPORT_ID' \
           and column_for_cleaning != 'DEST_AIRPORT_ID':
               print(self._df[column_for_cleaning].unique())

           #--------------------------------------------------------------------
           # Mark none digit values in order raw holding this value to be deleted.
           #--------------------------------------------------------------------
           self._df[column_for_cleaning] = \
           self._df[column_for_cleaning].apply(p4_mark_none_digit)

           #--------------------------------------------------------------------
           # Printing for debug
           #--------------------------------------------------------------------
           if column_for_cleaning != 'ARR_DELAY' \
           and column_for_cleaning != 'ORIGIN_AIRPORT_ID' \
           and column_for_cleaning != 'DEST_AIRPORT_ID':
               print(self._df[column_for_cleaning].unique())

           #--------------------------------------------------------------------
           # All digit are casted into integer
           #--------------------------------------------------------------------
           self._df[column_for_cleaning] = \
           self._df[column_for_cleaning].apply(lambda x: int(x))

           #--------------------------------------------------------------------
           # Printing for debug
           #--------------------------------------------------------------------
           if column_for_cleaning != 'ARR_DELAY' \
           and column_for_cleaning != 'ORIGIN_AIRPORT_ID' \
           and column_for_cleaning != 'DEST_AIRPORT_ID':
               print(self._df[column_for_cleaning].unique())

           if column_for_cleaning != 'ARR_DELAY' :
               #----------------------------------------------------------------
               # Excepted values from feature ARR_DELAY
               # raws with strictly negative value are deleted.
               #----------------------------------------------------------------
               self._df = self._df[self._df[column_for_cleaning]>=0] 


           #--------------------------------------------------------------------
           # Printing for debug
           #--------------------------------------------------------------------
           if column_for_cleaning != 'ARR_DELAY' \
           and column_for_cleaning != 'ORIGIN_AIRPORT_ID' \
           and column_for_cleaning != 'DEST_AIRPORT_ID':
               print(self._df[column_for_cleaning].unique())

           #--------------------------------------------------------------------
           # Months belongs to interval [1, 12]
           #--------------------------------------------------------------------
           if column_for_cleaning == 'MONTH' :
               self._df = self._df[self._df[column_for_cleaning]<=12] 

           #print(self._df.shape,self._df[column_for_cleaning].min(),self._df[column_for_cleaning].max())

           print(column_for_cleaning+" : Done!\n")
   #-------------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def _convert_periodic_values(self, list_periodic_feature=None) :
      '''Convert features from list given as parameter into cosinus values.'''

         
      if list_periodic_feature is None  or 0 == len(list_periodic_feature) :
         pass
      else :
         self.oLinearDelayPredictor._list_periodic_feature = list_periodic_feature.copy()
         for periodic_feature in list_periodic_feature :
            if periodic_feature == 'CRS_DEP_TIME' :
               self._df[periodic_feature] = self._df[periodic_feature].apply(cb_convert_floathour_to_sin)
            else :
               min_value = self._df[periodic_feature].min()
               max_value = self._df[periodic_feature].max()
               self._df[periodic_feature] = \
               self._df[periodic_feature].\
               apply(cb_convert_integer_to_cos,min_value=min_value,max_value=max_value)

      return
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def _user_test_split(self,list_for_user_test=None) :
      '''Split the whole dataframe in two parts : dataframe for building 
      data model and dataframe for tests that will be used for user tests.
       
      Rows from dataframe are shuffled then a fraction of rows are splited 
      for users test.
      '''
      
      #-------------------------------------------------------------------------
      # Dataframe rows are shuffle.
      #-------------------------------------------------------------------------
      rows = self._df.sample().shape[0]

      #-------------------------------------------------------------------------
      # A fraction of whole dataframe rows are used for end-user tests
      #-------------------------------------------------------------------------
      rows = self._df.shape[0]
      fract_rows = int(rows*self._frac_test)
      df_user_test = self._df.sample(fract_rows)

      # ------------------------------------------------------------------------ 
      # All features excluded from this list are droped
      # from dataframe user test.
      # Features ORIGIN_CITY_NAME and DEST_CITY_NAME will be issued from 
      # df_route dataframe.
      # ------------------------------------------------------------------------ 
      list_user_feature_to_drop = list()
      if list_for_user_test is not None :
         for feature in df_user_test.columns :
            if feature not in list_for_user_test :
               list_user_feature_to_drop.append(feature)
      
      df_user_test = df_drop_list_column(df_user_test,list_user_feature_to_drop)    


      #-------------------------------------------------------------------------
      # User test dataframe is copied into LinearDelayPredictor object
      #-------------------------------------------------------------------------
      self._oLinearDelayPredictor._df_user_test = df_user_test.copy()

      #-------------------------------------------------------------------------
      # Remaining rows are used for building computational data
      # list of indexes outside of test dataframe is built and is used 
      # in order to filter rows for building data model.
      #-------------------------------------------------------------------------
      set_test = set(sorted(df_user_test.index.tolist()))
      set_all  = set(sorted(self._df.index.tolist()))
      
      set_difference= set_all.difference(set_test)
      list_index = list(set_difference)
      self._df = self._df.loc[list_index,:]
      
      df_route = self._oLinearDelayPredictor._df_route.copy()
      list_index = self.oLinearDelayPredictor._df_user_test.index.tolist()
      df_route = df_route.loc[list_index,:].copy()
      self.oLinearDelayPredictor._df_route = df_route.copy()
      
      #-------------------------------------------------------------------------
      # Climat model is also copied into oLinearDelayPredictor
      #-------------------------------------------------------------------------
      self.oLinearDelayPredictor.dict_climat = self.dict_climat
      
      #-------------------------------------------------------------------------
      # Resources used in this method are released.
      #-------------------------------------------------------------------------
      del(set_test)
      del(set_all)
      del(df_user_test)
   #----------------------------------------------------------------------------
      
   #-------------------------------------------------------------------------
   #
   #-------------------------------------------------------------------------
   def load_and_clean(self, list_month,list_to_keep,list_for_cleaning,\
   dict_skip_rows=None) :
      #----------------------------------------------------------------------
      # Tempory dataframe initialization; used for data agrregation 
      # issued formm files  
      #----------------------------------------------------------------------
      df_concat = pd.DataFrame()
      #----------------------------------------------------------------------
      # Files will be read and dumped every nb_file_dumped
      #----------------------------------------------------------------------
      nb_file_dumped = 0
      is_partial_file = True
      t0 = time.time()            
      
      if len(list_month) < self._modulo_month :
         is_partial_file = False   

      for month in list_month :
         #-------------------------------------------------------------------
         # Read file and store it into a dataframe
         # Some rows, depending on month-file, leading to read errors, are skipped .
         #-------------------------------------------------------------------
         if dict_skip_rows is not None :
            list_skip_rows = dict_skip_rows[month]
         else :
            list_skip_rows = None         
      
         self._df , list_col_notdigit = \
         p4_df_read_from_list_month(self._path_to_data+self.year, month,\
         list_skip_rows=list_skip_rows)

         #-------------------------------------------------------------------
         # Clean columns regarding column list in list_to_keep
         #-------------------------------------------------------------------
         self._df = self._df[list_to_keep]   
         
         #-------------------------------------------------------------------
         # Drop rows with undefined values
         #-------------------------------------------------------------------
         self._df.dropna(inplace=True, axis=0)
         self.clean(list_to_keep)

         #-------------------------------------------------------------------
         # Aggregation : cleaned dataframe is aggregated into df_concat
         #-------------------------------------------------------------------
         df_concat = pd.concat([df_concat, self._df ],axis=0)

         #-------------------------------------------------------------------
         # Clear dataframe memory before coyping concatened dataframe in it.
         #-------------------------------------------------------------------
         self._df = pd.DataFrame()

         #-------------------------------------------------------------------
         # Dataframe serialisation every 4 read files
         #-------------------------------------------------------------------
         if is_partial_file :
            if 0 == int(month) % 4 :
               #------------------------------------------------------------
               # Copy back aggregated dataframe
               #------------------------------------------------------------
               self._df = pd.DataFrame()
               self._df = df_concat.copy()

               #----------------------------------------------------------------------
               # Clean values from list when these values do not match some criterias.
               # This criteria depends on features into list_for_cleaning.
               #----------------------------------------------------------------------
               self._clean_value_from_list(list_for_cleaning)

               #----------------------------------------------------------------------
               # resources used in this method are released.
               #----------------------------------------------------------------------
               del(df_concat)


               nb_file_dumped += 1
               fileName = self._path_to_data+"partFile_"+str(nb_file_dumped)
               with open(fileName,"wb") as dumpedFile :
                 oPickler = pickle.Pickler(dumpedFile)
                 oPickler.dump(self._df)
               print("\nPartial file = "+str(fileName)+" dumped!\n")    
               #----------------------------------------------------------------------
               # Clear concatened dataframe memory 
               #----------------------------------------------------------------------
               df_concat =pd.DataFrame()    
            else :
               pass
         
         else :
            #----------------------------------------------------------------
            # Copy back aggregated dataframe
            #----------------------------------------------------------------
            self._df = pd.DataFrame()
            self._df = df_concat.copy()

            #----------------------------------------------------------------
            # Clean values from list when these values do not match some criterias.
            # This criteria depends on features into list_for_cleaning.
            #----------------------------------------------------------------
            self._clean_value_from_list(list_for_cleaning)

            #----------------------------------------------------------------
            # resources used in this method are released.
            #----------------------------------------------------------------
            del(df_concat)

      #----------------------------------------------------------------------
      # Read all dumped files
      #----------------------------------------------------------------------
      
      if is_partial_file :
         self._df = pd.DataFrame()
         for nb_file in range(1,nb_file_dumped+1,1) :
            fileName = self._path_to_data+"/"+"partFile_"+str(nb_file)
            try:
               with open(fileName,"rb") as dataFile:
                  oUnpickler = pickle.Unpickler(dataFile)
                  #----------------------------------------------------------------
                  # Aggregation : cleaned dataframe is aggregated with df_concat
                  #----------------------------------------------------------------
                  self._df = pd.concat([self._df, oUnpickler.load() ],axis=0)
            except FileNotFoundError: 
               print("\n*** ERROR : file not found : "+fileName)
      else :
         pass

      #-------------------------------------------------------------------------
      # Fraction rows from the whole dataframe
      #-------------------------------------------------------------------------
      self._fract_whole_data()
      t1 = time.time()
      
      print("\n*** Time for reading and fractioning all data files: %0.3F" %(t1-t0))
   #----------------------------------------------------------------------------



   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def load_and_clean_deprecated(self, list_month, list_to_keep,\
   list_for_cleaning,dict_skip_rows=None) :
      ''' Read CSV files with names matching list of months given as parameter.
      Features from files are filtered with list list_for_cleaning.
      Some rows leading to read errors, are skipped when reading.

      '''
      #----------------------------------------------------------------------
      # Tempory dataframe initialization; used for data agrregation 
      # issued formm files  
      #----------------------------------------------------------------------
      df_concat = pd.DataFrame()
      t0 = time.time()
      for month in list_month :
         #----------------------------------------------------------------------
         # Read file and store it into working dataframe
         # Some rows, depending on month-file, leading to read errors, are skipped .
         #----------------------------------------------------------------------
         if dict_skip_rows is not None :
            list_skip_rows = dict_skip_rows[month]
         else :
            list_skip_rows = None         
         
         self._df , list_col_notdigit = \
         p4_df_read_from_list_month(self._path_to_data+self.year, month,list_skip_rows=list_skip_rows)
         
         #----------------------------------------------------------------------
         # Clean columns regarding column list in list_to_keep
         #----------------------------------------------------------------------
         self.clean(list_to_keep)

         #----------------------------------------------------------------------
         # Aggregation : cleaned dataframe is aggregated with df_concat
         #----------------------------------------------------------------------
         df_concat = pd.concat([df_concat, self._df ],axis=0)

         #----------------------------------------------------------------------
         # Clear dataframe memory before coyping concatened dataframe in it.
         #----------------------------------------------------------------------
         self._df = pd.DataFrame()

      t1 = time.time()
      print("Time for reading data : %0.3F" %(t1-t0))
      
      #----------------------------------------------------------------------
      # Copy back aggregated dataframe
      #----------------------------------------------------------------------
      self._df = pd.DataFrame()
      self._df = df_concat.copy()
      
      #----------------------------------------------------------------------
      # Clean values from list when these values do not match some criterias.
      # This criteria depends on features into list_for_cleaning.
      #----------------------------------------------------------------------
      self._clean_value_from_list(list_for_cleaning)

      #----------------------------------------------------------------------
      # resources used in this method are released.
      #----------------------------------------------------------------------
      del(df_concat)

      return
   #----------------------------------------------------------------------------


   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def split_delay(self) :
      '''Splitting of dataframe into negative delay and positive delay.
      New dataframe do host delays values >0 or delay values <=0
      '''
      for key in self._dict_delay_splitted.keys() :
         value = self._dict_delay_splitted[key]
         if value == 1 :
            if key == 'neg' :
               self._df = self._df[self._df['ARR_DELAY'] <=0]
            elif key == 'pos' :
               self._df = self._df[self._df['ARR_DELAY'] >0]
            else : 
               print("*** WARNING : Unknown value={} for splitting direction={} ".format(value,key))      
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def _build_qualitative_list(self) :
      '''Builds list of features to be encoded.
      This list is built from references features list, target list and 
      quantitative features list.
      * Target list : y feature that is used as target when solving Ax = y
      * References list : list of names used as reference for human readability
      * Quantitative list : list of features used as A matrix for solving 
      equation Ax = b
      '''        
      if self._list_target is None :
         print("*** WARNING : empty target list !")
         return
         
      if self._list_quantitative is None :
         print("*** WARNING : empty quantitative list !")
         return

      if self._list_target is None :
         print("*** WARNING : empty reference list !")
         return

               
      #-------------------------------------------------------------------------
      # Les aeroports d'origine et de destination sont exclus du modèle car 
      # remplacés par la variable HROUTE
      # Pour eviter le dataleakage, les features DISTANCE et CSR_ARR_TIME sont
      # exclues du modele.
      #-------------------------------------------------------------------------
      if 'HROUTE' in self._df.columns :
          self._list_route_excluded = ['ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID'\
          ,'HROUTE','DISTANCE']

      #-------------------------------------------------------------------------
      # Construction de la liste des variables qualitatives.
      #-------------------------------------------------------------------------
      self._list_qualitative = list()
      for col in self._df.columns :
          if col not in self._list_target :
              if col not in self._list_quantitative:
                  if col not in self._list_ref :
                      if col not in self._list_route_excluded :
                         if col not in self._list_excluded :
                             self._list_qualitative.append(col)

   #----------------------------------------------------------------------------
   

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def compress(self, my_object) :
      ''' Compress object given as parameter using zlib object.'''
      # Compress:
      compressed = zlib.compress(pickle.dumps(my_object))
      return compressed
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def decompress(self, compressed_object) :
      ''' Decompress object given as parameter using zlib object.'''
      object = pickle.loads(zlib.decompress(compressed_object))
      return object
   #----------------------------------------------------------------------------


   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def remove_outliers_delay(self):
      '''Remove values from ARR_DELAY feature if this values are outliers.
      Limits inf. and sup. values for outliers are computed using boxplot description.
      '''
      if self._is_delay_outlier_removed is True :
         zmin, zmax = df_boxplot_min_max(self._df , 'ARR_DELAY')

         start = self._df.shape[0]
         self._df = self._df[self._df['ARR_DELAY']<zmax]
         self._df = self._df[self._df['ARR_DELAY']>zmin]
         end = self._df.shape[0]
         self._percent_removed_outliers = (start-end)*100/start
         print("Pourcent valeurs outliers écrêtées : %0.2f" %(self._percent_removed_outliers))
      else :
         pass
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def clean_deprecated(self, list_to_keep) :
      '''Clean all columns outside from list_to_keep
      Drop raws from which values are Nan.
      '''
      #-------------------------------------------------------------------------
      # Build list to drop from list given as parameter
      #-------------------------------------------------------------------------
      list_col_drop = list()
      for col in self._df.columns:
          if col not in list_to_keep :
              list_col_drop.append(col)

      self._df = df_drop_list_column(self._df,list_col_drop)        

      ser_col_nan = self._df.isnull().any()
      list_col_nan = list()
      for col, status in ser_col_nan.iteritems() :
          if status is  True :
              list_col_nan.append(col)
              
      print("Columns targeted for nan : "+str(list_col_nan))

      self._df.dropna(axis=0, how='any', inplace=True)   
   #----------------------------------------------------------------------------
   
   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def clean(self, list_to_keep) :
      '''Clean all columns outside from list_to_keep
      Drop raws from which values are Nan.
      '''
      #-------------------------------------------------------------------------
      # Build list of columns to drop from list given as parameter
      #-------------------------------------------------------------------------
      list_col_drop = list()
      for col in self._df.columns:
         if col not in list_to_keep :
            list_col_drop.append(col)

      #-------------------------------------------------------------------------
      # Columns from built list are droped from dataframe
      #-------------------------------------------------------------------------
      self._df = df_drop_list_column(self._df,list_col_drop)        

      ser_col_nan = self._df.isnull().any()
      list_col_nan = list()
      for col, status in ser_col_nan.iteritems() :
         if status is  True :
            list_col_nan.append(col)

      print("Columns targeted for nan : "+str(list_col_nan))

      #-------------------------------------------------------------------------
      # Drop all rows with any undefined value
      #-------------------------------------------------------------------------
      self._df.dropna(axis=0, how='any', inplace=True)   

   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def route_deprecated(self):
      ''' Build routes : origin airport to destination airport.
      
      Routes are stored into a dataframe column with named HROUTE.
      
      A hash is computed from origin and destination cities.
      This hash is then labelized in order to be stored into datafrmae column 
      HROUTE.
      '''

      self._df['STR_ORIGIN_AIRPORT_ID'] = self._df['ORIGIN_AIRPORT_ID'].apply(lambda x: str(x))
      self._df['STR_DEST_AIRPORT_ID'] = self._df['DEST_AIRPORT_ID'].apply(lambda x: str(x))

      ser_route = pd.Series((self._df['ORIGIN_CITY_NAME'] \
                             + self._df['DEST_CITY_NAME']\
                             + self._df['STR_ORIGIN_AIRPORT_ID']\
                             + self._df['STR_DEST_AIRPORT_ID']),index=self._df.index)

      ser_route = ser_route.apply(lambda val: ("0x"+hashlib.md5(val.encode()).hexdigest()))

      del(self._df['STR_ORIGIN_AIRPORT_ID'])
      del(self._df['STR_DEST_AIRPORT_ID'])

      le = preprocessing.LabelEncoder()
      le.fit(ser_route)
      ser_route = (le.transform(ser_route)).copy()

      
      self._df['HROUTE'] = ser_route.copy()
      del(ser_route)
      
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def route(self) :
      ''' Build routes : origin airport to destination airport.
      
      Routes are stored into a dataframe column with named HROUTE.
      
      A hash is computed from origin and destination cities.
      This hash is then labelized in order to be stored into datafrmae column 
      HROUTE.
      '''
      if self._is_route_in_model is False :
         # Only one global route
         self._df['HROUTE'] = 0
      else :   
         df_route = pd.DataFrame()
         df_route['HROUTE'] = \
         pd.Series((self._df['ORIGIN_CITY_NAME'] + self._df['DEST_CITY_NAME']),index=self._df.index)

         df_route['HROUTE'].apply(lambda val: ("0x"+hashlib.md5(val.encode()).hexdigest()))

         #----------------------------------------------------------------------
         # Hash values are labelized in order to be encoded.
         #----------------------------------------------------------------------
         le = preprocessing.LabelEncoder()
         le.fit(df_route['HROUTE'])
         df_route['HROUTE'] = (le.transform(df_route['HROUTE'])).copy()
         
         self._df['HROUTE'] = df_route['HROUTE'].copy()
         self._route_count = len(self._df['HROUTE'].unique())

         #----------------------------------------------------------------------
         # Clean local variables
         #----------------------------------------------------------------------
         del(le)
         del(df_route)
      #-------------------------------------------------------------------------
      # Store routes into oLinearDelayPredictor
      #-------------------------------------------------------------------------
      list_route_reference = ['ORIGIN_CITY_NAME','DEST_CITY_NAME','HROUTE']
      
      self.oLinearDelayPredictor._df_route = self._df[list_route_reference].copy()
            
   #----------------------------------------------------------------------------

      
   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def _get_route(self, frequency='max') :
      '''  TO BE REWORKED!
      Returns origin and destination cities matching with most 
      frequented route.
      '''
   
      origin_city_name = None
      dest_city_name = None
         
      if frequency == 'min' :
         hroute_index = self._df.HROUTE.value_counts(ascending=True).index[0]
         legend = "Route la moins fréquentée    : "
         
      if frequency == 'mean' :
         hroute_index = self._df.HROUTE.value_counts(ascending=True).index[1500]
         legend = "Route moyennement fréquentée : "

      if frequency == 'max' :
         hroute_index = self._df.HROUTE.value_counts(ascending=False).index[0]
         legend = "Route la plus fréquentée : "

      origin_city_name = self._df[self._df.HROUTE==hroute_index].ORIGIN_CITY_NAME.unique()
      dest_city_name = self._df[self._df.HROUTE==hroute_index].DEST_CITY_NAME.unique()
      
      print(legend+origin_city_name+" --> "+dest_city_name)
      return origin_city_name[0], dest_city_name[0]
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def filter_carrier_model(self):
      '''Filter data regarding carrier identifiers list set in configuration.
      '''
      return
      if self._is_carrier_model is True :
         for carrier_id in self._list_carrier_id :
            self._df = self._df[self._df['AIRLINE_ID']==carrier_id]
            del(self._df['AIRLINE_ID'])
      else :
         pass
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def _build_week_of_month(self) :
      ''' Creates a new feature WEEK_OF_MONTH.
      This feature values belongs to interval [1,5]
      '''
      if 'WEEK_OF_MONTH' in self._list_excluded :
         pass
      else :
         self._df['WEEK_OF_MONTH'] = pd.Series(-1, index=self._df.index)

         self._df['WEEK_OF_MONTH'] = self._df['DAY_OF_MONTH'].apply(p4_week_of_month)
      return
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def _model_preprocessing(self, list_periodic_feature=None,list_for_user_test=None) :
      '''Proceed to data-preparation in order to issue a computational data model.
      This includes :
         --> Remove outliers values from ARR_DELAY feature.
         --> Split delays values as positive and negative (optional)
         --> Builds model per carrier (optional, not implemented)
         --> Builds model per route (optional, implemented)
      '''
   

      #-------------------------------------------------------------------------
      # Outliers removing
      #-------------------------------------------------------------------------
      print("\n*** _model_preprocessing() : "+format(self._df.columns))
      if self._is_delay_outlier_removed is True :
         self.remove_outliers_delay()

      #-------------------------------------------------------------------------
      # Splitting dataframe into positive and negative delays, considering 
      # splitting configuration.
      #-------------------------------------------------------------------------
      self.split_delay()

         
      #-------------------------------------------------------------------------
      # Taking into account carrier into data model
      #-------------------------------------------------------------------------
      self.filter_carrier_model()   

      #-------------------------------------------------------------------------
      # builds WEEK_OF_MONTH feature
      #-------------------------------------------------------------------------
      self._build_week_of_month()
      
      
      #-------------------------------------------------------------------------
      # Taking into account route in data model
      #-------------------------------------------------------------------------
      #if self._is_route_in_model is True :
      self.route()
      if self._is_route_restricted is True :
         origin_city_name, dest_city_name = self._get_route()
         print("Modele pour la route "+origin_city_name+" --> "+ dest_city_name)
         self._df = self._df[self._df['ORIGIN_CITY_NAME']==origin_city_name]
         self._df = self._df[self._df['DEST_CITY_NAME']==dest_city_name]      

      # ------------------------------------------------
      # Dataframe is splitted for having a dataframe 
      # for user tests
      # ------------------------------------------------ 
      self._user_test_split(list_for_user_test=list_for_user_test)

      #----------------------------------------------------------------------
      # Convert periodic values to cos values
      #----------------------------------------------------------------------
      self._convert_periodic_values(list_periodic_feature)

      
   #----------------------------------------------------------------------------


   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def _row_drop_from_list_value(self, list_value_excluded) :
      '''Drop rows contained into list of values given as parameter.
      '''
      if list_value_excluded is None :
         return
      
      for value_excluded in list_value_excluded :

         for col_num in range(self._df[self._list_qualitative].columns.shape[0]):
            arr_index = None
            list_index_condition = None
            if value_excluded in self._df[self._list_qualitative].values[:, col_num] : 
               col_name = self._df[self._list_qualitative].columns[col_num]
               print("Column number = {} / col name= {}".format(col_num, col_name))

               condition = self._df[self._list_qualitative][col_name] == value_excluded
               arr_index = np.where(condition == True)
               if len(arr_index) >0 :
                  list_index_condition = condition.index[arr_index].tolist()

            if list_index_condition is not None :
               for index_condition in list_index_condition :
                  print("Dataframe : before (rows,columns) = {}".format(self._df.shape))
                  self._df = self._df.drop(axis=0, index=index_condition, inplace= False)
                  print("Dataframe : before (rows,columns) = {}".format(self._df.shape))
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def _data_for_computation(self,list_value_excluded = None) :
      '''Build data model for computation : encoder, scaler, matrix, vectors...

      For each one of the route :
         Features belonging to qualitative list are encoded.
         Features belonging to quantitative list are scaled then joined to encoded 
         features.
      
      When _is_low_memory flag is activated, columns from dataframe that are 
      not used anymore are droped.
      '''

      # ------------------------------------------------
      # Columns from list list_value_excluded are dropped
      # ------------------------------------------------ 
      self._row_drop_from_list_value(list_value_excluded)

      # ---------------------------------------------------------------------
      # Features from dataframe are processed considering processor dictionary
      # ---------------------------------------------------------------------
      for feature in self.dict_feature_processor.keys() :
         cb_function  = self.dict_feature_processor[feature]
         self._df[feature] = self._df[feature].apply(cb_function)
      

      # ------------------------------------------------
      # Data are computed based on routes
      # ------------------------------------------------ 
      list_route_to_drop = list()
      print("\n *** _data_for_computation() : encoding features= {}".format(self._list_qualitative))
      nb_count_route = 0
      nb_route = len(self.list_route)
      for route in self.list_route :
         nb_count_route += 1
         sparse_X = None
         sparse_col_encoded = None     
         encoder = None
         X_quantitative_std = None
         std_scale = None
         X_std = None

         # ---------------------------------------------------------------------
         # Create route filter 
         # ---------------------------------------------------------------------
         df_route = self._df[self._df.HROUTE == route]
                  
         # ---------------------------------------------------------------------
         # Data issue from qualitative features are encoded
         # ---------------------------------------------------------------------
         print("\n*** Route= "+str(route)+":"+str(nb_count_route)+"/"+str(nb_route)+" Encoding ...")
         encoder = preprocessing.OneHotEncoder() 
         try :
            sparse_col_encoded = encoder.fit_transform(df_route[self._list_qualitative].values)
         except ValueError as valueError :
            print("\n*** Erreur encodage : {}".format(valueError))


         # ---------------------------------------------------------------------
         # Build target value
         # ---------------------------------------------------------------------
         y = df_route[self._list_target].values
         
         # ---------------------------------------------------------------------
         # Data model issued from quantitative features 
         # is built
         # ---------------------------------------------------------------------

         if self._list_quantitative is not None and 0 < len(self._list_quantitative) :
            X_quantitative_std = df_route[self._list_quantitative].values
            #print("\n***X_quantitative_std.shape= {}".format(X_quantitative_std.shape))

         if X_quantitative_std is None :      
            X_std = sparse_col_encoded
            
         else :
            # ------------------------------------------------------------------
            # Quantitative features are scaled then joined to encoded features.
            # ------------------------------------------------------------------


            # Some track...
            self._col_quant_count =  X_quantitative_std.shape[1]
            if X_quantitative_std.shape[0] <= 0 :
               print("***WARNING : No data! Skipping route = "+str(route))
               list_route_to_drop.append(route)
            else :
               # ------------------------------------------------
               # Conversion from integer type into float type
               # ------------------------------------------------
               X_quantitative_std = X_quantitative_std.astype(float)
               
               # ------------------------------------------------
               # Data scaling
               # ------------------------------------------------
               std_scale = preprocessing.StandardScaler().fit(X_quantitative_std)
               X_quantitative_std = std_scale.transform(X_quantitative_std)
               
               # --------------------------------------------------
               # Quantatitatives data are transformed as a sparse structure
               # --------------------------------------------------
               sparse_X = scipy.sparse.csr_matrix(X_quantitative_std)

               # --------------------------------------------------
               # Sparse structures aggregation
               # --------------------------------------------------
               X_std = scipy.sparse.hstack((sparse_X, sparse_col_encoded))
               
         if X_std is not None :
            dict_route_data =  {'X_std':X_std,'y':y }
            self._dict_route_data[route] = dict_route_data.copy()
            dict_model_route = {'encoder':encoder,'std_scale':std_scale}
            self.oLinearDelayPredictor._dict_model_route[route] \
            = dict_model_route.copy()


         # --------------------------------------------------
         # Drop all local variables
         # --------------------------------------------------
         if std_scale is not None :
            del(std_scale)

         if encoder is not None :
            del(encoder)

         if sparse_col_encoded is not None :
            del(sparse_col_encoded)

         if sparse_X is not None :
            del(sparse_X)

         if X_quantitative_std is not None :
            del(X_quantitative_std)

         if X_std is not None :
            del(X_std)

         if y is not None :
            del(y)

      #-------------------------------------------------------------------------
      # Failed routes are drop from data model
      #-------------------------------------------------------------------------
      
      #if len(list_route_to_drop) > 0 :
      #   set_route = set(sorted(self._list_route))
      #   set_route_to_drop  = set(sorted(list_route_to_drop))
         
      #   set_difference= set_route.difference(set_route_to_drop)
      #   self._list_route = list(set_difference)

      # ------------------------------------------------------------------------ 
      # Columns from dataframe used for data are droped 
      # This may reduce memory amount usage.
      # ------------------------------------------------------------------------ 
      if self._is_low_memory is True :
         self._df = df_drop_list_column(self._df,self._list_qualitative) 
         self._df = df_drop_list_column(self._df,self._list_quantitative) 
         self._df = df_drop_list_column(self._df,self._list_excluded) 
         self._df = df_drop_list_column(self._df,self._list_route_excluded) 
         self._df = df_drop_list_column(self._df,self._list_target) 
         self._df = df_drop_list_column(self._df,self._list_ref) 

   #----------------------------------------------------------------------------



   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def _list_store(self, list_ref, list_quant, list_target, list_excluded, list_periodic_feature) :
      ''' Store into object lists given as parameter.
      These lists will be used as filter when data model will be built.
      Input : 
         list_ref : features in human readable format
         list_quant : quantitative type features 
         list_target: delays feature
         list_excluded : some features to be excluded from data model
      '''
      self._list_ref    = list()
      self._list_quantitative  = list()
      self._list_target = list()
      self._list_excluded = list()
      

      if list_ref is not None :
         self._list_ref    = list_ref.copy()
      else :
         print("***ERROR _list_store() : list_ref is None")
         return

      if list_quant is not None :
         self._list_quantitative  = list_quant.copy()
      else :
         print("***INFO _list_store() : list_quant is None")

      if list_target is not None :
         self._list_target = list_target.copy()
      else :
         print("***ERROR _list_store() : list_target is None")
         return
         
      if list_excluded is not None :
         self._list_excluded = list_excluded.copy()
      else :
         print("***INFO _list_store() : list_excluded is None")

      if list_periodic_feature is not None :
         self._list_periodic_feature = list_periodic_feature.copy()
      else :
         print("***INFO _list_store() : list_periodic_feature is None")

      return
   #----------------------------------------------------------------------------
   
   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def data_build(self, list_ref, list_quant, list_target, list_excluded\
   ,list_for_user_test=None,list_value_excluded=None,list_periodic_feature=None) :
      '''This method allows to build data for computation.
      Lists given as parameters are used as filters on order to build 
      computational data from dataframe.
      Input : 
         list_ref : some features in human readable format
         list_quant : quantitative type features 
         list_target: delays feature
         list_excluded : som features to be excluded from data model
      Computed data result are kept in structures : 
         self.X_std
         self.y
      '''
      print("\n *** 1 Shape =  {}".format(self._df.shape))

      # ------------------------------------------------
      # Climat model integration
      # ------------------------------------------------
      self.climat_model_build()

      # ------------------------------------------------
      # Lists used for filter are stored into object.
      # ------------------------------------------------
      self._list_store(list_ref, list_quant, list_target, list_excluded, list_periodic_feature)
      
      # ------------------------------------------------
      # Data model pre-processing depends on 
      # data model configuration. 
      # Configuration lists are applied to dataframe.
      # ------------------------------------------------
      #print("\n *** 2 Shape =  {}".format(self._df.shape))
      self._model_preprocessing(list_periodic_feature,list_for_user_test)        

      # ------------------------------------------------
      # Building qualitative features list
      # ------------------------------------------------
      #print("\n *** 3 Shape =  {}".format(self._df.shape))
      self._build_qualitative_list()

      # ------------------------------------------------
      # Build data model for computation : matrix, vectors...
      # ------------------------------------------------
      #print("\n *** 4 Shape =  {}".format(self._df.shape))
      
      self._data_for_computation(list_value_excluded)
   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def _model_compute(self, route, regresion_model_name, best_alpha) :
      ''' Compute coefficients of the model which name is given as parameter.

      Computation follows those steps : 
      1) training model along with train data-set and testing model 
      along with test data-set. 
      2) Metrics such as R2, MAE, MSE are computed and stored into a dictionary.
      3) Best hyper-parameter is stored into model a dictionary.
      4) All coefficients for linear regression are stored into a model-dictionary.
      
      Input : 
         route : route identifier for which model is computed
         regresion_model_name : regression model name
         best_alpha : optimal hyper-parameter
      '''
   
      dict_model_result = dict()
      regresion_model = None
      y_predict = None


      if 'LinearRegression' == regresion_model_name :
         regresion_model = linear_model.LinearRegression()
         
      elif 'DummyRegressor' == regresion_model_name : 
         regresion_model = dummy.DummyRegressor(strategy='median')

      elif 'Ridge' == regresion_model_name : 
         regresion_model = linear_model.Ridge(alpha=best_alpha)
         
      elif 'Lasso' == regresion_model_name : 
         regresion_model = linear_model.Lasso(alpha=best_alpha)

      elif 'SGDRegressor' == regresion_model_name : 
         
         regresion_model = linear_model.SGDRegressor(loss='squared_loss'\
         , penalty=best_alpha['penalty'], alpha=best_alpha['alpha']\
         , l1_ratio=best_alpha['l1_ratio'], fit_intercept=True\
         , max_iter=best_alpha['max_iter'], tol=None, shuffle=True, verbose=0\
         , epsilon=0.1\
         , random_state=None, learning_rate='invscaling', eta0=0.01\
         , power_t=0.25, warm_start=False, average=False, n_iter=None)

      else : 
         print("\n*** WARNING : Estimator not supported : \n"+str(regresion_model_name))


      #-------------------------------------------------------------
      # Per route : get all already scaled train and test data set
      #-------------------------------------------------------------
      dict_route_data = self._dict_route_data[route].copy()
      X_std = dict_route_data['X_std']
      y = dict_route_data['y']
      
      X_train_std, X_test_std, y_train, y_test = \
            model_selection.train_test_split(X_std, y, test_size = self._test_size)
            
      #-------------------------------------------------------------
      # Training step
      #-------------------------------------------------------------
      try :
         regresion_model.fit(X_train_std, y_train.ravel())
      except ValueError as valueError :
         origin, destination = self.oLinearDelayPredictor.get_cities_route(route)
         #-------------------------------------------------------------
         # Per route : route in error are recorded into dict_route_data
         #-------------------------------------------------------------
         dict_model_result['model'] = regresion_model
         dict_model_result['alpha'] = None
         dict_model_result['R2']  = None
         dict_model_result['MAE'] = None
         dict_model_result['MSE'] = None
         
         dict_model = dict()
         dict_model[regresion_model_name] = dict_model_result
         dict_route_data['result'] = dict_model.copy()
         self._dict_route_data[route] = dict_route_data.copy()    
         del(dict_route_data)  
         return
      #-------------------------------------------------------------
      # Predictions
      #-------------------------------------------------------------
      y_predict = regresion_model.predict(X_test_std)

      #-------------------------------------------------------------
      # Per route : record of regresion model, R2, MSE, MAE
      # dict_route_data is added with a new field : regresion_model_name
      #-------------------------------------------------------------
      dict_model_result['model'] = regresion_model
      dict_model_result['alpha'] = best_alpha
      dict_model_result['R2']  = regresion_model.score(X_test_std, y_test)
      dict_model_result['MAE'] = metrics.mean_absolute_error(y_predict, y_test)
      dict_model_result['MSE'] = metrics.mean_squared_error(y_predict, y_test)

      dict_model = dict()
      dict_model[regresion_model_name] = dict_model_result
      dict_route_data['result'] = dict_model.copy()
      self._dict_route_data[route] = dict_route_data.copy()    
      del(dict_route_data)  
      
      #-------------------------------------------------------------
      # Adding regression model matching this route into 
      # LinearDelayPredictor object
      #-------------------------------------------------------------
      dict_model_route = self._oLinearDelayPredictor.dict_model_route
      dict_model = dict_model_route[route]

      dict_model[regresion_model_name] = regresion_model
      self._oLinearDelayPredictor._dict_model_route[route] = dict_model.copy()

   #----------------------------------------------------------------------------

   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def _model_best_hyper_parameters(self,route, linear_model_name, list_alphas, n_alphas) :
      ''' Compute best hyper-parameter for linear model given as parameter.
      Best parameter is searched using a cross-validation algorithm.
      Input : 
            linear_model_name : linear model name
            list_alphas       : list of hyper-parameters values; usage depends 
                                on linear model name
            n_alphas          : number of hyper-parameters values; usage depends 
                                on linear model name
      Output : 
            best_alpha : best hyper-parameter
      '''
      n_alphas = 100
      best_alpha = None
      nb_route = len(self._dict_route_data)
      
      #-------------------------------------------------------------
      # Per route : get all already scaled train and test data set
      #-------------------------------------------------------------
      dict_route_data = self._dict_route_data[route].copy()
      X_std = dict_route_data['X_std']
      y = dict_route_data['y']
      
      X_train_std, X_test_std, y_train, y_test = \
            model_selection.train_test_split(X_std, y, test_size = self._test_size)

      t0 = time.time()
      if linear_model_name == "Ridge" :
         ridgecv = linear_model.RidgeCV(alphas= list_alphas\
         , scoring='neg_mean_absolute_error', cv=3)
         try :
            ridgecv.fit(X_train_std, y_train)
         except ValueError as valueError :
            print("*** ERROR : X_train_std dimensions = {}".format(X_train_std.shape))
            return None

         y_predict = ridgecv.predict(X_test_std)
         
         best_alpha = ridgecv.alpha_
         
         print("RidgeCV best MAE =  %0.3f " %ridgecv.score(X_test_std, y_test))
         print("RidgeCV optimal alpha =  %0.3f" %best_alpha)

      elif linear_model_name == "Lasso" :
         lassocv = linear_model.LassoCV(eps=0.001, n_alphas=n_alphas, cv=3)
         try :
            lassocv.fit(X_train_std, y_train)
         except ValueError as valueError :
            print("*** ERROR : X_train_std dimensions = {}".format(X_train_std.shape))
            return None
         y_predict = lassocv.predict(X_test_std)
         
         best_alpha = lassocv.alpha_
         
         print("LassoCV best MAE =  %0.3f " %lassocv.score(X_test_std, y_test))
         print("LassoCV optimal alpha =  %0.3f" %best_alpha)      
         
      elif linear_model_name == "SGDRegressor" :
         parameters = list_alphas
         gscv = GridSearchCV(linear_model.SGDRegressor(), parameters, cv=3\
         , scoring='neg_mean_absolute_error', refit=True)
         try :
            gscv.fit(X_train_std, y_train.ravel())
         except ValueError as valueError :
            print("*** ERROR : X_train_std dimensions = {} / Error= {}".format(X_train_std.shape,valueError))
            return None
         #best_model = reg.best_estimator_
         best_alpha = gscv.best_params_
         
      else :
         print("*** ERROR : Model not supported : "+str(linear_model_name))

      t1 = time.time()
      return best_alpha
   #----------------------------------------------------------------------------


   #----------------------------------------------------------------------------
   #
   #----------------------------------------------------------------------------
   def _model_build(self, dict_models_parameters) :
      ''' Build linear regression model from data built in method data_build().
      Keys from dict_models_parameters dictionary contain models from 
      sklearn.linear_model library.
      Each value from is a new dictionary containing hyper-parameters to be 
      optimized.
      '''
      dict_model_route_error = dict()
      list_route_error = list()
      
      #-------------------------------------------------------------------------
      # Dictionary containing all models with routes error is initialized
      #-------------------------------------------------------------------------
      for regresion_model_name in dict_models_parameters.keys() :
         dict_model_route_error[regresion_model_name] = list_route_error
         
      #-------------------------------------------------------------------------
      # Building regression model for each route in data-model
      #-------------------------------------------------------------------------
      #for route in self.list_route :
      
      i_route = 0
      nb_route = len(self._dict_route_data)
      
      for route in self._dict_route_data.keys() :
         i_route += 1
         status = False     
         t0 = time.time() 
         for regresion_model_name in dict_models_parameters.keys() :
            
            if 'LinearRegression' == regresion_model_name :
               #----------------------------------------------------------------
               # Search for best hyper-parameters is not required
               #----------------------------------------------------------------
               best_alpha = None
               status = self._model_compute(route, regresion_model_name, best_alpha)
               
            elif 'DummyRegressor' == regresion_model_name : 
               best_alpha = None
               status = self._model_compute(route, regresion_model_name, best_alpha)

            elif 'Ridge' == regresion_model_name : 
               #----------------------------------------------------------------
               # Search for best hyper-parameters : extract search parameters 
               # from dictionary given as parameter.
               #----------------------------------------------------------------
               dict_parameters = dict_models_parameters[regresion_model_name]
               list_alphas = dict_parameters['list_alphas']
               n_alphas = dict_parameters['n_alphas']            
               best_alpha = self._model_best_hyper_parameters(route, regresion_model_name\
               , list_alphas, n_alphas) 
               if best_alpha is not None :                  
                  #-------------------------------------------------------------
                  # Compute model along with best hyper-parameter
                  #-------------------------------------------------------------
                  status = self._model_compute(route, regresion_model_name, best_alpha)
               else :
                  status = False
            elif 'Lasso' == regresion_model_name : 
               #----------------------------------------------------------------
               # Search for best hyper-parameters : extract search parameters 
               # from dictionary given as parameter.
               #----------------------------------------------------------------
               dict_parameters = dict_models_parameters[regresion_model_name]
               list_alphas = dict_parameters['list_alphas']
               n_alphas = dict_parameters['n_alphas']            

               best_alpha = self._model_best_hyper_parameters(route, regresion_model_name\
               , list_alphas, n_alphas) 

               if best_alpha is not None :            
                  #-------------------------------------------------------------
                  # Compute model along with best hyper-parameter
                  #-------------------------------------------------------------
                  status = self._model_compute(route, regresion_model_name, best_alpha)
               else :
                  status = False
            elif  "SGDRegressor" == regresion_model_name :

               #----------------------------------------------------------------
               # Search for best hyper-parameters : extract search parameters 
               # from dictionary given as parameter.
               #----------------------------------------------------------------
               dict_parameters = dict_models_parameters[regresion_model_name]
               list_alphas = dict_parameters['list_alphas']
               n_alphas = dict_parameters['n_alphas']            

               best_alpha = self._model_best_hyper_parameters(route, regresion_model_name, list_alphas, n_alphas) 
               if best_alpha is not None :            
                  #-------------------------------------------------------------
                  # Compute model along with best hyper-parameter
                  #-------------------------------------------------------------
                  status = self._model_compute(route, regresion_model_name, best_alpha)
               else :
                  status = False
            else : 
               print("\n*** WARNING : Estimator not supported : \n"+str(regresion_model_name))
            if status is False :
               #----------------------------------------------------------------
               # Update dictionary of routes where model can't be computed
               #----------------------------------------------------------------
               list_route_error = dict_model_route_error[regresion_model_name]           
               list_route_error.append(route)
               dict_model_route_error[regresion_model_name] = list_route_error
               # Process next linear model 
            t1 = time.time()
            print("Route="+str(route)+" : "+str(i_route)+"/"+str(nb_route)+": Elapsed time = %0.2F" %(t1-t0))

      #-------------------------------------------------------------------------
      # Store lists to be used in LinearDelayPredictor object
      #-------------------------------------------------------------------------
      self._oLinearDelayPredictor.dict_model_route_error = dict_model_route_error.copy()
      self._oLinearDelayPredictor._list_excluded = self._list_excluded.copy()
      self._oLinearDelayPredictor._list_periodic_feature = self._list_periodic_feature.copy()
      
   #----------------------------------------------------------------------------

   
         
#-------------------------------------------------------------------------------

