# uncompyle6 version 3.2.3
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.4 |Anaconda, Inc.| (default, Jan 16 2018, 18:25:17) 
# [GCC 7.2.0]
# Embedded file name: /home/bangui/Dropbox/Perso/Formation/Openclassroom/Datascientist/P4/heroku/LinearDelayPredictor.py
# Compiled at: 2018-06-18 14:27:09
# Size of source mod 2**32: 40146 bytes
import pandas as pd 
import numpy as np 
import hashlib 
import pickle 
import time 
import scipy 
import zlib
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def day_of_week_to_string(day_of_week):
    """Converts day of week number given as a parameter into a string formated as 
    one of the followings strings : MON, TUE, WED, THU, FRI, SAT, SUN, --
    """
    day_of_week = int(day_of_week)
    if 1 == day_of_week:
        str_day_of_week = 'MON'
    else:
        if 2 == day_of_week:
            str_day_of_week = 'TUE'
        else:
            if 3 == day_of_week:
                str_day_of_week = 'WED'
            else:
                if 4 == day_of_week:
                    str_day_of_week = 'THU'
                else:
                    if 5 == day_of_week:
                        str_day_of_week = 'FRI'
                    else:
                        if 6 == day_of_week:
                            str_day_of_week = 'SAT'
                        else:
                            if 7 == day_of_week:
                                str_day_of_week = 'SUN'
                            else:
                                str_day_of_week = '--'
    return str_day_of_week
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def convert_mn_to_hour(mn) :
    """Convert string hour formated as HHMM into hours ranged from 0 to 23"""
    hour = int (mn/60)
    return hour
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def day_or_month_to_string(day_or_month):
    """Returns a string formated as XX
    Input value may be a day or a month.
    """
    str_day_or_month = str(day_or_month)
    if 1 == len(str_day_or_month):
        str_day_or_month = '0' + str_day_or_month
    return str_day_or_month
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def crs_to_string(crs_value):
    """Returns a string formated as following : HH:MM"""
    str_value = str(int(crs_value))
    str_len = len(str_value)
    str_hour = ''
    str_min = ''
    hour_delimiter = str_len - 2
    if 0 >= hour_delimiter:
        str_min = str_value[:str_len]
        str_hour = '0'
    else:
        str_min = str_value[hour_delimiter:str_len]
        str_hour = str_value[:hour_delimiter]
        if 1 == len(str_hour):
            str_hour = '0' + str_hour
    str_crs = str_hour + ':' + str_min
    return str_crs
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def json_process(df_selected):
    """Returns a json like structure from dataframe given as parameter.
    
    The returned json structure is formated according following template: 
    {
    "_results": [
    { "route": "645657", "origin": "Los Angeles", "destination": "Tempa", "dep":"MM-DD hh-mn"
    ,"expected": "MM-DD hh-mn", "evaluated":"mm","measured":mm}
    ]
    }      
    
    """
    json_result = '{\n'
    json_result += '\t "_results":[\n'
    for flight_id in df_selection.index.tolist():
        month = df_selection.loc[flight_id].MONTH
        json_result += '\t\t{ "route": "' + str(route) + '",' + '"name": "' + movie_title + '"' + '   },\n'

    json_result = json_result[:-1]
    json_result = json_result[:-1]
    json_result += '\n\t]\n}'
    return json_result
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def convert_integer_to_sin(int_value, **kwargs):
    """Convert integer value given as parameter into a value issue from cosinus.
    Input :
        int_value : integer to be converted
    Output :
        sinus value
    """
    PI = np.pi
    min_value = kwargs['min_value']
    max_value = kwargs['max_value']
    int_value -= min_value
    max_value -= min_value
    teta = int_value * PI / max_value
    return np.sin(teta)
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_identity(value):
    """This method is used as a callback to be applied on users parameters when 
    delay is predicted.
    It returns same value then the one given as parameter.
    """
    return value
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def week_of_month(day_of_month):
    """Assigns a number for each one of the raw as following : 
       For DAY_OF_MONTH in ] 0,  7] : 1
       For DAY_OF_MONTH in ] 7, 14] : 2
       For DAY_OF_MONTH in ]14, 21] : 3
       For DAY_OF_MONTH in ]21, 28] : 4
       For DAY_OF_MONTH in ]28, ...]: 5
    """
    if 0 < day_of_month:
        if day_of_month <= 7:
            return 1
        if 7 < day_of_month:
            if day_of_month <= 14:
                return 2
            if 14 < day_of_month:
                if day_of_month <= 21:
                    return 3
                if 21 < day_of_month:
                    if day_of_month <= 28:
                        return 4
                    if 28 < day_of_month:
                        return 5
                    return 0
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_convert_floathour_to_mn(float_value):
    """ Converts hour into minutes.
    
    Input :
        float_value : 1 to 4 digits values in such format : x.0 --> number of mn
    Output :
        mn : number of minutes 
    """

    str_value = str(int(float_value))
    str_len = len(str_value)
    hour_delimiter = str_len - 2
    if 0 >= hour_delimiter:
        str_min = str_value[:str_len]
        str_hour = '0'
    else:
        str_min = str_value[hour_delimiter:str_len]
        str_hour = str_value[:hour_delimiter]
    try:
        mn = int(str_hour) * 60 + int(str_min)
    except ValueError as valueError:
        print("*** ERROR : cb_convert_floathour_to_mn() : input value can't be converted : " + str(float_value))

    return mn
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_convert_floathour_to_sin(float_value, max_value=1439):
    """ Apply sinus fonction to float given as parameter.
    
    Input :
        float_value : 1 to 4 digits values in such format : x.0 --> xxxx.0
    Output :
        sinus value
    """
    str_value = str(int(float_value))
    str_len = len(str_value)
    hour_delimiter = str_len - 2
    if 0 >= hour_delimiter:
        str_min = str_value[:str_len]
        str_hour = '0'
    else:
        str_min = str_value[hour_delimiter:str_len]
        str_hour = str_value[:hour_delimiter]
    try:
        mn = int(str_hour) * 60 + int(str_min)
    except ValueError as valueError:
        print("*** ERROR : cb_convert_floathour_to_sin() : input value can't be converted : " + str(float_value))

    teta = mn * np.pi / max_value
    return np.sin(teta)
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def load_dumped(fileName=None):
    """This class method allows to load a dumped object of type 
    LinearDelayPredictor
    """
    if fileName is None :
       dumpFileName = 'oLinearDelayPredictor.dump'
       fileName = './data/' + dumpFileName
    else :
      pass

    oLinearDelayPredictor = None
    try:
        with open(fileName, 'rb') as (dataFile):
            oUnpickler = pickle.Unpickler(dataFile)
            oLinearDelayPredictor = oUnpickler.load()
    except FileNotFoundError:
        print('\n*** ERROR : file not found : ' + fileName)

    if oLinearDelayPredictor._df_route_compressed is not None:
       oLinearDelayPredictor._df_route = \
       pickle.loads(zlib.decompress(oLinearDelayPredictor._df_route_compressed))
       del(oLinearDelayPredictor._df_route_compressed)

    if oLinearDelayPredictor._df_user_test_compressed is not None:
       oLinearDelayPredictor._df_user_test = \
       pickle.loads(zlib.decompress(oLinearDelayPredictor._df_user_test_compressed))
       del(oLinearDelayPredictor._df_user_test_compressed)
    
    if oLinearDelayPredictor._dict_model_route_compressed is not None:
     oLinearDelayPredictor._dict_model_route \
     = pickle.loads(zlib.decompress(oLinearDelayPredictor._dict_model_route_compressed))
     del (oLinearDelayPredictor._dict_model_route_compressed)
    
    #---------------------------------------------------------------------------
    # Remove duplicated indexes :
    #---------------------------------------------------------------------------
    if oLinearDelayPredictor._df_user_test is not None:
       oLinearDelayPredictor._df_user_test = \
       oLinearDelayPredictor._df_user_test[~oLinearDelayPredictor._df_user_test.index.duplicated()]
    
    return oLinearDelayPredictor
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def print_delays(dict_result_delay):
    """Dislay results into dictionary given as parameter """
    print('\n----- Results--------')
    if dict_result_delay is not None:
        for model_name in dict_result_delay.keys():
            result_delay = dict_result_delay[model_name]
            if result_delay is not None:
                print('Model = %s : Delay for flight = %1.2F' % (model_name, result_delay))
            else:
                print(('Model = {} : Delay for flight = {}').format(model_name, result_delay))

    else:
        print('\n*** ERROR : no result into dictionary!\n')
    print('----- Results--------\n')
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
class LinearDelayPredictor:
    """ This class implements all attributes and linear regression models 
    computed into P4DataModel class that are required for making a delay 
    estimation.
    There is one regression model per route.
    A route matches with a tuple of (ORIGIN_CITY_NAMRE,DEST_CITY_NAME).
    """

    #----------------------------------------------------------------------------
    #
    #----------------------------------------------------------------------------
    def __init__(self, path_to_data=None):
        if path_to_data is None:
            path_to_data = './data/'
        self._path_to_data = path_to_data
        self._model_name = 'LinearRegression'
        self._dict_model_route = dict()
        self._dict_model_route_compressed = dict()
        self._df_route = pd.DataFrame()
        self._dict_model_route_error = dict()
        self._df_route_compressed = None
        self._df_user_test_compressed = None
        self._list_periodic_feature = list()
        self._list_excluded = list()
        self._list_quantitative_identity = list()
        self._list_quantitative_cos = list()
        self._df_user_test = pd.DataFrame()
        self._dict_feature_processor = dict()
        self._dict_climat = dict()
        self._fract_user_test = -1
    #----------------------------------------------------------------------------

    #----------------------------------------------------------------------------
    #  PROPERTIES
    #----------------------------------------------------------------------------
    def _get_dict_model_route(self):
        return self._dict_model_route.copy()

    def _set_dict_model_route(self, dict_model_route):
        self._dict_model_route = dict_model_route.copy()

    def _get_model_name(self):
        return self._model_name

    def _set_model_name(self, model_name):
        self._model_name = model_name

    def _set_df_route(self, df_route):
        self._df_route = df_route.copy()

    def _get_dict_model_route_error(self):
        return self._dict_model_route_error

    def _set_dict_model_route_error(self, dict_model_route_error):
        self._dict_model_route_error = dict_model_route_error.copy()

    def _get_dict_climat(self):
      return self._dict_climat.copy()

    def _set_dict_climat(self, dict_climat):
      if dict_climat is not None :
         self._dict_climat =  dict_climat.copy()

    def _get_fract_user_test(self):
      return self._fract_user_test
      
    def _set_fract_user_test(self, fract_user_test):
      self._fract_user_test = fract_user_test
      
    model_name = property(_get_model_name, _set_model_name)
    df_route = property(_set_df_route)
    dict_model_route_error = property(_get_dict_model_route_error, _set_dict_model_route_error)
    dict_model_route = property(_get_dict_model_route, _set_dict_model_route)
    dict_climat = property(_get_dict_climat, _set_dict_climat)
    fract_user_test = property(_get_fract_user_test, _set_fract_user_test)

    #---------------------------------------------------------------------------
    #  
    #---------------------------------------------------------------------------
    def get_random_flights(self, nb_flights=10):
        """Returns a dafarame with nb_flights rows randomly selected.
        """
        df_selection = self._df_user_test.sample(nb_flights)
        return df_selection
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #  
    #---------------------------------------------------------------------------
    def get_random_flights_not_validated(self, nb_flights=10):
        """Returns a dafarame with nb_flights rows randomly selected.
        All rows from returned dataframe belongs to a valid route.
        Returned dataframe contains routes from which at least one model 
        name is available
        """
        #-------------------------------------------------------------
        # Filtering df_route from _dict_model_route and 
        # _dict_model_route_error
        #-------------------------------------------------------------
        df_route = pd.DataFrame()
        df_concat = pd.DataFrame()
        
        # Get routes that are available for some model names
        for model_name in self._dict_model_route_error.keys() :
            #Get error routes for this model name
            list_route_error = self._dict_model_route_error[model_name]

            # For all routes recorded in this object
            for route in self._dict_model_route.keys()  :
                if route not in list_route_error:
                    # Get route from this model
                    df_concat = self._df_route[self._df_route.HROUTE==route]
                    df_route = pd.concat([df_route, df_concat],axis=0)
        
        #-------------------------------------------------------------
        # Filtering df_user_test from df_route index
        #-------------------------------------------------------------
        df_user_test = pd.DataFrame()
        for index in df_route.index :
            ser_temp = self._df_user_test.loc[index]
            df_user_test = pd.concat([df_user_test, ser_temp],axis=1)

        df_user_test = df_user_test.transpose().copy()
        nb_rows = min(nb_flights, df_user_test.shape[0])
        if 0 == df_user_test.shape[0] :
            return None
        df_selection = df_user_test.sample(nb_rows)
        return df_selection
    #---------------------------------------------------------------------------


    #---------------------------------------------------------------------------
    #  
    #---------------------------------------------------------------------------
    def copy(self, object):
        """ Copies attributes from object given as parameter into 
        this object."""
        
        self._path_to_data = object._path_to_data
        self._dict_model_route = object._dict_model_route.copy()
        self._model_name = object._model_name
        if object._df_route is not None:
            self._df_route = object._df_route.copy()
        self.dict_model_route_error = object.dict_model_route_error.copy()
        self._list_periodic_feature = object._list_periodic_feature.copy()
        self._list_excluded = object._list_excluded.copy()
        self._df_user_test = object._df_user_test.copy()
        self._dict_feature_processor = object._dict_feature_processor.copy()
        self.dict_climat = object.dict_climat

    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #  
    #---------------------------------------------------------------------------
    def print(self, is_route=False):
        print('\n------------------- LinearDelayPredictor ----------------------')
        print(('Current model name                   = {}').format(self._model_name))
        print(('Routes in error                      = {}').format(self.dict_model_route_error))
        print(('List of periodic features            = {}').format(self._list_periodic_feature))
        print(('List of excluded features            = {}').format(self._list_excluded))
        print(('List of quant features to cos()      = {}').format(self._list_quantitative_cos))
        print(('List of quant features to identity() = {}').format(self._list_quantitative_identity))
        print(('Features processor dictionary        = {}').format(self._dict_feature_processor))
        print(('Climatic model                       = {}').format(self.dict_climat))
        print(('Fraction of dumped data user test    = {}').format(self.fract_user_test))
        
        if is_route is True:
            for route in self._dict_model_route.keys():
                print('\n---------- Route : ' + str(route))
                dict_model = self._dict_model_route[route]
                for key in dict_model.keys():
                    print('Key name : ' + str(key))
                    if key == 'model':
                        print('Model value : ' + str(dict_model[key]))

        print('---------------------------------------------------------------\n')
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #  
    #---------------------------------------------------------------------------
    def _linear_combination(self, result_vector, regression_model):
        """ Compute a linear combination between result_vector given as 
        parameter and coefficients issues from linear regression model."""
        
        #-----------------------------------------------------------------------        
        # Use model name issued from configuration
        # DummyRegressor does not returns any value.
        #-----------------------------------------------------------------------        
        model_name = self.model_name
        if 'DummyRegressor' == model_name:
            return
        else:
            vector_raws = result_vector.shape[0]
            vector_columns = result_vector.shape[1]
            coeff_raws = regression_model.coef_.shape[0]
            coeff = regression_model.coef_.reshape(-1, 1)
            if vector_raws == coeff_raws:
                flight_delay = np.dot(coeff.T, result_vector)
            else:
                flight_delay = np.dot(result_vector, coeff)
            return flight_delay[0][0]
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #  
    #---------------------------------------------------------------------------
    def _param_vectorize(self, route, flight_param):
        """ Change flight_param value into a vector, considering route 
        given as parameter."""

        list_qualitative_value = list()
        list_quantitative_value = list()
        for key_type in flight_param.keys():
            if key_type == 'qualitative' or key_type == 'quantitative':
                dict_type_value = flight_param[key_type]
                for cb_function in dict_type_value.keys():
                    list_value = dict_type_value[cb_function]
                    for value in list_value:
                        if key_type == 'qualitative':
                            list_qualitative_value.append(cb_function(value))
                        elif key_type == 'quantitative':
                            list_quantitative_value.append(cb_function(value))
                            
        #print("\n*** list_quantitative_value = {}".format(list_quantitative_value))
        #print("\n*** list_qualitative_value = {}".format(list_qualitative_value))

        #-----------------------------------------------------------------------
        # Get values issue from climatic model : they are 2 last elements 
        # from list_qualitative_value list
        #-----------------------------------------------------------------------
        origin_state_abr = list_qualitative_value[-2]
        if origin_state_abr not in self.dict_climat.keys() :
         origin_climat_value = 3
        else :      
         origin_climat_value = self.dict_climat[origin_state_abr]
         
        dest_state_abr = list_qualitative_value[-1]
        if dest_state_abr not in self.dict_climat.keys() :
         dest_climat_value = 3
        else :      
         dest_climat_value = self.dict_climat[dest_state_abr]
        
        list_qualitative_value[-2] = origin_climat_value
        list_qualitative_value[-1] = dest_climat_value
        
        #-----------------------------------------------------------------------
        # Get a linear regression model per route.
        #-----------------------------------------------------------------------
        if route not in self._dict_model_route.keys() :
         print("\n *** ERROR : no model for route= "+str(route))
         return None
        else :
           dict_model = self._dict_model_route[route]
           encoder = dict_model['encoder']
           std_scale = dict_model['std_scale']
           X_vector = np.array(list_qualitative_value)
           X_vector = X_vector.reshape(1, -1)

        #-----------------------------------------------------------------------
        # apply scaling and encodings
        #-----------------------------------------------------------------------
        #print("\n*** X_vector= {}".format(X_vector))
        sparse_encoded = encoder.transform(X_vector)

        quantitative_array = \
        (np.array(list_quantitative_value, dtype=float)).\
        reshape(len(list_quantitative_value), -1)
        
        X_quantitative_std = std_scale.transform(quantitative_array)
        sparse_X = scipy.sparse.csr_matrix(X_quantitative_std)
        try:
            X_std = scipy.sparse.hstack((sparse_X, sparse_encoded))
        except ValueError as valueError:
            print(('*** ERROR : scipy.sparse.hstack() : \
            sparse_X.shape = {} / sparse_encoded.shape= {}').\
            format(sparse_X.toarray().shape, sparse_encoded.toarray().shape))
            return

        return X_std.toarray()
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def get_cities_route(self, route=0):
        """Returns tuple (ORIGIN_CITY_NAME, DEST_CITY_NAME) matching with route 
        value given as parameter."""
        
        origin = None
        destination = None
        if self._df_route is not None:
            if 0 < len(self._df_route):
                if route in self._df_route.HROUTE.unique().tolist():
                    df = self._df_route[self._df_route['HROUTE'] == route]
                    origin = df.ORIGIN_CITY_NAME.unique()[0]
                    destination = df.DEST_CITY_NAME.unique()[0]
                else:
                    print('*** ERROR : no cities found for route = ' + str(route))
            print('*** WARNING : dataframe for routes is empty!')
        return (origin, destination)
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _get_route(self, origin_city_name, dest_city_name):
        df1 = self._df_route[self._df_route['ORIGIN_CITY_NAME'] == origin_city_name]
        df2 = df1[df1['DEST_CITY_NAME'] == dest_city_name]
        route = df2.HROUTE.unique()[0]
        del df1
        del df2
        return route
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _forecast_delay(self, list_qualitative, list_quantitative, list_route, is_all_model=False):
        """Returns evaluated delay from given parameters.
        Parameters provided by user are transformed as a vector.
        Linear combination is applied to this vector.

        Result of this linear combination is returned as delayed arrival time in mn.
        Delay may be positive (delayed arrival) or negative (advanced arrival or on time)
        Input :
              * [month,day_of_month,day_of_week,airline_id]
           list_quantitative : contains list of quantitatives features ordered as following  :
              * [crs_dep_time]
           list_route : contains origin and destination cities :
              * [origin_city_name,dest_city_name]
           is_all_model : use all models for prediction.
        Output :
           result_delay : None when is_all_model flag i sactivated; otherwise, predicted delay
                          for model assiigned into this object.
           dict_result_delay : None when is_all_model flag is activated; otherwise, 
                               dictionary containing delay result per model.
                               Dictionary keys are models used for predictions.
        """
        #print('\n--------------- DELAY EVALUATION ----------------------\n')
        dict_result_delay = None
        result_delay = None
        dict_qualitative_value = dict()
        dict_quantitative_value = dict()
        day_of_month = list_qualitative[0]
        
        if 'WEEK_OF_MONTH' not in self._list_excluded:
            list_qualitative.append(week_of_month(day_of_month))
        if 'MONTH' in self._list_periodic_feature:
            month = list_qualitative[0]
            list_qualitative[0] = cb_convert_integer_to_sin(month, min_value=1, max_value=31)

        #-----------------------------------------------------------------------
        # No process is applied for qualitative values.
        #-----------------------------------------------------------------------
        dict_qualitative_value[cb_identity] = list_qualitative
        
        self._list_quantitative_cos = list()
        self._list_quantitative_identity = list()
        
        
        # ---------------------------------------------------------------------
        # Fill lists of features to be processed as callback.
        # TBD : remove _list_quantitative_cos and _list_quantitative_identity
        # and replace them with dict_feature_processor
        # ---------------------------------------------------------------------
        self._list_quantitative_identity = list()
        for quantitative_feature in list_quantitative:
            if 'CRS_DEP_TIME' in self._list_periodic_feature:
                self._list_quantitative_cos.append(quantitative_feature)
            else:
                #self._list_quantitative_identity.append(quantitative_feature)
                pass
        # ---------------------------------------------------------------------
        # Data are processed considering processor dictionary
        # Te Be Changed
        # ---------------------------------------------------------------------
        #for quantitative_feature in self._dict_feature_processor :
        # for cb_function in self._dict_feature_processor[quantitative_feature]:
        #    dict_quantitative_value[cb_function] = self._list_quantitative_identity
        for feature in list_quantitative:
            cb_function = self._dict_feature_processor['CRS_DEP_TIME']
            dict_quantitative_value[cb_function] = list_quantitative

        #dict_quantitative_value[cb_convert_floathour_to_sin] = self._list_quantitative_cos
        #dict_quantitative_value[cb_convert_floathour_to_sin] = self._list_quantitative_identity
        
        flight_param = {'qualitative':dict_qualitative_value, 
         'quantitative':dict_quantitative_value}
         

        flight_param['origin_city_name'] = list_route[0]
        flight_param['dest_city_name'] = list_route[1]

        #print("\n*** flight_param *** {}".format(flight_param)) 

        origin_city_name = flight_param['origin_city_name']
        dest_city_name = flight_param['dest_city_name']
        route = self._get_route(origin_city_name, dest_city_name)

        if route in self.dict_model_route_error:
            print('*** ERROR : route in error : ' + str(route) + ' Origin : ' + \
            origin_city_name + ' Destination : ' + dest_city_name)
        else:
            #print("Flight parameters = {}".format(flight_param))
            result_vector = self._param_vectorize(route, flight_param)
            if result_vector is None :
               return None, None
            #print("\n *** result_vector = {}".format(result_vector))
            dict_model = self._dict_model_route[route]
            dict_result_delay = dict()
            if is_all_model is True:
                #print("\n*** Route= "+str(route)+" Mutiple models!")
                model_name_save = self.model_name
                for key in dict_model.keys():
                    if key != 'encoder':
                        if key != 'std_scale':
                            self.model_name = key
                            #print("\n*** Route= "+str(route)+" Model name= "+str(self.model_name))
                            regression_model = None
                            list_route_error = self.dict_model_route_error[self.model_name]
                            if route in list_route_error:
                              #print("\n*** Route= "+str(route)+" In list of route error!")
                              for model_name in self.dict_model_route_error.keys():
                                  if 0 == len(self.dict_model_route_error[model_name]):
                                      self.model_name = model_name
                                      regression_model = dict_model[self.model_name]
                                      break

                            else:
                              #print("\n*** Route= "+str(route)+" not in error!")
                              regression_model = dict_model[self.model_name]
                            if regression_model is not None:
                              result_delay = self._linear_combination(result_vector, regression_model)
                            dict_result_delay[self.model_name] = result_delay

                self.model_name = model_name_save
            else:
                #print("\n*** Route= "+str(route)+" Single model= "+str(self.model_name))
                regression_model = None
                list_route_error = self.dict_model_route_error[self.model_name]
                if route in list_route_error:
                    #print("\n*** Route= "+str(route)+" in list error = {}"+format(list_route_error))
                    for model_name in self.dict_model_route_error.keys():
                        if 0 == len(self.dict_model_route_error[model_name]):
                            self.model_name = model_name
                            regression_model = dict_model[self.model_name]
                            break

                else:
                    regression_model = dict_model[self.model_name]
                    #print("\n*** Model name= {} / dict_model= {}".format(regression_model, dict_model))
                
                if regression_model is not None:
                    result_delay = self._linear_combination(result_vector, regression_model)
                    #print("\n*** result_delay= {}".format(result_delay))
                    
                dict_result_delay[self.model_name] = result_delay
            return result_delay, dict_result_delay
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def dump(self, dumpFileName=None):
        """Dump object given as parameter into a dumped file """
        if dumpFileName is None:
            dumpFileName = 'oLinearDelayPredictor.dump'
        fileName = './data/' + dumpFileName
        
        
        # ----------------------------------------------------------------------
        # Dump a random fraction of user_test dataframe. All routes required 
        # into dataframe user test are included into df_route dataframe.
        # 
        # ----------------------------------------------------------------------
        print("\n *** Fraction of user test dataframe: size= "+format(self._df_user_test.shape))
        
        if self._fract_user_test > 0 :
           rows = self._df_user_test.shape[0]
           fract_rows = int(rows*self._fract_user_test)
           self._df_user_test = self._df_user_test.sample(fract_rows).copy()

           # --------------------------------------------------------------
           # Filter rows from df_route based on index from _df_user_test
           # --------------------------------------------------------------
           nb_index = len(list(self._df_user_test.index))
           print("\n *** Filter of route dataframe based on user dataframe : "+str(nb_index)+"...")
           print("\n *** Dataframe route size= "+format(self._df_route.shape))
           df_route2 = pd.DataFrame()
           count_index = 0
           for index in list(self._df_user_test.index) :
            count_index += 1
            df_route2 = pd.concat([df_route2, self._df_route.loc[index,:]], axis=0)
            if 0 == count_index % 1000: 
               print("\n--- count index= "+str(count_index))
           #
           if 0 in df_route2.columns:
            del(df_route2[0])    
           self._df_route = df_route2.copy()
           del(df_route2)
           print("\n *** Fraction of Dataframe route size= "+format(self._df_route.shape))        
           print("\n *** Dataframe user test size= "+format(self._df_user_test.shape))

           #-----------------------------------------------------------------------
           # Reset fraction of user test dataframe to be dumped
           # ----------------------------------------------------------------------
           self._fract_user_test = -1

        #-----------------------------------------------------------------------
        # Do compress _df_route dataframe and release uncompressed resource 
        # ----------------------------------------------------------------------
        print("\n *** Route dataframe compression...")
        self._df_route_compressed = zlib.compress(pickle.dumps(self._df_route))
        self._df_route = pd.DataFrame()
        
        #-----------------------------------------------------------------------
        # Do compress _df_user_test dataframe and release uncompressed resource 
        # ----------------------------------------------------------------------
        print("\n *** User test dataframe compression...")
        self._df_user_test_compressed \
        = zlib.compress(pickle.dumps(self._df_user_test))
        self._df_user_test = pd.DataFrame()
        

        #-----------------------------------------------------------------------
        # Do compress dict_model_route and release uncompressed resource 
        # ----------------------------------------------------------------------
        print("\n *** Data route compression...")
        self._dict_model_route_compressed \
        = zlib.compress(pickle.dumps(self._dict_model_route))
        self._dict_model_route = dict()


        #-----------------------------------------------------------------------
        # Dump object...
        # ----------------------------------------------------------------------
        print("\n *** Dump object in file= "+fileName+"...")
        try :
           with open(fileName, 'wb') as (dumpedFile):
               oPickler = pickle.Pickler(dumpedFile)
               oPickler.dump(self)
        except pickle.PicklingError as picklingError :
           print("*** ERROR : dumping into "+str(fileName)+" Error= "+str(picklingError))
        finally :
           #--------------------------------------------------------------------
           # Uncompressed compressed structure.
           # Delete compressed structure.
           #--------------------------------------------------------------------
           print("\n *** Route dataframe decompression ...")
           self._df_route = pickle.loads(zlib.decompress(self._df_route_compressed))
           del (self._df_route_compressed)
           self._df_route_compressed = pd.DataFrame()

           print("\n *** User dataframe decompression ...")
           self._df_user_test = pickle.loads(zlib.decompress(self._df_user_test_compressed))
           del (self._df_user_test_compressed)
           self._df_user_test_compressed = pd.DataFrame()

           print("\n *** User dataframe decompression ...")
           self._dict_model_route \
           = pickle.loads(zlib.decompress(self._dict_model_route_compressed))
           del (self._dict_model_route_compressed)
           self._dict_model_route_compressed = dict
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def dump_empty_df(self, dumpFileName=None):
        """Dump self object given as parameter into a dumped file 
        Dataframe are assigned to None.
        """
        
        save_self = LinearDelayPredictor()
        save_self.copy(self)
        
        if dumpFileName is None:
            dumpFileName = 'oLinearDelayPredictor.dump'
        fileName = './data/' + dumpFileName
        self._df_route = pd.DataFrame()
        self._df_user_test = pd.DataFrame()
        self._dict_model_route = dict()
        try :
           with open(fileName, 'wb') as (dumpedFile):
               oPickler = pickle.Pickler(dumpedFile)
               oPickler.dump(self)
        except pickle.PicklingError as picklingError :
           print("*** ERROR : dumping into "+str(fileName)+" Error= "+str(picklingError))
        finally :
         self.copy(save_self)
         del(save_self)
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def json_selection_builder(self, df_selection):
        """Builds a json formated list of flights aimed for user selection.
        Input : 
            df_selection : a subset of dataframe issued from LinearDelayPredictor.
        Output :
            list_json_selection : a list with elements formated as json.
        """
        json_selection = '{"_select":['
        for flight_id in df_selection.index.tolist():
            json_selection += '{'   
            month = df_selection.loc[flight_id].MONTH
            day_of_month = df_selection.loc[flight_id].DAY_OF_MONTH
            day_of_week = df_selection.loc[flight_id].DAY_OF_WEEK
            carrier = df_selection.loc[flight_id].CARRIER
            origin_city_name = df_selection.loc[flight_id].ORIGIN_CITY_NAME
            dest_city_name = df_selection.loc[flight_id].DEST_CITY_NAME
            crs_dep_time = df_selection.loc[flight_id].CRS_DEP_TIME
            crs_arr_time = df_selection.loc[flight_id].CRS_ARR_TIME
            flight_number = df_selection.loc[flight_id].FL_NUM
            arr_delay = df_selection.loc[flight_id].ARR_DELAY
            route = self._get_route(origin_city_name, dest_city_name)
            str_dep_time = crs_to_string(crs_dep_time)
            str_arr_time = crs_to_string(crs_arr_time)
            str_month = day_or_month_to_string(month)
            str_day_of_month = day_or_month_to_string(day_of_month)
            str_day_of_week = day_of_week_to_string(day_of_week)
            str_departure = str_day_of_week + ' ' + str_month + '-' + str(day_of_month) + ' ' + str_dep_time
            
            json_selection += '"id":"' + str(flight_id) + '",' + '"flight":' + '"' + str(flight_number) + '",' + '"company":' + '"' + str(carrier) + '",' + '"origin":' + '"' + str(origin_city_name) + '",' + '"destination":' + '"' + str(dest_city_name) + '",' + '"departure":' + '"' + str(str_departure) + '",' + '"arrival":' + '"' + str(str_arr_time)
            json_selection += '},'

        json_selection = json_selection[:-1]
        json_selection += ']}'
        return json_selection
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _json_result_builder(self, selected_id, result_delay, dict_result_delay):
        """Returns result, json formated 
              Input : 
                 * selected_id : flight identifier (internal identifier)
                 * result_delay : value of evaluated delay
                 * dict_result_delay : values of evaluated delay for any regression model
                 implemented in this class.
              Output :
                 * result, json formated on such way : 
                 {"_result":[{"id":"365431","model":"LinearRegression","evaluated_delay":"8","measured_delay":"-14",}
        ]}
              
              """
        json_result = str()
        if result_delay is not None:
            json_result += '{"_result":[{"id":"' + str(selected_id) + '",'
            model_name = self.model_name
            if model_name == 'SGDRegressor':
               model_name ='ElasticNet'
            json_result += '"model":"' + str(model_name) + '",'
            json_result += '"model":"' + str(result_delay) + '",'
            measured_delay = dict_result_delay['measured']
            json_result += '"measured_delay":"' + str(measured_delay) + '",'
            json_result += '}]}'
        if dict_result_delay is not None:
            json_result += '{"_result":[{"id":"' + str(selected_id) + '",'
            for key in dict_result_delay.keys():
                if key == 'DummyRegressor':
                    continue
                if key == 'measured':
                    measured_delay = dict_result_delay[key]
                    json_result += '"measured_delay":"' + str(measured_delay) + '"'
                else:
                    model_name = key
                    # Changing SGDRegressor into ElasticNet if required
                    model_name2 = model_name
                    if model_name == 'SGDRegressor':
                     model_name2 ='ElasticNet'
                    evaluated_delay = dict_result_delay[model_name]
                    evaluated_delay = int(evaluated_delay)
                    json_result += '"model":"' + str(model_name2) + '",' + '"evaluated_delay":' + '"' + str(evaluated_delay) + '",'

            json_result += '}]}'
        return json_result
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    # 
    #---------------------------------------------------------------------------
    def evaluate_delay(self, selected_id):
        """This is entry point for computing delay estimation.
        Input : 
         selected_id : supposed to be provided from a user interface.
        Output :
         estimated delay returned into a json format.
        """
        #-----------------------------------------------------------------------
        # List of qualitatives values matching with selected_id
        #-----------------------------------------------------------------------
        ser_user_test = self._df_user_test.loc[selected_id, :]
        list_qualitative_features = [
         ser_user_test.AIRLINE_ID
         ,ser_user_test.MONTH
         ,ser_user_test.DAY_OF_MONTH
         ,ser_user_test.DAY_OF_WEEK
         ,ser_user_test.ORIGIN_STATE_ABR
         ,ser_user_test.DEST_STATE_ABR
         ]
         
        list_quantitative_features = [ser_user_test.CRS_DEP_TIME]
        #print("\n*** list_qualitative_features = {}".format(list_qualitative_features))
        #print("\n*** list_quantitative_features = {}".format(list_quantitative_features))
        list_route = [ser_user_test.ORIGIN_CITY_NAME, ser_user_test.DEST_CITY_NAME]

        result_delay, dict_result_delay = \
        self._forecast_delay(list_qualitative_features, \
        list_quantitative_features, list_route, is_all_model=True)

        arr_delay = ser_user_test.ARR_DELAY
        dict_result_delay['measured'] = arr_delay
        if dict_result_delay is not None:
            result_delay = None
        json_result = \
        self._json_result_builder(selected_id, result_delay, dict_result_delay)
        return json_result
    #---------------------------------------------------------------------------
        
