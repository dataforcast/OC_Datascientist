import pandas as pd
import numpy as np
import time

from scipy import sparse

import p5_util
import p6_util

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
class P6_PostClassifier() :
    '''This class implements a POST classifier model.
    It allows to provide a list of suggested tags when a post is submitted.
    '''

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def __init__(self, path_to_model=None) :
        if path_to_model is not None :
            self._path_to_model = path_to_model
        else :
            self._path_to_model = str()

        self._dict_model = dict()
        self.is_verbose = True
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    # 
    #---------------------------------------------------------------------------
    def strprint(self, mystr):
        '''Encapsulation of print function.
        
        If flag is_verbose is fixed to True, then print takes place.

        Input :
        * mystr : string to be printed.

        Output : none

        '''
        if self.is_verbose is True:
            print(mystr)
        else:
            pass
        return
    #---------------------------------------------------------------------------
    
    
    #---------------------------------------------------------------------------
    # Properties
    #---------------------------------------------------------------------------
    def _get_path_to_model(self) :
      return self._path_to_model
    
    path_to_model = property(_get_path_to_model)
    #---------------------------------------------------------------------------
    
    
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def show(self):
        '''Show attributes from class.
        '''
        is_verbose_save = self.is_verbose
        self.is_verbose = True
        self.strprint("\n  ")
        self.strprint("Verbose  ................: "+str(is_verbose_save))
        self.is_verbose = is_verbose_save

        self.strprint("Path model name  ........: "+str(self._path_to_model))
        for model_name in self._dict_model.keys():
            self.strprint("Model name ..............: "+str(model_name))
            self.strprint("Model ...................: "+str(self._dict_model[model_name]))
            self.strprint('')
        
        return
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def load_model_from_file_name(self, model_file_name) :
        ''' Read dumped model from model file name given as parameter.
        Once loaded, model is inserted in model list.
        If model name is still present in list then it is removed from list.
        
        '''
        if(model_file_name is None or 0 == len(model_file_name)):
            print("*** ERROR : wrong model name= "+str(model_file_name))
            return
        
        model_name, file_extension = model_file_name.split('.')

        if('dump' != file_extension) :
            print("*** ERROR : malformed model name= "+str(model_file_name)\
            +" File model name extension must be 'dump'")
            return
        
        model = p5_util.object_load(self._path_to_model+'/'+model_file_name)
        
        self._dict_model[model_name] = model
        
        return
    #---------------------------------------------------------------------------
    

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def suggest(self, df_post) :
        pass
    #---------------------------------------------------------------------------
#-------------------------------------------------------------------------------        

