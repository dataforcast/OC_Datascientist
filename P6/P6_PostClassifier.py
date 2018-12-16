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
        self._dict_vectorizer = dict()
        self.is_verbose = True
        self._df_post = pd.DataFrame()
        self._ser_post= pd.Series()
        self._nb_top_words = 10
        self._vectorizer_name=str()
        self._model_name=str()
        self._LIST_VECTORIZER=['BOW','TFIDF']
        self._LIST_MODEL=['LDA']
        self._dict_lda_topic = dict()
        
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
    
    def _set_model_name(self,model_name) :
        self._model_name = model_name

    def _get_model_name(self) :
        return self._model_name

    def _set_vectorizer_name(self,vectorizer_name) :
        self._vectorizer_name = vectorizer_name
        
    def _get_vectorizer_name(self) :
        return self._vectorizer_name

    path_to_model = property(_get_path_to_model)
    model_name = property(_get_model_name, _set_model_name)
    vectorizer_name = property(_get_vectorizer_name, _set_vectorizer_name)
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
        if(0<len(self._dict_model)) :
            for model_name in self._dict_model.keys():
                self.strprint("Model name ..............: "+str(model_name))
                self.strprint("Model ...................: "\
                +str(self._dict_model[model_name]))
                self.strprint('')
                
        if(0<len(self._dict_vectorizer)) :
            for model_name in self._dict_vectorizer.keys():
                self.strprint("Vectorizer name ..............: "\
                +str(model_name))
                self.strprint("Vectorizer ...................: "\
                +str(self._dict_vectorizer[model_name]))
                self.strprint('')

        if(0<len(self._df_post)) :
            self.strprint("Title ...................: "+self._df_post.Title[0])
            self.strprint("Body  ...................: "+self._df_post.Body[0])
        self.strprint("Nb top words ............: "+str(self._nb_top_words))
        if(0 < len(self._model_name)):
            self.strprint("Current model name  .....: "+str(self._model_name))
        if(0 < len(self._vectorizer_name)):
            self.strprint("Current vectorizer name .: "\
            +str(self._vectorizer_name))
        if( 'LDA' == self._model_name ):
            self.strprint("LDA topics ..............: "\
            +str(len(self._dict_lda_topic)))
        
        return
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _load_dumped_model(self, model_name, model_file_name,model_type):
        '''Read a model from a dumped file and store the model into a 
        dictionary structures as follwing : {'model_name':model}
        
        Input : 
            * model_name : name of the model (LDA, TFIDF,...)
            * model_file_name : name of the file model is dumped in.
            * model_type : type of model, used for storing.
            Supported model type : CLASSIFIER, VECTORIZER
        Output
            * dictionary holding the loaded dumped model.
        '''
        
        #-----------------------------------------------------------------------
        # Checking input
        #-----------------------------------------------------------------------
        if(model_file_name is None or 0 == len(model_file_name)):
            print("*** ERROR : wrong model name= "+str(model_file_name))
            return        

        #-----------------------------------------------------------------------
        # File name is checked regarding  extension.
        #-----------------------------------------------------------------------
        core_name, file_extension = model_file_name.split('.')
        if('dump' != file_extension) :
            print("*** ERROR : malformed model name= "+str(model_file_name)\
            +" File model name extension must be 'dump'")
            return
        
        #-----------------------------------------------------------------------
        # Model is loaded considering file name.
        #-----------------------------------------------------------------------
        model = p5_util.object_load(self._path_to_model+'/'+model_file_name)

        #-----------------------------------------------------------------------
        # Loaded model is stored depending type of model.
        #-----------------------------------------------------------------------
        if 'CLASSIFIER' == model_type:
            self._dict_model[model_name] = model
        elif 'VECTORIZER' == model_type:
            self._dict_vectorizer[model_name] = model
        else :
            print("***ERROR : Model type= "+str(model_type)+" NOT SUPPORTED!")            
        return
    #---------------------------------------------------------------------------
        
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def load_model_from_file_name(self, model_name, model_file_name\
    , vectorizer_name, vectorizer_file_name) :
        ''' Read dumped model from model file name given as parameter.
        Once loaded, model is inserted in model list.
        If model name is still present in list tshen it is removed from list.
        
        '''
        self._load_dumped_model(vectorizer_name, vectorizer_file_name\
        , 'VECTORIZER')
        self._load_dumped_model(model_name, model_file_name, 'CLASSIFIER')
        
                
        return
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def update_model(self):
        feature_names \
        = self._dict_vectorizer[self._vectorizer_name].get_feature_names()
        
        for model_name, model in self._dict_model.items():
            if model_name == 'LDA' :
                model = self._dict_model[model_name]
                self._dict_lda_topic \
                    = p6_util.p6_lda_display_topics(model\
                    , feature_names, self._nb_top_words)
            else :
                print("*** ERROR : model name="+str(model_name)\
                +" not yet implemented!")
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _standardization(self) :
        if( 0 < len(self._df_post)):
            #-------------------------------------------------------------------
            # POST is standadardized
            #-------------------------------------------------------------------
            post_id = 0
            body  = self._df_post.Body.iloc[post_id]
            title = self._df_post.Title.iloc[post_id]
            post  = body+title
            self._ser_post = p6_util.p6_str_standardization(post)


        else :
            pass
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def suggest(self, body,title) :
        ''' Creates a dataframe structured in order to be processed by p6_util 
        functions.
        '''
        post = {'Body': [body], 'Title': [title]}

        self._df_post=pd.DataFrame(data=post) 
        self._standardization()
        
        list_post_word = self._ser_post.iloc[0].split()

        dict_topic_result \
        = p6_util.p6_lda_get_topic_from_list_word(self._dict_lda_topic\
        , list_post_word)
    
        #---------------------------------------------------------------------------
        # Dictionary is converted into a dataframe with 2 columns : Count and Words.
        #---------------------------------------------------------------------------
        df_topic_result = pd.DataFrame.from_dict( dict_topic_result, orient='index')
        df_topic_result.rename(columns={0:'Count', 1:'Words'}, inplace=True)
        
        #---------------------------------------------------------------------------
        # Get topics having count >= 1
        #---------------------------------------------------------------------------
        df_topic_result_pos = df_topic_result[df_topic_result.Count>=1]

        #---------------------------------------------------------------------------
        # Building the list of unique words belonging to LDA topics.
        #---------------------------------------------------------------------------
        list_all_word = list()
        for list_word in df_topic_result_pos.Words.tolist() :
            list_all_word += [word for word in list_word]
            
        list_word_result = list(set(list_all_word))

        return list_word_result
    #---------------------------------------------------------------------------
#-------------------------------------------------------------------------------        

