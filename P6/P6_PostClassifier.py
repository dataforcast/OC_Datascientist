import pickle 

import numpy as np
import pandas as pd
import numpy as np
import time
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from scipy import sparse

import p5_util
import p6_util

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def load_dumped(fileName=None):
    ''''This class method allows to load a dumped object of type 
    P6_PostClassifier
    '''
    if fileName is None :
       dumpFileName = 'oP6_PostClassifier.dump'
       fileName = './data/' + dumpFileName
    else :
      pass
    print(fileName)
    oP6_PostClassifier = None
    try:
        with open(fileName, 'rb') as (dataFile):
            oUnpickler = pickle.Unpickler(dataFile)
            oP6_PostClassifier = oUnpickler.load()
    except FileNotFoundError:
        print('\n*** ERROR : file not found : ' + fileName)

    
    return oP6_PostClassifier
#-------------------------------------------------------------------------------

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
        self._nb_top_words = 6
        self._vectorizer_name=str()
        self._model_name=str()
        self._LIST_VECTORIZER=['BOW','TFIDF']
        self._LIST_MODEL=['LDA','KRR']
        self._dict_lda_topic = dict()
        self._list_tag_ref = list()
        self._df_validation = pd.DataFrame()
        
        
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

    def _get_list_tag_ref(self):
        return self._list_tag_ref
        
    def _set_list_tag_ref(self, list_tag_ref):
        if list_tag_ref is not None :
            if 0 < len(list_tag_ref) :
                self._list_tag_ref = list_tag_ref.copy()
                
    def _get_df_validation(self) :
        return self._df_validation
    def _set_df_validation(self, df) :
        self._df_validation = df.copy()
        
    
    
    path_to_model = property(_get_path_to_model)
    model_name = property(_get_model_name, _set_model_name)
    vectorizer_name = property(_get_vectorizer_name, _set_vectorizer_name)
    list_tag_ref = property(_get_list_tag_ref , _set_list_tag_ref)
    df_validation = property(_get_df_validation , _set_df_validation)
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

        if(0 < len(self._model_name)):
            self.strprint("Current model name  .....: "+str(self._model_name))
        if(0 < len(self._vectorizer_name)):
            self.strprint("Current vectorizer name .: "\
            +str(self._vectorizer_name))
        if( 'LDA' == self._model_name ):
            self.strprint("LDA topics ..............: "\
            +str(len(self._dict_lda_topic)))
        
        self.strprint("List TAG ref ............: "\
        +str(len(self._list_tag_ref)))
        self.strprint("Max suggested TAG .......: "\
        +str(self._nb_top_words))
        self.strprint("Validation dataset ......: "\
        +str(len(self._df_validation.shape)))

        
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
        print(model_file_name)
        core_name, file_extension = model_file_name.split('.dump')
        if(0== len(core_name)) :
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
    , vectorizer_name=None, vectorizer_file_name=None) :
        ''' Read dumped model from model file name given as parameter.
        Once loaded, model is inserted in model list.
        If model name is still present in list tshen it is removed from list.
        
        Input : 
            * model_name : classifier model name 
            * model_file_name : file name in which classifier is dumped.
            * vectorizer_name : vectorizer name. May be None.
            * vectorizer_file_name : file name in which vectorizer is dumped.
    
        
        '''
        if vectorizer_file_name is not None :
            self._load_dumped_model(vectorizer_name, vectorizer_file_name\
            , 'VECTORIZER')
            self._vectorizer_name=vectorizer_name
    
        self._load_dumped_model(model_name, model_file_name, 'CLASSIFIER')
        
        #-----------------------------------------------------------------------
        # When LDA model is used, then LDA dictionary is upadted.
        #-----------------------------------------------------------------------
        if True :
            if model_name == 'LDA' :
                print("Updating LDA dict...")
                model = self._dict_model[model_name]
                feature_names \
                = self._dict_vectorizer[self._vectorizer_name].get_feature_names()

                self._dict_lda_topic \
                    = p6_util.p6_lda_display_topics(model\
                    , feature_names, self._nb_top_words)
            
        
                
        return
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def update_model_deprecated(self):
        '''In case of LDA model, dictionary of topics is built.
        '''
        feature_names \
        = self._dict_vectorizer[self._vectorizer_name].get_feature_names()
        
        for model_name, model in self._dict_model.items():
            if model_name == 'LDA' :
                model = self._dict_model[model_name]
                self._dict_lda_topic \
                    = p6_util.p6_lda_display_topics(model\
                    , feature_names, self._nb_top_words)
            else :
                #print("*** ERROR : model name="+str(model_name)\
                #+" not yet implemented!")
                pass
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
    def _suggest_lda(self, list_post_word):
        '''LDA implementation for TAG suggestions.
        Input : 
            * list_post_word : list of words issued from a POST.
        Output : 
            * list_tag_suggested : list of suggested TAG.
        '''

        dict_topic_result \
        = p6_util.p6_lda_get_topic_from_list_word(self._dict_lda_topic\
        , list_post_word)

        #-----------------------------------------------------------------------
        # Dictionary is converted into a dataframe with 2 columns : 
        # Count and Words.
        #-----------------------------------------------------------------------
        df_topic_result = pd.DataFrame.from_dict( dict_topic_result, orient='index')
        df_topic_result.rename(columns={0:'Count', 1:'Words'}, inplace=True)

        #-----------------------------------------------------------------------
        # Get topics having count >= 1
        #-----------------------------------------------------------------------
        df_topic_result_pos = df_topic_result[df_topic_result.Count>=1]

        #-----------------------------------------------------------------------
        # Building the list of unique words belonging to LDA topics.
        #-----------------------------------------------------------------------
        list_all_word = list()
        for list_word in df_topic_result_pos.Words.tolist() :
            list_all_word += [word for word in list_word]
            
        list_tag_suggested = list(set(list_all_word))
        return list_tag_suggested
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _suggest_krr(self, list_post_word):
        '''KRR implementation for TAG suggestions.
        Input : 
            * list_post_word : list of words issued from a POST.
        Output : 
            * list_tag_suggested : list of suggested TAG issued from KRR model.
        '''
        X = self._dict_vectorizer['TFIDF'].transform(self._ser_post)
        list_tag = self._dict_model['KRR'].predict(X)
        

        X = np.array(list_tag[0])
        B = -np.sort(-X)
        C=B[:self._nb_top_words]
        arr_index = np.where(X>=C[self._nb_top_words-1])[0]
        
        list_tag_suggested = list()
        for col in arr_index:
            list_tag_suggested.append(self._list_tag_ref[col])
        return list_tag_suggested
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _suggest_logreg(self, list_post_word):
        '''Logistic Regression implementation for TAG suggestions.
        Input : 
            * list_post_word : list of words issued from a POST.
        Output : 
            * list_tag_suggested : list of suggested TAG issued from model.
        '''
        X = self._dict_vectorizer['TFIDF'].transform(self._ser_post)

        y_pred = self._dict_model['LogReg'].predict(X)
        print(y_pred.shape)
        dict_row_col_true = p6_util.p6_get_dict_row_col_from_csrmatrix(y_pred)
        

        list_tag_suggested = list()
        if False :
            if 0 < len(list_tag) :
                X = np.array(list_tag[0])
                B = -np.sort(-X)
                C=B[:self._nb_top_words]
                arr_index = np.where(X>=C[self._nb_top_words-1])[0]
                

                for col in arr_index:
                    list_tag_suggested.append(self._list_tag_ref[col])
        return list_tag_suggested
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def suggest(self, body, title, tags) :
        ''' This is the entry point for all implemented models allowing to 
        suggest TAG frol a given POST.
        A POST is formed with body and title given s parameters function.
        
        A dataframe is created and structured in order to be processed by 
        p6_util functions.
        
        Input :
            * body : detailed POST description.
            * title : title of POST.
        Output
            * list of TAGs to be suggested.
        '''
        list_tag_suggested = list()
        list_tag_suggested_fw = list()
        #-----------------------------------------------------------------------
        # POST is standardized
        #-----------------------------------------------------------------------
        post = {'Body': [body], 'Title': [title]}

        self._df_post=pd.DataFrame(data=post) 
        self._standardization()
        
        list_post_word = self._ser_post.iloc[0].split()
        
        #-----------------------------------------------------------------------
        # TAG are suggested from prediction model 
        #-----------------------------------------------------------------------
        if self._model_name == 'LDA' :
            list_tag_suggested = self._suggest_lda(list_post_word)
            if True :
                list_tag_fw = list()
                for word in list_tag_suggested :

                    list_tuple_score = process.extract(word, self._list_tag_ref)

                    list_tag_suggested_fw += [tuple_score[0] for tuple_score \
                    in list_tuple_score if tuple_score[1] >= 95]
                list_tag_suggested_fw = list(set(list_tag_suggested_fw))

        elif self._model_name == 'KRR' :
            list_tag_suggested = self._suggest_krr(list_post_word)       
        elif  self._model_name == 'LogReg' :
            list_tag_suggested = self._suggest_logreg(list_post_word)               
        else :
            print("*** ERROR : model name= "+str(self._model_name)+" Not yet implemented!")        

        list_assigned_tags \
        = p6_util.clean_marker_text(tags,leading_marker='<', trailing_marker='>')
        
        return list_tag_suggested, list_tag_suggested_fw, list_assigned_tags
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def process_post(self, post_id):
        is_available = False
        if post_id is None :
            while is_available is False:
                df_sample = self._df_validation.sample()
                body = df_sample.Body.iloc[0]
                if 200 <= len(body) :
                    title=df_sample.Title.iloc[0]
                    tag_a = df_sample.Tags.iloc[0]
                    is_available = True
        else :
            body = self._df_validation.Body.iloc[post_id]
            title = self._df_validation.Title.iloc[post_id]
            tag_a = self._df_validation.Tags.iloc[post_id]

        list_tag_suggested, list_tag_suggested_fw, list_assigned_tags\
        =self.suggest(body, title, tag_a)

        return list_tag_suggested, list_tag_suggested_fw, list_assigned_tags, body
    #---------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------        

