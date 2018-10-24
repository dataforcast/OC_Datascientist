##!/usr/bin/python
##-*- coding: utf-8 -*-

import os
import pandas as pd
import glob
import pickle 

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_directory(directory):
    """Creates a directory with name given as parameter.
    If directory already exists, then all files inside are deleted.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Directory "+directory+" created!")
    else:
        print("Directory "+directory+" Already exists!")
        list_file_name = [ f for f in os.listdir(directory) ]
        for file_name in list_file_name :
            absolute_file_name = directory+"/"+file_name
            #print(absolute_file_name)
            os.remove(absolute_file_name)
    assert os.path.isdir(directory)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_remove_stopwords(item, lang='english', mode='lower') :
    """This function removes some stopwords form item given as parameter.
    Removed stopwords are issued from 'nltk.corpus.stopwords'.
    
    """
    list_word=item.split()
    item_no_stopwords_1=[ word for word in list_word if word.lower() not in stopwords.words(lang) \
                      and not str(word).isdigit()]
    item_no_stopwords=[word for word in item_no_stopwords_1 if word.lower() not in ['cnn']]
    
    item_no_stopwords=" ".join(item_no_stopwords)
    if mode == 'upper':
        return item_no_stopwords.upper()
    elif mode =='lower':
        return item_no_stopwords.lower()
    else:
        print("*** ERROR no mode defined for word! ***")
        return None
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_remove_punctuation(item, list_char_remove=None, mode='lower') :
    tokenizer = nltk.RegexpTokenizer(r'[ a-zA-Z0-9]')
    tokenized_list = tokenizer.tokenize(item.lower())

    item_no_punctuation_1=[ char for char in item.lower() \
    if char in tokenized_list ]

    #---------------------------------------------------------------------------
    # Remove additional punctuations provided into list_char_remove
    #---------------------------------------------------------------------------
    if list_char_remove is not None:
        item_no_punctuation=[ char for char in item_no_punctuation_1 \
        if char not in list_char_remove ]
    else:
        item_no_punctuation = item_no_punctuation_1.copy()
        
    item_no_punctuation="".join(item_no_punctuation)
    if mode == 'upper':
        return item_no_punctuation.upper()
    elif mode =='lower':
        return item_no_punctuation.lower()
    else:
        print("*** ERROR no mode defined for word! ***")
        return None
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def file_write(content, directory, file_count, extension):

    file_name = directory+'/'+str(file_count)
    file_name += extension

    file_to_write = open(file_name,'wt')
    file_to_write.write(content)
    file_to_write.close()
    return
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_tfidf_vectorizer(dict_article_highlight):
    """Returns tfidf vectorizer allowing to transform a tokenized text into a vector.
    Due to the fact that highlights are a subset of articles, then, only tokenized  articles
    are used to compute tfidf vectorizer.
    
    Input : 
        * dict_article_highlight : dictionary containing tokenized articles and corresponding highlights.
        Dictionary keys are file root names.
        Dictionary values are tuple (list_tokenized_article, list_tokenized_highlight,...)
    Output :
        TFIDF vectorizer
    """
    #--------------------------------------------------------------
    # Retrieve tokenized articles from dictionary
    #--------------------------------------------------------------
    tuple_pos = 0
    
    list_tokenized_corpus=list()
    for root_file_name in dict_article_highlight.keys():
        #--------------------------------------------------------
        # Get list of tokens issued from a given file
        #--------------------------------------------------------
        list_token=dict_article_highlight[root_file_name][tuple_pos]

        #--------------------------------------------------------
        # Tokens from a file are joined into a single string
        #--------------------------------------------------------
        str_tokenized_file=' '.join(str(e) for e in list_token)

        #--------------------------------------------------------
        # String is appened into the list of strings.
        # This list compounds all strings issued from tokenized files.
        # This is the tokenized corpus.
        #--------------------------------------------------------
        list_tokenized_corpus.append(str_tokenized_file)
    
    #--------------------------------------------------------
    # TF-IDF weights are computed for the whole tokenized 
    # corpus.
    #--------------------------------------------------------
    vectorizer=TfidfVectorizer(norm="l2")
    vectorizer.fit_transform(list_tokenized_corpus)
    return vectorizer

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def dict_plot_frequency(dict_freq, doc_type, query=None):
    """Plot word frequencies onto a bar diagram from words into a dictionary.
    
    Input : 
        * dict_freq : nltk.probability.FreqDist object type; a kind of Python
        dictionary containing words as keys and words frequencies as values.
        
        * doc_type : string used for title.
        
        * query : used to restrict (keys, values) to be displayed.
        dict_freq is converted as a DataFrame object. Then query method is
        applied on such object.
    """
        
    if dict_freq is not None:
        df = pd.DataFrame.from_dict(dict_freq, orient='index')
        if 'Freq' not in df.columns:
            df.rename(columns={0:'Freq'}, inplace=True)
            df.sort_values(by = 'Freq', inplace=True, ascending=False) 
        else:
            pass
        title = "Words frequency for : "+doc_type
        if query is None:
            z_=df.plot(kind='bar', title=title, color='orange')
        else:
            title += " / "+query
            z_=df.query(query).plot(kind='bar', title=title, color='orange')
    else:
        pass
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_root_name(absolute_file_name, file_extension=".story"):
    """Returns root name from a file name ended with a file extension.
    Input:
        * absolute_file_name : file name preceeded with a path composed with characters '/'
        * file_extension : string ending a filename.
    """
    list_splited = absolute_file_name.split('/')
    root_name = list_splited[-1]
    root_name = root_name.split(file_extension)[0]
    return root_name
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def read_cnn_corpora(data_path, p_restriction=None, read_count=None):
    """Read all files from data_path.
    If p_restriction is not None, then a restricted list of file is read.
    If read_count is not None, then read_count files are read.
    
    Articles are separated from highlights. Hilights from each article 
    are compounded.
    
    Output :
        * dict_article : dictionary containing all read articles. 
        Dictionary keys are root-name from files from which article are issued.

        * dict_highlight : dictionary containing all compounded highlights. 
        Dictionary keys are root-name from files from which article are issued.
    """
    #-----------------------------------------------------------
    # Used to restrict the read to some files.
    #-----------------------------------------------------------
    iglob__ = None
    if p_restriction is not None:
        iglob__ = glob.iglob(data_path+"/"+p_restriction+".story")
    else:
        iglob__ = glob.iglob(data_path+"/*.story")
            
    
    #list_article=list()
    #list_highlight=list()
    
    dict_article=dict()
    dict_highlight=dict()
    
    file_count=0

    for file_name in iglob__:
    
        #------------------------------------------------------
        # Get root-name from file name; it is used as key for 
        # dictionaries.
        #------------------------------------------------------
        file_id = get_root_name(file_name)

        #------------------------------------------------------
        # Read content from each file
        #------------------------------------------------------
        file_to_read = open(file_name,'rt') 
        data_file = file_to_read.read()
        file_to_read.close()    

        #------------------------------------------------------
        # Content article is separated from highlights inside
        # article.
        #------------------------------------------------------
        list_highlight_ = data_file.rsplit("\n\n@highlight")

        #------------------------------------------------------
        # Article is extracted from list
        #------------------------------------------------------
        article = list_highlight_[0]

        #------------------------------------------------------
        # Article is removed from highlights list; 
        # list_highlight_ now contains highlights only.
        #------------------------------------------------------
        list_highlight_.remove(article)
        
        #------------------------------------------------------
        # Highlights list is compounded into a single sentance
        #------------------------------------------------------
        highlight=' '.join(str(word) for word in list_highlight_)
        
        #------------------------------------------------------
        # Dictionary is filled with highlights matching with
        # articles.
        #------------------------------------------------------
        #list_article.append(article)
        #list_highlight.append(highlight)
        
        dict_article[file_id]=article
        dict_highlight[file_id]=highlight
        
        file_count += 1
        if file_count % 1000 == 0:
            print("Files read : "+str(file_count))
        if read_count is not None:
            if file_count == read_count:
                break
            else:
                pass
        else:
            pass

    return dict_article, dict_highlight
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def write_train_set(train_dir, dict_article, dict_highlight):
    for root_name in dict_article.keys():
        tokenized_article = dict_article[root_name]
        str_tokenized_article=' '.join(str(e) for e in tokenized_article)
        file_write(str_tokenized_article, train_dir, root_name, '.art')
        
        tokenized_highlight = dict_highlight[root_name]
        str_tokenized_highlight=' '.join(str(e) for e in tokenized_highlight)
        file_write(str_tokenized_highlight, train_dir, root_name, '.hig')
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def read_train_set(train_dir, p_restriction=None, read_count=None):
    #-----------------------------------------------------------
    # Used to restrict the read to some files.
    #-----------------------------------------------------------
    file_count=0
    iglob__ = None
    
    dict_article=dict()
    dict_highlight=dict()
    
    if p_restriction is not None:
        iglob_art__ = glob.iglob(train_dir+"/"+p_restriction+".art")
        iglob_hig__ = glob.iglob(train_dir+"/"+p_restriction+".hig")
    else:
        iglob_art__ = glob.iglob(train_dir+"/*.art")
        iglob_hig__ = glob.iglob(train_dir+"/*.hig")

    for file_name_hig in iglob_hig__:
        file_root_name = file_name_hig.split('.hig')[0]
        file_name_art=file_root_name+'.art'    

        file_to_read = open(file_name_hig,'rt') 
        data_highlight = file_to_read.read()
        file_to_read.close()    

        file_to_read = open(file_name_art,'rt') 
        data_article = file_to_read.read()
        file_to_read.close()    

        root_name = get_root_name(file_root_name, file_extension='.hig')
        #------------------------------------------------------
        # Dictionary is filled with tokenized highlight
        #------------------------------------------------------
        dict_highlight[root_name]=data_highlight

        #------------------------------------------------------
        # Dictionary is filled with tokenized article
        #------------------------------------------------------
        dict_article[root_name]=data_article
               
        #------------------------------------------------------
        # Statistics over read files.
        #------------------------------------------------------
        file_count +=1
        if file_count % 1000 == 0:
            print("Files read : "+str(file_count))
        else:
            pass

        #------------------------------------------------------
        # If limit is reached in case of restrictions over number 
        # of files to be read then read is halted.
        #------------------------------------------------------
        if read_count is not None:
            if file_count == read_count:
                break
            else:
                pass
        else:
            pass

    
    return dict_article, dict_highlight
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def compute_frequency_deprecated(dict_content):
    
    list_content = list()
    for root_name in dict_content.keys():
        content= dict_content[root_name]
        list_content += content

    freq_content = nltk.FreqDist(list_content)

    return freq_content
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def compute_frequency(dict_content, doc_type="string"):
    """Returns the frequency of each word incuded in the values of doc_content.
    Input : 
        * dict_content : dictonary containing documents for any key.
        * doc_type : when value is 'string', then each document is encoded as a
        unique string. When value is tokenized, then each document is tokenized 
        and the tokens frequencies are returned.
    Output : 
        words or token frequencies included into all documents.
    """
    list_content = list()
    if doc_type == "string" :
        for root_name in dict_content.keys():
            content= dict_content[root_name]
            tokenized_content = content.split(' ')
            list_content += tokenized_content
    elif doc_type=="tokenized" :
        for root_name in dict_content.keys():
            content= dict_content[root_name]
            tokenized_content = content.split(' ')
            list_content += tokenized_content
    else : 
        print("*** ERROR : Unkown document type= "+str(doc_type))
        return None
    
    freq_content = nltk.FreqDist(list_content)

    return freq_content
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def compute_frequency_sentence(dict_content, token_mode='split'):
    """Computes each word frequency from a dictionary of contents.
    Dictionary values are sentences. Sentence is identified to a document.
    Dictionary keys are original document identifiers.
    
    Contents from dict_content are tokenized using split() method as default 
    mode.
    
    Other mode option is nltk to tokenize contents.
    """
    
    list_content = list()
    if token_mode == 'split' :
        for root_name in dict_content.keys():
            content= dict_content[root_name]
            tokenized_content = content.split(' ')
            list_content += tokenized_content
    elif token_mode == 'nltk' :
        for root_name in dict_content.keys():
            content= dict_content[root_name]
            tokenized_content = nltk.word_tokenize(content)
            list_content += tokenized_content
    else :
        print("*** ERROR : unknown tokenization mode= "+str(token_mode))
        return None

    freq_content = nltk.FreqDist(list_content)

    return freq_content
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_tfidf_vectorizer(dict_document, doc_type='tokenized'):
    """Returns dictionary containing tfidf values for each word 
    from all values of dict_document.
    Input:
        * dict_document : dictionary containing documents as values while keys 
        are documents identifiers.
        * doc_type : tokenized or string
    Output:
        * dict_tfidf : dictionaries from which values are tfidf and keys are 
        word from dict_document values.
    """
    list_token = list()
    if doc_type == 'tokenized' :
        #-----------------------------------------------------------------------
        # Each document is a list of tokens; lists are aggregated all togethers.
        #-----------------------------------------------------------------------
        for root_name in dict_document.keys():
            tokenized_document= dict_document[root_name]
            list_token += tokenized_document
    elif doc_type == 'string' :
        #-----------------------------------------------------------------------
        # Each document is a string; 
        # String is tokenized as a list of tokens. Then, all lists are 
        # aggregated to each others.
        #-----------------------------------------------------------------------
        for root_name in dict_document.keys():
            document= dict_document[root_name]
            tokenized_document = nltk.word_tokenize(document)
            list_token += tokenized_document
    else : 
        print("\n***ERROR  : Unknown document type= "+str(doc_type))
        return None

    vectorizer=TfidfVectorizer(norm="l2", use_idf=True)

    csr_matrix = vectorizer.fit_transform(list_token)

    return vectorizer
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_dict_tfidf(tfidf_vectorizer):
    """Returns a dictionary from which keys are words and values are tfidf.
    Words and tfidf values are extracted from tfidf_vectorizer given as 
    parameter.
    """
    dict_tfidf=dict()
    for word in tfidf_vectorizer.vocabulary_.keys():
        index=tfidf_vectorizer.vocabulary_[word]
        dict_tfidf[word]=tfidf_vectorizer.idf_[index]

    return dict_tfidf
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def split_train_test(dict_article, dict_highlight, train_ratio = 0.7):

    list_key =  list(dict_article.keys())
    
    train_size = int(len(list_key)*train_ratio)
    
    list_train_key = list_key[:train_size]
    list_test_key = list_key[train_size:]

    dict_X_train = dict()
    dict_y_train = dict()
    for key in list_train_key:
        dict_X_train[key] = dict_article[key]
        dict_y_train[key] = dict_highlight[key]

    dict_X_test = dict()
    dict_y_test = dict()
    for key in list_test_key:
        dict_X_test[key] = dict_article[key]
        dict_y_test[key] = dict_highlight[key]

    
    return dict_X_train, dict_X_test, dict_y_train, dict_y_test
#-------------------------------------------------------------------------------
    
