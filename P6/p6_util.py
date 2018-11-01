

import numpy as np
import pandas as pd
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from scipy import sparse

from bs4 import BeautifulSoup


from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer

from fuzzywuzzy import fuzz
from fuzzywuzzy import process


import p5_util


LIST_EMBEDDING_MODE=['tfidf','bow','ngram']
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def clean_marker_text(text,leading_marker=None, trailing_marker=None):
    """ Remove trailer and leading markers from a given text.
    Returns a list of words that were encapsulated between markers.
    """
    if leading_marker is not None:
        list1=text.split(leading_marker)
        str2 = " ".join(list1)
    else :
        str2 = text
    if trailing_marker is not None: 
        list2=str2.split(trailing_marker)
    else :
        list2 = str2.split()

    list3 = [element.strip() for element in list2 if 0 < len(element)]
    return list3
#-------------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_df_standardization(ser, is_stemming=False, is_lem=True, verbose=True) :
    """ Applies all pre-processing actions over a Series ser given as 
    parameter.
    
    Returned Series is a cleaned one.
    """
    if verbose is True :
        print("\nCleaning text in-between markers <code></code> markers...")
    ser = ser.apply(cb_remove_marker,args=('code',))

    if verbose is True :
        print("\nCleaning LXML markers...")
    ser = ser.apply(cb_clean_lxml)

    if verbose is True :
        print("\nRemove verbs from sentences...")
    ser = ser.apply(cb_sentence_filter)

    if verbose is True :
        print("\nFiltering alpha-numeric words from sentences...")
    ser = ser.apply(cb_remove_verb_from_sentence)

    if verbose is True :
            print("\nRemoving stopwords...")
    ser= ser.apply(cb_remove_stopwords)

    if is_lem is True:
        if verbose is True :
            print("\nLemmatization ...")
        lemmatizer=WordNetLemmatizer()
        ser=ser.apply(p5_util.cb_lemmatizer,args=(lemmatizer,'lower'))

    if is_stemming is True:
        if verbose is True :
            print("\nEnglish stemming ...")
        stemmer=SnowballStemmer('english')
        ser=ser.apply(p5_util.cb_stemmer,args=(stemmer,'lower'))
    return ser
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_df_from_FreqDist(freqDist, query=None):
    """Returns a dataframe from a given nltk.probability.FreqDist object class.
    The returned dataframe have a single column named Freq.
    """
    if freqDist is not None:
        df = pd.DataFrame.from_dict(freqDist, orient='index')
        if 'Freq' not in df.columns:
            df.rename(columns={0:'Freq'}, inplace=True)
            df.sort_values(by = 'Freq', inplace=True, ascending=False) 
        else:
            pass
    if query is not None:
        df=df.query(query)
    else :
        pass
    return df
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_list_tag_original(str_tag) :
    str_tag_original = re.sub("[^a-zA-Z0-9=++# ]", " ", str_tag )
    list_tag_original = [tag for tag in str_tag_original.split(' ') \
    if tag not in ['',]]
    return list_tag_original
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_list_tag_stat_tfidf(sentence, vectorizer, tag_original_count=0, tag_ratio=1.0) :
    """ Returns a list of TAGS from sentence given as parameter.
    List of returned TAGs count is a ratio from sentence words count.
    
    This fuction appplies for TFIDF vectorization only. This mean, documents are
    vectors from which components values are TFIDF values of 
    terms from vocabulary.
    
    Those values are stored into csr_matrix given as parameter 
    while terms from vocabulary are stored into vectorizer.
    
    Selected tags belong both to vocabulary and words from given sentence.

    Terms from vocabulary having greater occurence value are selected as TAGs.
    
    """

    list_term_sentence = sentence.split(' ')
    #---------------------------------------------------------------------------
    # Using vectorized sentence in csrmatrix, get indexes from all features 
    # with frequencies >0 
    #---------------------------------------------------------------------------
    csrmatrix= vectorizer.transform([sentence])
    arr_index = np.where(csrmatrix.A>0)[1]

    #---------------------------------------------------------------------------
    # Using array of indexes from features contained in sentence, 
    # vocabulary terms are extracted and regarded as TAGs.
    #---------------------------------------------------------------------------
    list_tag_sentence= [tag for tag, index in vectorizer.vocabulary_.items() \
    if index in arr_index]

    #---------------------------------------------------------------------------
    # Build dictionary {TAG:TFIDF} where TAG are issued from the given row.
    #---------------------------------------------------------------------------
    dict_tfidf = dict()
    for tag_sentence in list_tag_sentence : 
        index = vectorizer.vocabulary_[tag_sentence]
        dict_tfidf[tag_sentence]= vectorizer.idf_[index]

    #---------------------------------------------------------------------------
    # Get tag_count most greater value from dict_index
    #
    #---------------------------------------------------------------------------
    if tag_ratio is not None :
        tag_count = int(len(list_term_sentence)*tag_ratio)
    else :
        tag_count = tag_original_count
        
    df_row_tfidf = pd.DataFrame.from_dict(dict_tfidf, orient='index')

    if 0 < df_row_tfidf.shape[0] and 0< df_row_tfidf.shape[1]:
        df_row_tfidf = df_row_tfidf.rename(columns={0:'TFIDF'}, inplace=False)
        df_row_tfidf.sort_values(by=['TFIDF'],  ascending=False, inplace=True)
    else : 
        return None

    #---------------------------------------------------------------------------
    # TFIDF tag_count greatest values are returned.
    #---------------------------------------------------------------------------
    return sorted(list(df_row_tfidf.index[:tag_count]))
    
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_dict_vocab_frequency(vectorizer, csr_matrix):
    """Returns a dictionary {index:frequency} where : 
         * index values are indexes from vocabulary
         * frequency is the number of occurences of a term from vocabulary

       csr_matrix is used in order to compute occurences for any feature.
       A feature from csr_matrix is a term in vocabulary.

       This fuction appplies for BOW vectorization only.
    """
    
    if type(vectorizer) is not CountVectorizer:
        print("\n*** ERROR : get_dict_frequency function should apply with \
        CountVectorizer type only!")
        return None
    else: 
        pass

    #-------------------------------------------------------------------------------
    # Get list of indexes for any features
    #-------------------------------------------------------------------------------
    list_feature_index = [index  for (tag, index) \
    in vectorizer.vocabulary_.items()]

    #-------------------------------------------------------------------------------
    # Total occurence is computated for any feature.
    #-------------------------------------------------------------------------------
    arr_total_occurence = csr_matrix.toarray().sum(axis=0)

    #-------------------------------------------------------------------------------
    # Build dictionary {feature_index:occurence} for the given row
    #-------------------------------------------------------------------------------
    dict_vocabularyIndex_occurence = dict()
    for feature_index, total_occurence_value \
    in zip(list_feature_index, arr_total_occurence) :
        dict_vocabularyIndex_occurence[feature_index]=total_occurence_value

    return dict_vocabularyIndex_occurence
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_list_tag_stat_ngram(sentence, vectorizer, csr_matrix\
    , tag_ratio=1.0, tag_original_count=0) :
    #---------------------------------------------------------------------------
    # Checking parameters
    #---------------------------------------------------------------------------
    if type(vectorizer) is not CountVectorizer:
        print("\n*** ERROR : get_dict_frequency function should apply with \
        CountVectorizer type only!")
        return None
    else: 
        pass

    #---------------------------------------------------------------------------
    # Using vectorized sentence in csrmatrix, get indexes from all features 
    # with frequencies >0 
    #---------------------------------------------------------------------------
    csrmatrix= vectorizer.transform([sentence])
    arr_index = np.where(csrmatrix.A>0)[1]

    #---------------------------------------------------------------------------
    # Using array of indexes from features contained in sentence, 
    # vocabulary terms are extracted.
    #---------------------------------------------------------------------------
    list_tag= [tag for tag, index in vectorizer.vocabulary_.items() \
    if index in arr_index]

    #---------------------------------------------------------------------------
    # Build dictionary {TAG:TotalFreq} where TotalFreq is the sum over 
    # all rows from corpus matrix (csr_matrix) for any TAG belonging to list_tag.
    #---------------------------------------------------------------------------
    dict_frequency_row = dict()
    dict_vocab_frequency = get_dict_vocab_frequency(vectorizer, csr_matrix)
    for tag in list_tag : 
        index_tag = vectorizer.vocabulary_[tag]
        dict_frequency_row[tag] = dict_vocab_frequency[index_tag]

    #---------------------------------------------------------------------------
    # Dictionary is converted as a dataframe allowing values to be ordered.
    #---------------------------------------------------------------------------
    df_row_frequency \
    = pd.DataFrame.from_dict(dict_frequency_row, orient='index')
    
    df_row_frequency.rename(columns={0:'Frequency'}, inplace=True)
    df_row_frequency.sort_values(by=['Frequency'],  ascending=False, inplace=True)

    #---------------------------------------------------------------------------
    # Get tag_count most greater value from dict_index
    #---------------------------------------------------------------------------
    list_term_sentence = sentence.split(' ')
    if tag_ratio is not None :
        tag_count = int(len(list_term_sentence)*tag_ratio)
    else : 
        tag_count = tag_original_count
    return sorted(list(df_row_frequency.index[:tag_count]))
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_list_tag_stat_bow(sentence, vectorizer, csr_matrix, tag_ratio=1.0) :
    """ Returns a list of TAGS from sentence given as parameter.
    List of returned TAGs count is a ratio from sentence words count.
    
    This fuction appplies for BOW vectorization only. This mean, documents are
    vectors from which components values are occurences or frequencies of 
    terms from vocabulary. 
    
    Terms may be N-gram type with N>=1.
    
    Those frequencies values are stored into csr_matrix given as parameter 
    while terms from vocabulary are stored into vectorizer.
    
    Selected tags belong both to vocabulary and words from given sentence.

    Terms from vocabulary having greater occurence value are selected as TAGs.
    
    """
    #---------------------------------------------------------------------------
    # Checking parameters
    #---------------------------------------------------------------------------
    if type(vectorizer) is not CountVectorizer:
        print("\n*** ERROR : get_dict_frequency function should apply with \
        CountVectorizer type only!")
        return None
    else: 
        pass

    #---------------------------------------------------------------------------
    # Get intersection between words in vocabulary and and words in list_term.
    # TAG will be assigned from vocabulary issued from vectorization.
    #---------------------------------------------------------------------------
    list_term_sentence = sentence.split(' ')
    list_tag_row = list(set(list_term_sentence) & set(vectorizer.vocabulary_))

    #---------------------------------------------------------------------------
    # Build dictionary {TAG:Freq}
    #---------------------------------------------------------------------------
    dict_frequency_row = dict()
    dict_frequency = get_dict_vocab_frequency(vectorizer, csr_matrix)
    for tag_row in list_tag_row : 
        index = vectorizer.vocabulary_[tag_row]
        dict_frequency_row[tag_row] = dict_frequency[index]

    df_row_frequency \
    = pd.DataFrame.from_dict(dict_frequency_row, orient='index')
    
    df_row_frequency.rename(columns={0:'value'}, inplace=True)
    df_row_frequency.sort_values(by=['value'],  ascending=False, inplace=True)

    #---------------------------------------------------------------------------
    # Get tag_count most greater value from dict_index
    #---------------------------------------------------------------------------
    tag_count = int(len(list_term_sentence)*tag_ratio)

    return sorted(list(df_row_frequency.index[:tag_count]))
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_clean_lxml(str_lxml, mode='lower'):
    """This function clean (remove) all LXML markers from str_lxml given as 
    parameter.
    
    BeautifulSoup library is used to process LXML strings.
    
    It returns a cleaned string in low case.
    
    If mode value given as parameter is neither lower nor upper, then cleaned 
    string is returned in lower mode.
    """

    if 'lower' == mode : 
        soup = BeautifulSoup(str_lxml.lower(),"lxml") 
    elif 'uper' == mode : 
        soup = BeautifulSoup(str_lxml.upper(),"lxml") 
    else : 
        soup = BeautifulSoup(str_lxml.lower(),"lxml") 
        
    return soup.get_text()

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_sentence_filter(sentence, mode='lower'):
    """Remove all patterns that are not alpha-digital words, 
    also not '=' nor '++' characters; replace then with ' ' character.    
    """
    
    sentence_filtered = re.sub("[^a-zA-Z0-9=++# ]", " ", sentence )
    if mode=='lower' :
        sentence_filtered = sentence_filtered.lower()
    elif mode=="upper" :
        sentence_filtered = sentence_filtered.upper()
    else:
        sentence_filtered = sentence_filtered.lower()

    return sentence_filtered
    
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_remove_code_source_compare(item):
    """This function removes from item given as parameter source code 
    comparaison expression such as A=B or A>=cf,...
    """
    my_text = item.replace(' = ','=')
    my_text = my_text.replace(' == ','==')
    my_text = my_text.replace(' > ','>')
    my_text = my_text.replace(' < ','<')
    my_text = my_text.replace(' >= ','>=')
    my_text = my_text.replace(' <= ','<=')
    my_text = my_text.replace('  ',' ')
    
    
    my_text = re.sub("([A-Za-z0-9]+)(=+)([A-Za-z0-9]+)",' ',my_text)    
    my_text = re.sub("([A-Za-z0-9]+)(>)([A-Za-z0-9]+)",' ',my_text)    
    my_text = re.sub("([A-Za-z0-9]+)(<)([A-Za-z0-9]+)",' ',my_text)    
    my_text = re.sub("([A-Za-z0-9]+)(>=)([A-Za-z0-9]+)",' ',my_text)    
    my_text = re.sub("([A-Za-z0-9]+)(<=)([A-Za-z0-9]+)",' ',my_text)    
    return my_text
#-------------------------------------------------------------------------------    

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_list_sentence_from_df(df, feature) : 
    if feature not in df.columns : 
        print("*** ERROR : no column named "+str()+" into dataframe!")
        return None

    list_sentence = list()
    for sentence in df[feature].tolist():
        #tokenized_sentence = sentence.split(' ')
        tokenized_sentence=nltk.word_tokenize(sentence)
        list_sentence.append(tokenized_sentence)    
    return list_sentence
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_remove_marker(str_marked,str_marker):
    """Removes text in between markers <str_marker> and </str_marker>
    
        * Input :         
            str_marked : marked string, mean, text in between markers
            str_marker : marker compounding removed text.
    """
    str_marked = str_marked.replace('\n','')
    marker_start='<'+str_marker+'>'
    marker_end='</'+str_marker+'>'
    str_cleaned = re.sub(marker_start+"(.*)"+marker_end,"", str_marked.lower())
    return str_cleaned
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_list_tag_from_post(post, ml_model, max_tag=5):
    """Returns a list of most appropriate tags from a post.
    Tags are computed from a M.L model.
    
    """
    list_tag_returned = list()
    tokenized_post = post.split()
    list_tag_pred = ml_model.predict_output_word(tokenized_post)
    tag_count = 0
    for tuple_word_percent in list_tag_pred :
        if tag_count < max_tag:
            tag = "<"+str(tuple_word_percent[0])+">"
            list_tag_returned.append(tag)
            tag_count += 1

    return list_tag_returned
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_remove_verb_from_sentence(sentence):
    """Remove verbs from sentence given as parameter and returns a sentence
    without any verb
    """
    
    #---------------------------------------------------------------------------
    # All sentences are tokenized and words are tagged.
    #---------------------------------------------------------------------------
    list_tagged = list()
    tokenized_sentence = nltk.word_tokenize(sentence)
    list_tagged += nltk.pos_tag(tokenized_sentence)
    
    #-----------------------------------------------------------------------
    # List of tagged words from sentence are filtered
    #-----------------------------------------------------------------------
    list_word_filtered = list()
    for tuple_word_tag in list_tagged:
        tag = tuple_word_tag[1]
        if 'V' == tag[0] :
            pass
        else:
            list_word_filtered.append(tuple_word_tag[0])
    #-----------------------------------------------------------------------
    # Tokenized words that have been filtered are compound into a sentence
    #-----------------------------------------------------------------------
    sentence = " ".join(list_word_filtered)
    return sentence

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_remove_stopwords(item, lang='english', mode='lower') :
    """This function removes some stopwords form item given as parameter.
    Removed stopwords are issued from 'nltk.corpus.stopwords'.
    
    """
    list_word=item.split()
    item_no_stopwords_1=[ word for word in list_word \
    if word.lower() not in stopwords.words(lang) \
                      and not str(word).isdigit()]
                      
    item_no_stopwords=[word for word in item_no_stopwords_1 \
    if word.lower() not in ['cnn','.','#','way','would','like']]
    
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
def compute_frequency_sentence(dict_content, token_mode='split'):
    """Computes each word frequency from a dictionary of contents.
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
def preprocess_post(question, is_stemming=False, is_lem=True, is_stopword=True\
    ,is_stopverb=True, is_stopalfanum=True):
    """Applies to a given question all transformations in order to clean 
    text model.
        Input : 
            * question : suite of words forming a question
            * is_stemming : when True, stemming is applied on given post.
            * is_lem : when True, lemmatization is applied on given post.
            * is_stopword : when True, engish stopwords are filtered from post.
            * is_stopverb : when True, engish verbs are filtered from post.
            * is_stopalfanum : when True, non alpha-numeric characters are filtered 
            from given question.
        Output :
            * dataframe with  standardized Body column.
    """
    df = pd.DataFrame({'Body':question}, index=[0,])
   
    #print("\nCleaning text in-between markers <code></code> markers...")
    df["Body"] = df.Body.apply(cb_remove_marker,args=('code',))

    #print("\nCleaning LXML markers...")
    df["Body"] = df.Body.apply(cb_clean_lxml)

    if is_stopalfanum is True : 
        #print("\nRemove non alfa-numeric patterns")
        df["Body"] = df.Body.apply(cb_sentence_filter)

    if is_stopverb is True : 
        #print("\nFiltering sentences...")
        df["Body"] = df.Body.apply(cb_remove_verb_from_sentence)

    if is_stopword is True : 
        #print("\nRemoving stopwords...")
        df["Body"] = df.Body.apply(cb_remove_stopwords)

    if is_lem is True:
        #print("\nLemmatization ...")
        lemmatizer=WordNetLemmatizer()
        df['Body']=df.Body.apply(p5_util.cb_lemmatizer,args=(lemmatizer,'lower'))

    if is_stemming is True:
        #print("\nEnglish stemming ...")
        stemmer=SnowballStemmer('english')
        df['Body']=df.Body.apply(p5_util.cb_stemmer,args=(stemmer,'lower'))



    return df
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_tag_count(str_tags):
    """Returns number of tags from a string str_tag, given as parameter.
    str_tags is expected having following format: <tag1><tag2>...<tagN>
    """
    tag_count = 0
    for char_marker in str_tags:
        if char_marker == '<':
            tag_count +=1
    return tag_count
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def taglist_stat_predict(df_corpus, document_index, embedding_mode, vectorizer\
                        , csr_matrix, p_tag_ratio):
    """Returns a predicted list of TAGS from a given row from df_corpus 
    dataframe.
    
    Prediction is based on statistics of terms over the whole corpus.

    Given dataframe contains corpus of documents.

    Each row from Dataframe contains a document from which returned TAGs are 
    predicted.

    TAGs are extracted from vocabulary handled into vectorizer operator that 
    is given as parameter.
    
    Input : 
        * df_corpus : dataframe that is a corpus represented as a Bag Of Words .
        * document_index : document identifier from corpus (dataframe) fro which 
        TAGs will be extracted.
        * vectorizer : operator that has been built in order to embbed documents
         words
        * embeding_mode : may be TF-IDF, BOW, COO(co-occurence)
        * p_tag_ratio : ratio of extracted TAGs=number_of_tags/number_of_words
        This allows to compute the number of TAGs to be returned.
    Output :
        * The list of ordered predicted TAGS
        * The list of original TAGS
        * The original document isuue from df_corpus at document_index position.
    
    """
    
    #---------------------------------------------------------------------------
    # Check parameters
    #---------------------------------------------------------------------------
    if 'Body' not in df_corpus.columns :
        print("\n*** ERROR : dataframe corpus must contain column named Body!")
        return None,None,None
    else : 
        pass

    if 'Tags' not in df_corpus.columns :
        print("\n*** ERROR : dataframe corpus must contain column named Tags!")
        return None,None,None
    else : 
        pass
        
    if embedding_mode not in LIST_EMBEDDING_MODE :
        print("\n *** ERROR : embedding mode not supported : "\
        +str(embedding_mode))
        return None,None,None
    else : 
        pass
    
    #---------------------------------------------------------------------------
    # Extract document from corpus thanks to row index.
    #---------------------------------------------------------------------------
    if document_index is not None:
        df_document = df_corpus[df_corpus.index==document_index]
    else : 
        df_document = df_corpus


    #---------------------------------------------------------------------------
    # Document is normalized
    #---------------------------------------------------------------------------
    ser_document_std = p6_df_standardization(df_document.Body, verbose=False)


    #---------------------------------------------------------------------------
    # Extract standardized sentence as a dictionary : {document_id:sentence}
    #---------------------------------------------------------------------------
    dict_sentence = ser_document_std.to_dict()
    sentence_std = dict_sentence[document_index]

    #---------------------------------------------------------------------------
    # Get formated list of original TAGs for this document.
    # Formating consists in changing TAGS under the form <tag1><tag2>...<tagN>
    # into a list under the form [tag1,tag2,...,tagN]
    #---------------------------------------------------------------------------
    str_original_tag=df_document.Tags[document_index]
    list_tag_original=get_list_tag_original(str_original_tag)

    #---------------------------------------------------------------------------
    # Number of TAGs assigned in this document (in this post)
    #---------------------------------------------------------------------------
    tag_original_count = len(list_tag_original)  
    
    #---------------------------------------------------------------------------
    # Extract TAGs from standardized document using vectorizer that contains
    # vocabulary and statistical values upon each document word.
    # Number of TAGs in list depends on tag_ratio criteria. This criteria 
    # applies on number of words in docuement.
    #---------------------------------------------------------------------------
    if embedding_mode == 'tfidf':
        list_predicted_tag \
        = get_list_tag_stat_tfidf(sentence_std, vectorizer\
        , tag_ratio=p_tag_ratio, tag_original_count=tag_original_count)
    elif embedding_mode == 'bow' or embedding_mode == 'ngram':
        list_predicted_tag \
        = get_list_tag_stat_ngram(sentence_std, vectorizer\
        , csr_matrix, tag_ratio=p_tag_ratio\
        , tag_original_count=tag_original_count)
    else :
        pass
    
    #---------------------------------------------------------------------------
    # Record original document as long as original TAG list
    #---------------------------------------------------------------------------
    str_original_document = df_corpus.iloc[document_index].Body
    
    return list_predicted_tag, list_tag_original, str_original_document
    
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_list_cluster_tag(dict_corpus, vectorizer, cluster_labels\
, cluster_id, p_tag_ratio, is_log=True):
    """Returns a list of tags for a given cluster.
    Input : 
        * dict_corpus : dictionary : {index:document} for the whole corpus
        * vectorizer : operator used for corpus vectorization
        * cluster_labels : list of clusters labels
        * cluster_id : a cluster identifier
        * p_tag_ratio : ration of tags considering question lentgh.
    Output : 
        * list of tags from cluster given as parameter.
    """
    #---------------------------------------------------------------------------
    # Get corpus indexes for any document assigned with the given cluster;
    # This allows to access all documents from dict_corpus belonging to cluster.
    #---------------------------------------------------------------------------
    arr_cluster_index = np.where(cluster_labels==cluster_id)[0]
    
    #---------------------------------------------------------------------------
    # Initialization of list of TAGs relative to the cluster.
    #---------------------------------------------------------------------------
    list_cluster_tag = list()

    #---------------------------------------------------------------------------
    # Process any document assigned with cluster_id
    #---------------------------------------------------------------------------
    document_count = len(arr_cluster_index)
    for index in arr_cluster_index :
        document = dict_corpus[index]
        #-----------------------------------------------------------------------
        # Get TAGs from document assigned with cluster_id; 
        # we get all N-grams from document.
        # Number of TAGs is limited with p_tag_ratio.
        #-----------------------------------------------------------------------
        list_cluster_tag_ = get_list_tag_stat_tfidf(document\
        , vectorizer, tag_ratio=p_tag_ratio)
        
        #-----------------------------------------------------------------------
        # Aggregate into a single list all TAGs from cluster.
        # These aggragated TAGs will caracterized the cluster.
        #-----------------------------------------------------------------------
        if list_cluster_tag_ is not None :
            list_cluster_tag += list_cluster_tag_

    if is_log is True:
        print("Cluster #"+str(cluster_id)+" : Number of documents= "\
        +str(document_count)+" : Done!")

    return list_cluster_tag, document_count
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_dict_list_cluster_tag( arr_cluster_label, dict_sof_document, vectorizer\
, p_tag_ratio):
    """Returns a dictionary formated as following : {cluster_id:list_tag}
    where :
    * cluster_id is a cluster identifier 
    * list_tag is the list of selected TAGs for cluster_id
    Input : 
        * arr_cluster_label : array containing cluster identifier for each row 
        of vectorized corpus.
        
        * dict_sof_document : standardized corpus formated 
        as following : {index:document}
        
        * vectorizer : operator used for corpus vectorization. 
        Handled dictionary from which TAGs are picked up.
        
        * p_tag_ratio : number of TAGS / number of document terms. 
        It is applied for each document belonging a cluster.
    """
    #---------------------------------------------------------------------------
    # Get all cluster identifiers from arr_cluster_label
    #---------------------------------------------------------------------------
    arr_cluster_id = np.unique(arr_cluster_label)
    
    dict_cluster_stat = dict()
    
    #---------------------------------------------------------------------------
    # For any cluster, retrieve list of TAGs
    #---------------------------------------------------------------------------
    dict_list_cluster_tag = dict()
    dict_df_freq_cluster_tag = dict()
    for cluster_id in arr_cluster_id:
        #-----------------------------------------------------------------------
        # Get, from corpus, all indexes rows assigned with cluster_id
        #-----------------------------------------------------------------------
        arr_corpus_index = np.where(arr_cluster_label==cluster_id)[0]

        #-----------------------------------------------------------------------
        # Retrieve TAGs related to cluster_id.
        # The number of returned TAGs is constrained with p_tag_ratio.
        #-----------------------------------------------------------------------
        is_activated = (0==cluster_id%10)
        list_cluster_tag, document_count \
        = get_list_cluster_tag(dict_sof_document, vectorizer\
        , arr_cluster_label, cluster_id, p_tag_ratio, is_log=is_activated)

        #-----------------------------------------------------------------------
        # For each TAG in list_cluster_tag, tags frequency is computed in order 
        # to filter those tags with greater frequency.
        #-----------------------------------------------------------------------
        freq_cluster_tag = nltk.FreqDist(list_cluster_tag)

        df_freq_cluster_tag \
        = pd.DataFrame.from_dict(freq_cluster_tag, orient="index")
                
        df_freq_cluster_tag.rename(columns={0:'Freq'}, inplace=True)
        df_freq_cluster_tag.sort_values(by=['Freq'], axis=0, ascending=False\
        , inplace=True)

        #-----------------------------------------------------------------------
        # For word clod display over a cluster
        #-----------------------------------------------------------------------
        dict_df_freq_cluster_tag[cluster_id] = df_freq_cluster_tag

        #-----------------------------------------------------------------------
        # Lets format n-gram for having a more convenenient data to be displayed as 
        # a wordcloud set: any bigram word is aggregated into a single word with a 
        # separation character.
        #-----------------------------------------------------------------------
        list_cluster_tag_formated \
        = [re.sub("[' ']", "_", ngram_tag ) \
        for ngram_tag in df_freq_cluster_tag.index]
        
        str_cluster_tag_formated = ' '.join(list_cluster_tag_formated)
        
        dict_list_cluster_tag[cluster_id] = str_cluster_tag_formated
        
        #-----------------------------------------------------------------------
        # For statistics over each cluster
        #-----------------------------------------------------------------------
        dict_cluster_stat[cluster_id] = document_count
        
        
    return dict_list_cluster_tag, dict_cluster_stat, dict_df_freq_cluster_tag
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_tag_intersect_cluster_list_tag(list_question_key_words\
, dict_cluster_list_tag) :
    """Returns a dictionary formated as following : {cluster_id, tag_count} 
    where tag_count is issued from intersection between list_question_key_words 
    and dict_cluster_list_tag[clster_id], for any cluster_id.
    
    Input : 
        * list_question_key_words : vectorized question
        * dict_list_cluster_tag : dictionary {cluster_id, list_tag} issue from
        a clustering process.
    Output : 
        * A dictionary {cluster_id, tag_count} where,for each cluster_id,  
        tag_count is the number of elements issued from intersection between 
        list_question_key_words and list_tag.
    """

    dict_cluster_intersect_count=dict()
    for cluster_id, list_cluster_tag in dict_cluster_list_tag.items() :
        intersect_count = len(set(list_cluster_tag)&set(list_question_key_words))
        dict_cluster_intersect_count[cluster_id] = intersect_count

    return dict_cluster_intersect_count
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------
# 
#-------------------------------------------------------------------------
def p6_build_dict_dict_index_filter(list_ref, list_list_tag_target):
    index_row=0
    
    dict_filter=dict()
    for list_tag_target in list_list_tag_target:
        #--------------------------------------------------------------
        # list_tag_target is a list of target tags over a given row.
        #--------------------------------------------------------------
        row_tag_filter = np.zeros(len(list_ref), dtype=bool)
        for tag_target in list_tag_target:
            # -----------------------------------------------------------------------------------
            # For a given tag from a row, a filter is built.
            # This filer is issued from condition (np.array(list_ref)==tag_target
            # row_tag_filter is then a list of booleans which size is size of list_ref.
            # This filter is aggregated with all tags from the row, using bitwise operator "|".
            # -----------------------------------------------------------------------------------
            row_tag_filter = (row_tag_filter) | (np.array(list_ref)==tag_target)
        #---------------------------------------------------------------------
        # Filter dictionay for each row is built.
        #---------------------------------------------------------------------        
        dict_filter[index_row] = list(row_tag_filter)
        index_row +=1
    return dict_filter
#-------------------------------------------------------------------------



#-------------------------------------------------------------------------
#
#-------------------------------------------------------------------------
def p6_encode_target(list_tags_ref, list_list_tags):
    """One hot encode the given list_list_tags into values from list_tags_ref.
    Returns a list of encoded values.
    
    """
    dict_index_filter \
    = p6_build_dict_dict_index_filter(list_tags_ref,list_list_tags)
    
    list_all_encoded_row=list()
    for row, index_filter in dict_index_filter.items():
        encoded_row = np.zeros(len(list_tags_ref), dtype=int)
        #-----------------------------------------------------------------------------------
        # np.where(condition) will return an array of indexes values.
        #-----------------------------------------------------------------------------------
        index_row_array = np.where(dict_index_filter[row])
        
        #-----------------------------------------------------------------------------------
        # Row is encoded with value 1 for index values from index_row_array
        #-----------------------------------------------------------------------------------
        for i in index_row_array:
            encoded_row[i]=1
            
        #-----------------------------------------------------------------------------------
        # Encoded row is added to list of encoded rows 
        #-----------------------------------------------------------------------------------
        list_all_encoded_row.insert(row,list(encoded_row))
    return list_all_encoded_row
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
#
#-------------------------------------------------------------------------
def p6_get_tags_from_dict(row, dict_filter, list_all_tags):
    index_array_tag = np.where(np.array(dict_filter[row])==1)
    return np.array(list_all_tags)[index_array_tag]
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
#
#-------------------------------------------------------------------------
def p6_get_list_tag_from_encoded_row(list_list_encoded_row,row,list_all_tags):
    list_encoded_row = list_list_encoded_row[row]
    arr_encoded_index = np.where(np.array(list_encoded_row)==1)
    return np.array(list_all_tags)[arr_encoded_index]
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
#
#-------------------------------------------------------------------------
def p6_get_list_all_tag(ser_tag):
    """Returns a list of unique tags from a given Series.
    """
    list_all_tags=list()
    #---------------------------------------------------------------------
    # All tags lists from any row are aggregated as a unique list of tags.
    # Tags may be duplicated onto this list.
    #---------------------------------------------------------------------
    for index, list_tags in ser_tag.items():
        list_all_tags =   list_all_tags+list_tags    

    #---------------------------------------------------------------------
    # Tags are made unique thanks to set()
    #---------------------------------------------------------------------
    list_all_tags = list(set(list_all_tags))
    return list_all_tags
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_encode_target_in_csr_matrix(list_ref_tags, list_tags_to_be_encoded) :
    """Returns a list of list of encoded TAGs that is compressed into a 
    CSR matrix.
    Input : 
        *  list_ref_tags : list of TAGs used as reference for encoding
        *  list_tags_to_be_encoded : list of TAGs to be encoded.
    Output:
        * CSR matrix of encoded TAGs
        Format of returned value is a CSR matrix with expanded list of lists  : 
            [
       Row 1--->[tag_1.1, tag_1.2, ..... tag_1.K],
                    .
                    .
                    .                
       Row p--->[tag_p.1, tag_p.2, ..... tag_p.K],
                    .
                    .
                    .                
       Row N--->[tag_N.1, tag_N.2, ..... tag_N.K],            
            ]
    """
    #---------------------------------------------------------------------------
    # For each row, TAGs represented as a list of elementaries TAG are encoded
    # Each row that is a string of TAGs is splitted into a list of elementary 
    # TAGs all rows are aggregated into a list
    #---------------------------------------------------------------------------
    list_list_encoded_row = p6_encode_target(list_ref_tags\
    , list_tags_to_be_encoded)
    
    #---------------------------------------------------------------------------
    # Conversion into CSR matrix fro easyness of computation
    #---------------------------------------------------------------------------
    csr_matrix_encoded_tag=sparse.csr_matrix(np.array(list_list_encoded_row))
    
    return csr_matrix_encoded_tag
#-------------------------------------------------------------------------------    
         
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_encode_ser_tag_2_csrmatrix(ser_tag, leading_marker='<'\
, trailing_marker='>'):
    """One hot encode a Series given as parameter.
    Each row from Series content has following format : <tag1><tag2>...<tagN>
    
    Returned value is a CSR matrix with expanded list of lists format : 
        [
   Row 1   [tag_1.1, tag_1.2, ..... tag_1.K],
                .
                .
                .                
   Row p   [tag_p.1, tag_p.2, ..... tag_p.K],
                .
                .
                .                
   Row N   [tag_N.1, tag_N.2, ..... tag_N.K],            
        ]
    """
    #---------------------------------------------------------------------------
    # Markers '<' and '>' are removed from tags.
    #---------------------------------------------------------------------------
    ser_tag = ser_tag.apply(clean_marker_text, leading_marker=leading_marker\
    , trailing_marker=trailing_marker)

    #---------------------------------------------------------------------------
    # A unique list of all TAGs is built : this is the vocabulary for TAGs**
    # This list is supposed to be completed enough for covering test tags dataset.
    #---------------------------------------------------------------------------
    list_ref_tags = p6_get_list_all_tag(ser_tag)

    csr_matrix_encoded_tag \
    = p6_encode_target_in_csr_matrix(list_ref_tags, ser_tag.tolist())      
    #---------------------------------------------------------------------------
    # For each row, TAGs represented as a list of elementaries TAG are encoded
    # Each row that is a string of TAGs is splitted into a list of elementary TAGs
    # All rows are aggregated into a list
    #---------------------------------------------------------------------------
    # list_list_encoded_row = p6_encode_target(list_all_tags, ser_tag.tolist())

    #---------------------------------------------------------------------------
    # Conversion into CSR matrix fro easyness of computation
    #---------------------------------------------------------------------------
    #csr_matrix_encoded_tag=sparse.csr_matrix(np.array(list_list_encoded_row))

    return csr_matrix_encoded_tag    
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_extract_list_tag_from_sof_tag(list_sof_tag_1, list_tag_suggested_1 \
    ,score_cutoff=0) :

    """Returns a list of TAGs extracted from a list of referenced TAGs 
    provided by Stack Over FLow.
    Extracted list of TAGs are TAGs from a suggested list of TAGs with greater 
    proximity with TAGs from Stack Over FLow list.
    """
    #---------------------------------------------------------------------------
    # Initialization for avoiding None return.
    #---------------------------------------------------------------------------
    list_extracted_tag = list()

    #---------------------------------------------------------------------------
    # Extract Tags from SOF tags list given the suggested tag list.
    #---------------------------------------------------------------------------
    for tag_suggested in list_tag_suggested_1 :
        tuple_extracted_tag = process.extractOne(tag_suggested, list_sof_tag_1\
        , score_cutoff=score_cutoff)
        if tuple_extracted_tag is not None :
            list_extracted_tag.append(tuple_extracted_tag[0])

    return list_extracted_tag
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_score_mean_string_simlarity(nb_test, df_corpus_test, list_sof_tag\
    , vectorizer, csr_matrix,p_tag_ratio=0.1, embeding_mode='bow' ) :
    """Returns mean similarity score for suggested tags.
    Input :
        * nb_test : number of tests to be evaluated
        * df_corpus_test : corpus of documents to be tested
        * vectorizer : vectorizer used for documents vectorization
        * csr_matrix : vectorized documents as TF_IDF matrix
        * p_tag_ratio : ratio of tags to be taking into account from TF-IDF 
          vectorization
        * embeding_mode : 
    Output : 
        * dictionary with, for each document evaluated, under the format:
            -> the mean similarity ratio
               If similarity ratio > 0 : they are more TAGs suggested 
               then original list.
            -> the number of matching TAGs between extracted TAGs and original 
            TAGs
        
    """
    idoc=0
    dict_match_result = dict()
    modulo = int(nb_test/10)
    print("\nTest mode {} covering {} documents\n"\
    .format(embeding_mode, nb_test))

    for index_document in range(0, nb_test):
        #-----------------------------------------------------------------------
        # Document is extracted from corpus
        #-----------------------------------------------------------------------
        df_document = df_corpus_test[df_corpus_test.index==index_document]

        #-----------------------------------------------------------------------
        # List of suggested TAGs is returned from TF-IDF matrix; weights from 
        # predicators with higher values are selected and related TAGs are
        # returned.
        #-----------------------------------------------------------------------
        list_tag_suggested, list_tag_original, str_original_document \
        = taglist_stat_predict(df_corpus_test, index_document\
                                    , embeding_mode\
                                   , vectorizer, csr_matrix, p_tag_ratio)
            
        #-----------------------------------------------------------------------
        # Similarity of list of suggested TAGs with Stack Over Flow TAGs is 
        # computed using Fuzzy module.
        # Similarity value range from 0 to 100.
        # When similarity value is 100, then evaluated TAGs are considered as 
        # matching. If similarity value is >= 90, then 2 TAGs are considered 
        # as similar.
        #
        # Result is returned in a list of extracted TAGs.
        #-----------------------------------------------------------------------
        if list_tag_suggested is not None:
            list_tag_extracted = p6_extract_list_tag_from_sof_tag(list_sof_tag\
            , list_tag_suggested, score_cutoff=90) 

            
            #-------------------------------------------------------------------
            # For each document, a mean similarity ratio is computed dividing 
            # number of extracted TAGs with number of original TAGs.
            #-------------------------------------------------------------------
            mean_similarity_ratio \
            = (len(list_tag_extracted)/len(list_tag_original))*100
            
            #-------------------------------------------------------------------
            # For each document, the matching count between extracted TAGs 
            # and original TAGs is computed.
            #-------------------------------------------------------------------
            match_count = \
            len(set(list_tag_extracted).intersection(set(list_tag_original)))
        else : 
            mean_similarity_ratio =0.
            match_count=0
            list_tag_extracted=list()

        #-----------------------------------------------------------------------
        # Result is stored into a dictionary along with list of extracted TAGs
        #  and list of original TAGs.
        #-----------------------------------------------------------------------
        dict_match_result[index_document] = (mean_similarity_ratio\
        , match_count/len(list_tag_original)\
        , list_tag_extracted, list_tag_original)

        #-----------------------------------------------------------------------
        # Tracking computation...
        #-----------------------------------------------------------------------
        idoc += 1
        if modulo > 0 :
            if idoc %modulo == 0 :
                print("Processed documents : {}/{}".format(idoc,nb_test ))
        else:
            print("Processed documents : {}/{}".format(idoc,nb_test ))
                
            
    return dict_match_result
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_stat_compute_result(dict_match_result):

    list_similarity_result = [value[0] for value in dict_match_result.values()]
    arr_similarity_result = np.array(list(list_similarity_result))

    list_matching_result = [value[1] for value in dict_match_result.values()]
    arr_matching_result = np.array(list(list_matching_result))

    #arr_result[0]=1
    percent_result \
    = (np.where(arr_similarity_result>=100)[0].shape[0]/len(list_similarity_result))*100
    print("\n*** Mean similarity indice >100: {0:1.2F} %".format(percent_result))

    percent_result \
    = (np.where(arr_similarity_result==0)[0].shape[0]/len(list_similarity_result))*100
    print("\n*** Mean similarity indice = 0: {0:1.2F} %".format(percent_result))


    percent_result = (np.where(arr_matching_result>0)[0].shape[0]/len(list_matching_result))*100
    print("\n*** Matching results : {0:1.2F} %".format(percent_result))
    
    return arr_similarity_result, arr_matching_result
#-------------------------------------------------------------------------------
    
