import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

import p6_util
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def dict_plot_frequency(dict_freq, doc_type, query=None, p_figsize=(10,10)):
    """Plot word frequencies onto a bar diagram from words into a dictionary.
    
    Input : 
        * dict_freq : nltk.probability.FreqDist object type; a kind of Python
        dictionary containing words as keys and words frequencies as values.
        
        * doc_type : string used for title.
        
        * query : used to restrict (keys, values) to be displayed.
        dict_freq is converted as a DataFrame object. Then query method is
        applied on such object.
    """
    z_=plt.figure(figsize=p_figsize)
    if dict_freq is not None:
        df= p6_util.get_df_from_FreqDist(dict_freq, query)
        title = "Words frequency for : "+doc_type
        title += " / "+query
        z_=df.plot(kind='bar', title=title, color='orange')
    else:
        print("\n*** ERROR : given FreqDist object is None!")
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def display_word_cloud(dict_word_freq, file_name=None):
    """Display a cloud of words from fiven dictionary formated as follow : 
    {word:word_count}.
    """
    wordcloud_generator = WordCloud(background_color='white', max_words=100)
    wordcloud_generator.generate_from_frequencies(dict_word_freq)
    
    if file_name is not None:
        wordcloud_generator.to_file(file_name)
    else:
        pass

    plt.figure(figsize=[20,10])
    plt.imshow(wordcloud_generator, interpolation='bilinear')

    plt.axis("off")
    plt.show()
#-------------------------------------------------------------------------------
    
