import os
import random

import pandas as pd
import numpy as np

import cv2
from PIL import ImageOps
from PIL import Image
from  sklearn import model_selection
from sklearn.decomposition import PCA


 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import to_categorical

import p3_util
import p5_util
import p6_util
import p7_util

import matplotlib.pyplot as plt
import cv2
import P7_DataBreed


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def mask_original_pil_image(self,df_pil_image_filtered, df_pil_image_original ):
    '''Build a new image from 2 dataframes containing informations and 
    decriptors related to splitted PIL images.
    
    Image filtering is based on KP density in splitted PIL image.
    Original informations from splitted image is used in order to display 
    filtered splitted image.
    Input : 
        * df_pil_image_filtered : dataframe contains informations from 
        filtered image.
        * df_pil_image_original : dataframe contains informations from 
        original image. This dataframe holds 2 levels indexes. First for rows 
        and second for columns.
    '''
    #-----------------------------------------------------------------------
    # Image to be returned is intialized.
    #-----------------------------------------------------------------------
    new_im = Image.new('L', self._std_size)

    #-----------------------------------------------------------------------
    # Building a black patch that replaces patch issued from splitted image 
    # in original dataframe when this part does not belongs to filtered image.
    #-----------------------------------------------------------------------
    x_delta = int(self._std_size[0]/self._split_ratio[0])
    y_delta = int(self._std_size[1]/self._split_ratio[1])
    black_image = Image.new('L', (x_delta,y_delta))
    
    
    #-----------------------------------------------------------------------
    # Step for Y is initalized.
    # df_pil_image_original
    #-----------------------------------------------------------------------
    y_delta = int(self._std_size[1]/self._split_ratio[1])
    x_offset = 0
    y_offset = 0
    for raw in np.unique(df_pil_image_original.index.labels[0].tolist()) :
        df_raw_filtered = df_pil_image_filtered[df_pil_image_filtered.raw==raw]
        list_col = df_raw_filtered.col.tolist()
        for col in np.unique(df_pil_image_original.index.labels[1].tolist()):
            if col in list_col :
                pil_image = df_pil_image_original.split_image.loc[raw,col]
                new_im.paste(pil_image, (x_offset,y_offset))
                x_offset += pil_image.size[0]
            else :
                pil_image = black_image
                new_im.paste(pil_image, (x_offset,y_offset))
                x_offset += pil_image.size[0]
        x_offset = 0
        y_offset +=y_delta
    return new_im
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def plot_match_descriptor(list_pil_image, is_plot=True) :
    list_kp = list()
    list_desc = list()

    for pil_image in list_pil_image :
        kp, desc = get_image_kpdesc(pil_image)
        list_kp.append(kp)
        list_desc.append(desc)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2SQR, crossCheck=True)
    # Match descriptors.

    matches = bf.match(list_desc[0],list_desc[1])

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    pil_image = list_pil_image[0]
    # Draw first 10 matches.
    pil_image = cv2.drawMatches(np.array(list_pil_image[0]),list_kp[0]\
                               ,np.array(list_pil_image[1]),list_kp[1]\
                               ,matches[:40]
                               ,np.array(pil_image), flags=2)
    if is_plot is True :
        plt.figure(figsize=(20,10))
        z_=plt.imshow(pil_image),plt.show()
    return pil_image

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def object_dump(oP7_DataBreed_to_dump, filename) :
    '''Dump oP7_DataBreed object into file given in parameter.
    Dictionary containing cv2.KeyPoint objects is erased because 
    pickling such objects fail.
    '''    
    oP7_DataBreed_to_dump._dict_breed_kpdesc = dict()

    if 'kp' in oP7_DataBreed_to_dump.df_pil_image_kpdesc.columns :
        oP7_DataBreed_to_dump.df_pil_image_kpdesc.kp \
        = oP7_DataBreed_to_dump.df_pil_image_kpdesc.kp.apply(lambda val:list())
    else :
        pass    

    if 'split_image' in oP7_DataBreed_to_dump.df_pil_image_kpdesc.columns:
        oP7_DataBreed_to_dump.df_pil_image_kpdesc.split_image\
        = oP7_DataBreed_to_dump.df_pil_image_kpdesc.split_image.apply(lambda val:list())
    else :
        pass
    p5_util.object_dump(oP7_DataBreed_to_dump,filename)
    print('*** INFO : object is saved removing cv2.KeyPoint objects!')
    return oP7_DataBreed_to_dump
    
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_dict_processor_list() :
    dict_pil_processor = dict()
    list_processor_id = list()
    
    list_pil_processor = P7_DataBreed.LIST_PIL_PROCESSOR.copy()
    for id, pil_processor in zip(range(0,len(list_pil_processor) )\
    ,list_pil_processor ) :
        dict_pil_processor.update({id:pil_processor})
        list_processor_id.append(id)        
    return list_processor_id,dict_pil_processor
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def update_object_save(oP7_DataBreed, is_saved=True, filename=None, is_new_attribute=True):
    
    if filename is None :
        filename='./data/oP7_DataBreed.dump'

    oP7_DataBreed_save = P7_DataBreed()

    if oP7_DataBreed is None :
        oP7_DataBreed = p5_util.object_load(filename)
    
    try: 
        oP7_DataBreed
    except NameError:
        print('*** INFO : oP7_DataBreed is not defined; loading...')
        oP7_DataBreed = p5_util.object_load(filename)
        is_saved = False

    oP7_DataBreed_save.copy(oP7_DataBreed,is_new_attribute=is_new_attribute)
    oP7_DataBreed = P7_DataBreed()
    oP7_DataBreed.copy(oP7_DataBreed_save,is_new_attribute=is_new_attribute)
    del(oP7_DataBreed_save)

    if is_saved is True:
        object_dump(oP7_DataBreed,filename)

    return oP7_DataBreed
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_keras_X_y_build_deprecated(ser_pil_image,ser_label):
    '''Convert Series of PIL image into numpy array allowing to feed Keras 
    dense layer. 
    Convert Series of labels into numpy array of pixels.
    
    Input :
        * ser_pil_image : Series of PIL images.
        * ser_label : Series of label.
    Output : 
        * arr_X : array of converted PIL images.
        * arr_y : array of converted Series label.
    '''

    #---------------------------------------------------------------------------
    # Build list of numpy arrays, each array is a PIL image.
    #---------------------------------------------------------------------------
    list_X = [np.array(pil_image) for pil_image in ser_pil_image]
    list_y = [y for y in ser_label]

    #---------------------------------------------------------------------------
    # Get dimensions values for weight, height and channels.
    #---------------------------------------------------------------------------
    weight  = list_X[0].shape[0]
    height  = list_X[0].shape[1]
    channel = list_X[0].shape[2]
    
    #---------------------------------------------------------------------------
    # Initialization of Series conversion.
    #---------------------------------------------------------------------------
    arr_X  = list_X[0].reshape((1, weight, height, channel))

    #---------------------------------------------------------------------------
    # Build array of images from list of numpy arrays.
    #---------------------------------------------------------------------------
    image_count = arr_X.shape[0]
    list_image_error = list()
    for image_i in range(1,image_count) :
        try :
            arr_X\
            = np.append(arr_X,list_X[image_i].reshape((1, weight, height, channel)),axis=0)
        except ValueError :
            list_image_error.append(image_i)
    print("INFO : number of errors = "+str(len(list_image_error)))
    #---------------------------------------------------------------------------
    # Build array of labels
    #---------------------------------------------------------------------------
    arr_y = np.array(list_y)
    
    return arr_X, arr_y
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_keras_X_y_build(ser_pil_image, ser_label, square=None, resize=None):
    '''Convert Series of PIL image into numpy array allowing to feed Keras 
    dense layer. 
    Convert Series of labels into numpy array of pixels.
    Image is resized or reshape as a square, depending parameters.
    
    Input :
        * ser_pil_image : Series of PIL images.
        * ser_label : Series of label.
        * square : when activated, then image is reshaped as a square. If None, 
        then image is resized.
    Output : 
        * arr_X : array of converted PIL images.
        * arr_y : array of converted Series label.
    '''
    
    #---------------------------------------------------------------------------
    # Initialization : convert all PIL images into Keras array of images.
    #---------------------------------------------------------------------------
    image_id = 0
    list_image_error = list()
    list_label_error = list()
    if resize is not None :
        pil_image_square = ser_pil_image.iloc[image_id].resize(resize)
    else :
        pil_image_square = pil_square(ser_pil_image.iloc[image_id], square=square)
    arr_keras_image_vstack = np.array(pil_image_square)
    
    weight  = arr_keras_image_vstack.shape[0]
    height  = arr_keras_image_vstack.shape[1]
    channel = arr_keras_image_vstack.shape[2]
    
    arr_keras_image_vstack \
    = arr_keras_image_vstack.reshape(1,weight, height,channel)
    
    arr_label_vstack = np.array(ser_label.iloc[image_id])

    #---------------------------------------------------------------------------
    # Convert all other PIL images into Keras array of images.
    # Build array of labels
    #---------------------------------------------------------------------------
    for (image_id, pil_image) , (label_id, label) \
    in zip(ser_pil_image.iloc[1:].items(), ser_label.iloc[1:].items() ):

        if resize is not None :
            pil_image_square = pil_image.resize(resize)
        else :
            pil_image_square = pil_square(pil_image, square=square)
    
        
        arr_keras = np.array(pil_image_square)
        
        weight  = arr_keras.shape[0]
        height  = arr_keras.shape[1]
        channel = arr_keras.shape[2]
        
        arr_keras = arr_keras.reshape(1,weight, height,channel)
        try :
            arr_label = np.array(ser_label.iloc[label_id])
        except IndexError:
            #print("*** ERROR for label indexes=: ("+str(image_id)+","+str(label_id)+")")
            list_label_error.append(label_id)
            continue
            
        try :
            arr_keras_image_vstack \
            = np.vstack((arr_keras_image_vstack,arr_keras))
            arr_label_vstack = np.vstack((arr_label_vstack,arr_label))
        except ValueError:
            #print("*** ERROR : "+str(arr_keras.shape))
            list_image_error.append(image_id)
            
    print("INFO : number of Image errors = "+str(len(list_image_error)))
    print("INFO : number of Label errors = "+str(len(list_label_error)))
    
    return arr_keras_image_vstack, arr_label_vstack
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_keras_X_train_test_build(ser_pil_image, ser_label\
, test_size=0.2, square=None, resize=None):
    '''Build train and test arrays based on given pandas Series.
    
    Input : 
        * ser_pil_image : Series containing PIL images.
        * ser_label : Series of labels related to PIL images.
        * test_size : percentage of test data-set.
        * square : 
        * resize : 
    Output :
        * arr_X_train : array of pixels containing train dataset
        * arr_X_test : array of pixels containing test dataset
        * arr_y_train : array of labels related to images from train dataset.
        * arr_y_test : array of labels related to images from test dataset.
        
    '''
    #---------------------------------------------------------------------------
    # Convert train dataset Series into numpy array allowing to feed Keras 
    # dense layer.
    #---------------------------------------------------------------------------
    print("Total images = "+str(len(ser_pil_image)))
    arr_keras_image, arr_label \
    = p7_keras_X_y_build(ser_pil_image, ser_label, square=square, resize=resize)
    print(arr_keras_image.shape, arr_label.shape)

    #---------------------------------------------------------------------------
    # Get train and test dataset returned as Series of PIL images
    #---------------------------------------------------------------------------
    arr_keras_image_train, arr_keras_image_test, arr_label_train, arr_label_test \
    = model_selection.train_test_split(arr_keras_image, arr_label\
    , test_size=test_size)    
    
    return arr_keras_image_train, arr_keras_image_test, arr_label_train\
    , arr_label_test
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def show_pil_image_and_kp(pil_image,breedname,is_plot=False, resize=None) :
    if resize is not None :
        pil_image = pil_image.resize(resize)
    else :
        pass
    kp, desc = get_image_kpdesc(pil_image)
    print("KP= "+str(len(kp)))
    print("DESC= "+str(desc.shape))
    dict_breed_kpdesc = {breedname:[(kp,desc)]}
    dict_pil_image = {breedname : [pil_image] }
    dict_breed_kpdesc_image = dict()
    
    for (breed, list_breed_kpdesc), list_image_pil \
    in zip(dict_breed_kpdesc.items(), dict_pil_image.values()):
        dict_breed_kpdesc_image[breed] \
        = [cv2.drawKeypoints(np.array(image_pil), kp, np.array(image_pil)) \
                                 for ((kp, desc),image_pil) \
                                 in zip(list_breed_kpdesc,list_image_pil)]
    if is_plot is True :
        p7_util.p7_image_pil_show(dict_breed_kpdesc_image,std_image_size=None)
    return dict_breed_kpdesc_image

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def process_breed_sample(dirbreed, list_image_name, resize, is_filtered=True) :
    dict_pil_image = dict()
    dict_pil_image['breedname'] = list()
    for image_name in list_image_name :
        image_path_name = dirbreed+'/'+image_name
        pil_image = p7_read_image(image_path_name)
        
        if is_filtered is False :
            dict_pil_image['breedname'].append([pil_image])
        else :
            dict_pil_image['resize']   = list()
            dict_pil_image['orig']     = list()
            dict_pil_image['2gray']    = list()
            dict_pil_image['equalize'] = list()

            

        dict_pil_image['orig'].append([pil_image])
    
        if resize is not None :
            pil_image = pil_resize(pil_image,resize)
            dict_pil_image['resize'].append([pil_image])

        pil_image = pil_2gray(pil_image)
        dict_pil_image['2gray'].append([pil_image])

        pil_image = pil_equalize(pil_image)
        dict_pil_image['equalize'].append([pil_image])
    return dict_pil_image
#-------------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_breedname_from_dirbreed(dirbreed):
    dirbreed_splitted = dirbreed.split('-')
    if len(dirbreed_splitted) == 2 :
        return dirbreed_splitted[1]
    else :
        return dirbreed
    #return dirbreed.split('-')[1]
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_image_kpdesc(pil_image):
    kp, desc = p7_util.p7_gen_sift_features(np.array(pil_image))
    return kp, desc
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def pil_gaussian(pil_image) :
    filename, pil_image = p7_util.p7_filter_gaussian(pil_image, size=3)
    return pil_image
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def pil_truncate(pil_image,std_size) :
    ''' 
        Input :
            * pil_image : PIL image format to be truncated 
            * std_size : tuple for size truncation
        Output : 
            * truncated PIL image.
    '''
    arr_image = np.array(pil_image)
    
    margin_w = int((arr_image.shape[0] -  std_size[0])/2)
    margin_h = int((arr_image.shape[1] -  std_size[1])/2)
    if margin_w >0 :
        # Horizontal truncation
        arr_image = arr_image[margin_w:,:]
        arr_image = arr_image[:-margin_w,:]
    else :
        pass
        
    if margin_h >0 :
        # Vertical truncation
        arr_image = arr_image[:,margin_h:]
        arr_image = arr_image[:,:-margin_h]
    else :
        pass

    return pil_resize(Image.fromarray(arr_image),std_size)     
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def pil_square(pil_image,square=None):
    '''Truncate image with a margin for having same size for height and weight.
    Input :
        * Rectangular PIL image 
        * square : when None value, then square is computed form image weight and 
        height. Otherwise, square is a tuple with square weight, height (same) 
        values.
    Output :
        * Squared PIL image 
    '''
    arr_image = np.array(pil_image)
    weight = arr_image.shape[0]
    height = arr_image.shape[1]
    if square is None :
        delta = weight - height
        margin = np.abs(int(delta/2))
        if weight >= height:
            margin_x = margin
            margin_y = 0
        else :
            margin_x = 0
            margin_y = margin
        
    else :
        margin_x = int((weight - square[0])/2)
        margin_y = int((height - square[1])/2)
        

    
    # Horizontal truncation
    arr_image = arr_image[margin_x:,:]
    if 0 < margin_x :
        arr_image = arr_image[:-margin_x,:]

    # Vertical truncation
    arr_image = arr_image[:,margin_y:]
    if 0 < margin_y:
        arr_image = arr_image[:,:-margin_y]


    
    #---------------------------------------------------------------------------
    # After truncation, normalization makes sure that weight and height are 
    # fixed with same value.
    #---------------------------------------------------------------------------
    weight = arr_image.shape[0]
    height = arr_image.shape[1]
    size = min(weight,height)
    std_size = (size,size)
    
    #return pil_image#Image.fromarray(arr_image)
    return pil_resize(Image.fromarray(arr_image),std_size)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def pil_edge(pil_image):
    edges = cv2.Canny(np.array(pil_image),pil_image.size[0],pil_image.size[1])
    pil_image = edges-np.array(pil_image)
    
    return Image.fromarray(pil_image)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_filter_median(pil_image) :
    filename, pil_image = p7_util.p7_filter_median(pil_image)
    return pil_image
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def pil_2gray(pil_image):
    return pil_image.convert('L')
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def pil_autocontrast(pil_image):
    return ImageOps.autocontrast(pil_image)
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def pil_equalize(pil_image):
    return ImageOps.equalize(pil_image)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def pil_resize(pil_image, resize):
    return pil_image.resize(resize)
#-------------------------------------------------------------------------------
        
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def pil_gradient(pil_image):
    kernel=np.array([[0,1,-1],[0,1,-1],[0,1,-1]])
    filename, pil_image \
    = p7_util.p7_filter_convolutional(pil_image, kernel, size=(3,3))
    return pil_image
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def pil_laplacien_kernel_4(pil_image):
    kernel=np.array([[0,-1,0],[-1,4,-1],[0,-1,0] ])
    filename, pil_image \
    = p7_util.p7_filter_convolutional(pil_image, kernel, size=(3,3))
    return pil_image
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def pil_laplacien_kernel_8(pil_image):
    kernel=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1] ])
    filename, pil_image \
    = p7_util.p7_filter_convolutional(pil_image, kernel, size=(3,3))
    return pil_image
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def pil_low_pass(pil_image):
    kernel=np.array([[1,1,1],[1,4,1],[1,1,1] ])
    filename, pil_image \
    = p7_util.p7_filter_convolutional(pil_image, kernel, size=(3,3))
    return pil_image
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_read_image(imagepathname, image_type='PIL') :
    pil_image_copy = None
    pil_image = p7_util.p7_pil_image_load(imagepathname\
            , is_verbose=False, std_size=None)

    return pil_image
    #return pil_image
#-------------------------------------------------------------------------------
        

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
class P7_DataBreed() :
    '''This class implements breeds data structure.
        oP7_DataBreed
            +
            |
            +-->load()
            |   |
            |   +--> p7_util.p7_load_dict_filename()
            |   
            +-->sampling()
            |   |
            |   +--> build_ser_number_breedname()
            |
            +-->build_sift_desc()
            |   |
            |   +--> build_ser_number_breedname()
            |   |
            |   +--> p7_util.p7_pil_image_load()
            |   |
            |   +--> kpdesc_build()
            |
            +-->kp_filter()
            |
            +-->build_arr_desc()
            |
            |   GMM clustering
            +   p5_util.gmm_hyper_parameter_cv()
            |
            +-->build_datakp_bof()
            |   |
            |   +--> ylabel_encode()
            |
            +-->train_test_build()


        
    '''
    
    LIST_PIL_PROCESSOR=[pil_square\
    , pil_edge\
    , p7_filter_median\
    , pil_2gray\
    , pil_autocontrast\
    , pil_equalize\
    , pil_gaussian\
    , pil_gradient\
    , pil_laplacien_kernel_4\
    , pil_laplacien_kernel_8\
    , pil_low_pass]

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def __init__(self, dir_path='./data/Images') :
        if dir_path is not None :
            self._dir_path = dir_path
        else :
            self._dir_path = str()

        self._dict_data = dict()
        self._total_image =0
        self.is_verbose = True
        self._std_size = (200,200)
        self._dict_img_pil = dict() 
        self._dict_breed_kpdesc = dict()
        self._dict_breed_sample = dict()
        self._list_breed_sample = list()
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None
        self._sampling_breed_count = 0
        self._sampling_image_per_breed_count = 0
        self._sampling_image_count = 0
        self._dict_cluster_model = dict()
        self._cluster_model_name = str()
        self._df_bof = pd.DataFrame()
        self._y_label = np.array(0)
        self._dict_breedname_id = dict()
        self._is_splitted = False
        self._Xdesc = np.zeros(128)
        self._ser_breed_number = pd.Series()
        self._dict_classifier = dict()
        self._classifier_name = str()
        self._list_restricted_image = list()
        #self._dict_split_pil_image = dict()
        self._split_ratio = (4,4)
        self._df_pil_image_kpdesc = pd.DataFrame()
        self._is_kp_filtered = True
        self._is_squarred = True
        self._is_random_sampling_image = True
        
        self._dict_pil_processor = dict()
        self._list_processor_id = list()
        
        list_pil_processor, dict_pil_processor \
        = build_dict_processor_list()
        
        self._dict_pil_processor = dict_pil_processor.copy()
        
        # No filter is assgined for proceccing.
        self._list_processor_id = list()#list_pil_processor.copy()
        if False :    
            list_pil_processor = P7_DataBreed.LIST_PIL_PROCESSOR.copy()
            for id, pil_processor in zip(range(0,len(list_pil_processor) )\
            ,list_pil_processor ) :
                self._dict_pil_processor.update({id:pil_processor})
                self._list_processor_id.append(id)
            
        self._image_process_count = 0
        self._list_selected_cluster = list()
        self._pca = None
        
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
    #
    #---------------------------------------------------------------------------
    def show(self, legend=str(), is_show=True):
        '''Show classes attributes
        '''
        if is_show is False :
            return
        
        self.strprint("\n "+str(legend))
        
        self.strprint("Path to data directory ........ : "+str(self._dir_path))
        self.strprint("Number of original breeds ..... : "\
        +str(len(self._dict_data)))
        self.strprint("Total number of images ........ : "\
        +str(self._total_image))       
        self.strprint("Standard images size .......... : "\
        +str(self._std_size))
        self.strprint("SIFT Image descriptors count .. : "\
        +str(len(self._dict_breed_kpdesc)))



        self.strprint("Number of images in sample .... : "\
        +str(self._sampling_image_count))
        self.strprint("Sampling : breeds count ....... : "\
        +str(self._sampling_breed_count))
        self.strprint("Sampling : images per breed ... : "\
        +str(self._sampling_image_per_breed_count))
        
        if self._X_train is not None :
            self.strprint("X train size .................. : "\
            +str(self._X_train.shape))
            self.strprint("y train size .................. : "\
            +str(self._y_train.shape))
        else :
            self.strprint("X train size .................. : "+str(0))
            self.strprint("y train size .................. : "+str(0))
            
        if self._X_test is not None :
            self.strprint("X test size ................... : "\
            +str(self._X_test.shape))
            self.strprint("y test size ................... : "\
            +str(self._y_test.shape))
        else :
            self.strprint("X test size ................... : "+str(0))
            self.strprint("y test size ................... : "+str(0))
        
        self.strprint("Clusters models  .............. : "\
        +str(self._dict_cluster_model.keys()))
        
        self.strprint("Current cluster model  ........ : "\
        +self._cluster_model_name)
        self.strprint("Bag of features dataframe ..... : "\
        +str(self._df_bof.shape))
        self.strprint("Encoded labels from dataset ... : "\
        +str(self._y_label.shape))
        self.strprint("Number of breeds in sample .... : "\
        +str(len(self._ser_breed_number)))
        self.strprint("Image splitted ................ : "\
        +str(self._is_splitted))
        self.strprint("Key point descriptors ......... : "\
        +str(self._Xdesc.shape))
        self.strprint("Classifier name ............... : "\
        +str(self._classifier_name))
        
        self.strprint("Supported classifiers ......... : "\
        +str(list(self.dict_classifier.keys())))

        self.strprint("Number of restricted images ... : "\
        +str(len(self._list_restricted_image)))

        self.strprint("Splitted parts ................ : "\
        +str(self._split_ratio))
        self.strprint("Dataframe images descriptors .. : {} / {}"\
              .format(self._df_pil_image_kpdesc.shape[0]\
              ,self._df_pil_image_kpdesc.columns))      
        self.strprint("KP filtering .................. : "\
        +str(self._is_kp_filtered))
        self.strprint("Squarred images ............... : "\
        +str(self._is_squarred))
        
        self.strprint("Nb of breeds into sampling .... : "\
        +str(len(self._list_breed_sample)))
        
        self.strprint("Random image sampling ......... : "\
        +str(self._is_random_sampling_image))

        self.strprint("Assigned filters identifiers .. : "\
        +str(self._list_processor_id))

        print()
        self.strprint("Assigned filters list ......... : ")
        for key, pil_processor  in self._dict_pil_processor.items() :
            processor_name = p7_util.p7_get_name_from_function(pil_processor)
            print("Identifier : {0}   Filter= {1}".format(key,processor_name))
        print()
        self.strprint("Assignable filters list ....... : ")
        for filter_id, pil_processor  \
        in zip(range(0,len(P7_DataBreed.LIST_PIL_PROCESSOR),1),P7_DataBreed.LIST_PIL_PROCESSOR) :
            processor_name = p7_util.p7_get_name_from_function(pil_processor)
            print("Identifier : {0}   Filter= {1}".format(filter_id,processor_name))


        print()
        self.strprint("Images processed count ........ : "\
        +str(self._image_process_count))

        self.strprint("List of selected clusters ..... : "\
        +str(self._list_selected_cluster))
    
        if self._pca is not None :
            self.strprint("PCA components ................ : "\
            +str(self._pca.n_components_))
        else : 
            self.strprint("PCA components ................ : None")
            

        self.strprint("")

    #---------------------------------------------------------------------------
        
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def copy(self, copied_object, is_new_attribute=True) :
        '''' Copies attributes from object given as parameter into 
        this copied_object.'''
        self._dir_path = copied_object.dir_path

        self._dict_data = copied_object._dict_data.copy()
        self._total_image = copied_object._total_image
        self.is_verbose = copied_object.is_verbose
        self._std_size = copied_object._std_size
        self._dict_img_pil = copied_object._dict_img_pil.copy()
        self._dict_breed_kpdesc = copied_object._dict_breed_kpdesc.copy()
        self._dict_breed_sample = copied_object._dict_breed_sample.copy()
        self._list_breed_sample = copied_object._list_breed_sample.copy()
        if copied_object._X_train is not None :
            self._X_train = copied_object._X_train.copy()
        if copied_object._y_train is not None :
            self._y_train = copied_object._y_train.copy()
        if copied_object._X_test is not None :
            self._X_test = copied_object._X_test.copy()
        if copied_object._y_test is not None :
            self._y_test = copied_object._y_test.copy()
        self._sampling_breed_count =copied_object._sampling_breed_count
        self._sampling_image_per_breed_count \
        = copied_object._sampling_image_per_breed_count
        self._dict_cluster_model = copied_object._dict_cluster_model.copy()
        self._cluster_model_name = copied_object._cluster_model_name
        self._df_bof = copied_object._df_bof.copy()
        self._y_label = copied_object._y_label.copy()
        self._dict_breedname_id = copied_object._dict_breedname_id.copy()
        self._is_splitted = copied_object._is_splitted
        self._Xdesc = copied_object._Xdesc.copy()
        self._ser_breed_number = copied_object._ser_breed_number.copy()    
        self._classifier_name = copied_object._classifier_name
        self._dict_classifier = copied_object._dict_classifier.copy()
        self._list_restricted_image  = copied_object._list_restricted_image.copy()
        #self._dict_split_pil_image = copied_object._dict_split_pil_image.copy()
        self._split_ratio = copied_object._split_ratio
        self._df_pil_image_kpdesc = copied_object._df_pil_image_kpdesc.copy()
        self._is_kp_filtered = copied_object._is_kp_filtered
        self._is_squarred = copied_object._is_squarred
        self._is_random_sampling_image = copied_object._is_random_sampling_image
        self._sampling_image_count = copied_object._sampling_image_count
        self._dict_pil_processor = copied_object._dict_pil_processor.copy()
        self._list_processor_id = copied_object._list_processor_id.copy()
        self._image_process_count = copied_object._image_process_count
        self._list_selected_cluster = copied_object._list_selected_cluster.copy()
        if copied_object._pca is not None :
            self._pca= copied_object._pca
        else : 
            self._pca= None
        
        if is_new_attribute is True :
            pass
        else :
            print("\n*** WARN : new attributes from copied_object are not \
            copied on target!\n")
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    # Properties
    #---------------------------------------------------------------------------
    def _get_dir_path(self) :
      return self._dir_path
    def _set_dir_path(self,dir_path) :
        self._dir_path = dir_path.copy()

    def _get_std_size(self) :
      return self._std_size
    def _set_std_size(self, std_size) :
      self._std_size = std_size

    def _get_df_desc(self) :
      return pd.DataFrame(self._Xdesc)
    def _set_df_desc(self, std_size) :
      print("*** WARN : method not implemented!!")




    def _get_sampling_breed_count(self) :
      return self._sampling_breed_count
    def _set_sampling_breed_count(self, sampling_breed_count) :
      print("*** WARN : method not implemented!!")

    def _get_sampling_image_per_breed_count(self) :
      return self._sampling_image_per_breed_count
    def _set_sampling_image_per_breed_count(self\
    , sampling_image_per_breed_count) :
      print("*** WARN : method not implemented!!")

    def _get_dict_cluster_model(self) :
      return self._dict_cluster_model
    def _set_dict_cluster_model(self, dict_cluster_model) :
      #self._dict_cluster_model = dict_cluster_model.copy()
      self._dict_cluster_model.update(dict_cluster_model)


    def _get_cluster_model_name(self) :
      return self._cluster_model_name
    def _set_cluster_model_name(self, cluster_model_name) :
      if cluster_model_name in self._dict_cluster_model.keys() :
        self._cluster_model_name = cluster_model_name
      else :
        print("\n*** WARN : cluster model= "+cluster_model_name+" does not exists in clusters dictionary !\n")

    def _get_cluster_model(self) :
        if self._cluster_model_name in self._dict_cluster_model.keys() :
            return self._dict_cluster_model[self._cluster_model_name]
        else :
            print("\n*** WARN : cluster model does not exists in clusters dictionary !\n")
    def _set_cluster_model(self, cluster_model) :
            print("\n*** WARN : assignement is not authorized !\n")

    def _get_nb_cluster(self) :
        nb_cluster = 0
        if 'GMM' == self._cluster_model_name :
            nb_cluster \
            = self._dict_cluster_model[self._cluster_model_name].n_components
        elif 'Kmeans' == self._cluster_model_name :
            nb_cluster \
            = self._dict_cluster_model[self._cluster_model_name].n_clusters

        elif 'Hierarchical_clustering' == self._cluster_model_name :
            nb_cluster \
            = self._dict_cluster_model[self._cluster_model_name].n_clusters
        
        else:
            print("\n*** WARN : cluster model does not exists in clusters dictionary !\n")
        return nb_cluster
   
    def _set_nb_cluster(self, nb_cluster) :
        print("\n*** WARN : assignement is not authorized !\n")

    def _get_df_bof(self) :
      return self._df_bof
    def _set_df_bof(self, df_bof) :
      self._df_bof = df_bof.copy()

    def _get_X_train(self) :
      return self._X_train
    def _set_X_train(self, X_train) :
        print("\n*** WARN : assignement is not authorized !\n")

    def _get_y_train(self) :
      return self._y_train
    def _set_y_train(self, y_train) :
        print("\n*** WARN : assignement is not authorized !\n")

    def _get_X_test(self) :
      return self._X_test
    def _set_X_test(self, X_test) :
        print("\n*** WARN : assignement is not authorized !\n")

    def _get_y_test(self) :
      return self._y_test
    def _set_y_test(self, y_test) :
        print("\n*** WARN : assignement is not authorized !\n")

    def _get_ylabel(self) :
      return self._y_label
    def _set_ylabel(self, ylabel) :
        print("\n*** WARN : assignement is not authorized !\n")

    def _get_dict_breedname_id(self) :
      return self._dict_breedname_id
    def _set_dict_breedname_id(self, dict_breedname_id) :
        print("\n*** WARN : assignement is not authorized !\n")

    def _get_is_splitted(self) :
      return self._is_splitted
    def _set_is_splitted(self, is_splitted) :
        print("\n*** WARN : assignement is not authorized !\n")

    def _get_Xdesc(self) :
      return self._Xdesc
    def _set_Xdesc(self, Xdesc) :
        print("\n*** WARN : assignement is not authorized !\n")

    def _get_classifier_name(self) :
      return self._classifier_name
    def _set_classifier_name(self, classifier_name) :
        self._classifier_name = classifier_name

    def _get_dict_classifier(self) :
      return self._dict_classifier.copy()
    def _set_dict_classifier(self, dict_classifier) :
        self._dict_classifier = dict_classifier.copy()

    def _get_classifier(self) :
        classifier_name = self._classifier_name
        classifier = self._dict_classifier[classifier_name]
        return classifier
    def _set_classifier(self, classifier) :
        print("\n*** WARN : assignement is not authorized !\n")
        
    
    def _get_list_restricted_image(self) :
      return self._list_restricted_image.copy()
    def _set_list_restricted_image(self, list_restricted_image) :
        self._list_restricted_image = list_restricted_image.copy()
        self._dict_breed_sample = dict()
        for breed_name, list_image_name in  self._list_restricted_image :
            for image_name in list_image_name :
                part1 = image_name.split('_')[0]
                dirbreed = part1+'-'+breed_name
                self._dict_breed_sample[dirbreed] = list_image_name
                break


    def _get_dict_split_pil_image(self) :
        dict_split_pil_image = dict()
        raw_new = 0
        if 0 == len(self.df_pil_image_kpdesc) :
            print("\n*** WARN : no SIFT descriptors extracted !")
        else :
            for (raw,col) in self.df_pil_image_kpdesc.index :
                if raw == raw_new :
                    breed = self.df_pil_image_kpdesc.loc[(raw,col)][1]
                    breed += "_"+str(raw)
                    
                    arr_raw_image \
                    = self.df_pil_image_kpdesc.loc[raw:raw]['split_image'].values
                    dict_split_pil_image[breed] = list(arr_raw_image)
                    raw_new +=1
        return dict_split_pil_image      
      
      
    def _set_dict_split_pil_image(self, dict_split_pil_image) :
        print("\n*** WARN : assignement is not authorized !\n")

    def _get_split_ratio(self) :
      return self._split_ratio
    def _set_split_ratio(self, split_ratio) :
        self._split_ratio = split_ratio
    
    def _get_df_pil_image_kpdesc(self) :
      return self._df_pil_image_kpdesc
    def _set_df_pil_image_kpdesc(self, Xdesc) :
        print("\n*** WARN : assignement is not authorized !\n")
    
    def _get_is_kp_filtered(self) :
      return self._is_kp_filtered
    def _set_is_kp_filtered(self, is_kp_filtered) :
        self._is_kp_filtered =is_kp_filtered
    
    def _get_is_squarred(self) :
      return self._is_squarred
    def _set_is_squarred(self, is_squarred) :
        self._is_squarred =is_squarred
    
    def _get_list_breed_sample(self) :
      return self._list_breed_sample
    def _set_list_breed_sample(self, list_breed_sample) :

        if (list_breed_sample is not None) and (0 < len(list_breed_sample)):
            self._list_breed_sample = list_breed_sample.copy()
            self._is_random_sampling_image=False
        else :
            self._list_breed_sample = list()
            self._is_random_sampling_image=True
        
    def _get_is_random_sampling_image(is_random_sampling_image) :
      return self._is_random_sampling_image
    def _set_is_random_sampling_image(self, is_random_sampling_image) :
        self._is_random_sampling_image =is_random_sampling_image
    
    def _get_sampling_image_count(self) :
      return self._sampling_image_count
    def _set_sampling_image_count(self, sampling_image_count) :
        self._sampling_image_count =sampling_image_count

    def _get_dict_pil_processor(self) :
      return self._dict_pil_processor
    def _set_dict_pil_processor(self, dict_pil_processor) :
        print("\n*** WARN : assignement is not authorized !\n")
    
    def _get_list_processor_id(self) :
      return self._list_processor_id
    def _set_list_processor_id(self, list_processor_id) :
        self.list_processor_update(None)
        self._list_processor_id = list_processor_id.copy()
        self.list_processor_update(self._list_processor_id)
    

    def _get_image_process_count(self) :
      return self._image_process_count
    def _set_image_process_count(self, image_process_count) :
        print("\n*** WARN : assignement is not authorized !\n")    
    
    def _get_list_selected_cluster(self) :
      return self._list_selected_cluster
    def _set_list_selected_cluster(self, list_selected_cluster) :
      self._list_selected_cluster = list_selected_cluster.copy()
    
    def _get_pca(self) :
        return self._pca
    def _set_pca(self, pca) :
        print("\n*** WARN : assignement is not authorized !\n")    

    
    dir_path = property(_get_dir_path,_set_dir_path)
    std_size = property(_get_std_size,_set_std_size)
    df_desc  = property(_get_df_desc, _set_df_desc)
    
    sampling_breed_count  = property(_get_sampling_breed_count\
    , _set_sampling_breed_count)
    
    sampling_image_per_breed_count=property(_get_sampling_image_per_breed_count\
    , _set_sampling_image_per_breed_count)
    
    dict_cluster_model  = property(_get_dict_cluster_model\
    , _set_dict_cluster_model)
    cluster_model_name = property(_get_cluster_model_name\
    ,_set_cluster_model_name)
    
    cluster_model = property(_get_cluster_model, _set_cluster_model)
    nb_cluster = property(_get_nb_cluster, _set_nb_cluster)
    df_bof = property(_get_df_bof, _set_df_bof)
    X_train = property(_get_X_train, _set_X_train)
    X_test = property(_get_X_test, _set_X_test)
    y_train = property(_get_y_train, _set_y_train)
    y_test = property(_get_y_test, _set_y_test)
    y_label = property(_get_ylabel, _set_ylabel)
    dict_breedname_id = property(_get_dict_breedname_id, _set_dict_breedname_id)
    is_splitted = property(_get_is_splitted, _set_is_splitted)
    Xdesc = property(_get_Xdesc, _set_Xdesc)
    classifier_name = property(_get_classifier_name, _set_classifier_name)
    dict_classifier = property(_get_dict_classifier, _set_dict_classifier)
    classifier = property(_get_classifier,_set_classifier)
    list_restricted_image = property(_get_list_restricted_image\
    ,_set_list_restricted_image)
    dict_split_pil_image = property(_get_dict_split_pil_image\
    ,_set_dict_split_pil_image)
    split_ratio = property(_get_split_ratio, _set_split_ratio)
    df_pil_image_kpdesc = property(_get_df_pil_image_kpdesc\
    , _set_df_pil_image_kpdesc)
    is_kp_filtered = property(_get_is_kp_filtered, _set_is_kp_filtered)
    is_squarred = property(_get_is_squarred, _set_is_squarred)
    list_breed_sample = property(_get_list_breed_sample, _set_list_breed_sample)
    is_random_sampling_image \
    = property(_get_is_random_sampling_image, _set_is_random_sampling_image)
    sampling_image_count \
    = property(_get_sampling_image_count, _set_sampling_image_count)
    dict_pil_processor \
    = property(_get_dict_pil_processor, _set_dict_pil_processor)
    list_processor_id \
    = property(_get_list_processor_id, _set_list_processor_id)
    image_process_count = property(_get_image_process_count\
    , _set_image_process_count)
    
    list_selected_cluster = property(_get_list_selected_cluster\
    , _set_list_selected_cluster)
    
    pca = property(_get_pca , _set_pca)
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def kpdesc_pca_reshape(self, nb_components) :
        ''' Reshape KP array using PCA reduction.
        '''

        #-----------------------------------------------------------------------
        # Get standardized data
        #-----------------------------------------------------------------------
        X_scaled = p3_util.df_get_std_scaled_values(self.df_desc)

        #-----------------------------------------------------------------------
        # Build PCA algorithm
        #-----------------------------------------------------------------------
        self._pca = PCA(n_components=nb_components)
        self._pca.fit(X_scaled)

        X_projected = self._pca.transform(X_scaled)

        #-----------------------------------------------------------------------
        # Replace original data with reduced PCA data.
        #-----------------------------------------------------------------------
        self._Xdesc = X_projected.copy()
    #---------------------------------------------------------------------------


    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def list_processor_update(self, list_processor_id=None, is_verbose=False) :
        if list_processor_id is None :
            list_proc_id, dict_proc_list = build_dict_processor_list()
            self._dict_pil_processor = dict_proc_list.copy()
            self._list_processor_id = None #list_proc_id.copy()
            
        else :
            dict_pil_processor = dict()
            for processor_id in list_processor_id :
                dict_pil_processor[processor_id] \
                = self._dict_pil_processor[processor_id]
            self._dict_pil_processor = dict_pil_processor.copy()
            self._list_processor_id = list_processor_id.copy()

        if is_verbose is True :
            self.strprint("Assignable filters list ....... : ")
            for key, pil_processor  in self._dict_pil_processor.items() :
                processor_name = p7_util.p7_get_name_from_function(pil_processor)
                print("Identifier : {0}   Filter= {1}".format(key,processor_name))
            
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def keras_image_train_test(self, test_size = 0.3, resize=None, square=None) :
        df = self.df_build()
        nClass = len(np.unique(df['label']))

        X_train, X_test, y_train, y_test \
        = p7_keras_X_train_test_build(df['image'], df['label'] \
        , test_size=test_size, square=square, resize=resize)
        
        # X_train and X_test are normalized
        X_train = X_train / 255
        X_test  = X_test / 255

        # y_train and y_test are encoded
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)


        return X_train, X_test, y_train, y_test, nClass
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def df_build(self) :
        '''Build dataframe issue from sampling. 
        Data from sampling is handled by dictionary that is structured as 
        following : {dirbreed:list_of_image_file_name}
        where  : 
            * dirbreed is the directory name for a breed
            * list_of_image_file_name is the list of all image files under 
            dirbreed directory.

        Built dataframe is structured as following : 
            * columns : breed, label, image
                --> breed : human readable breed name
                --> label : encoded label for breed name.
                --> image : content of image in a PIL format.
        '''
        dict_image = dict()
        label = 0
        df = pd.DataFrame()
        new_breedname = str()
        
        for dirbreed, list_imagefilename in self._dict_breed_sample.items():
            for imagefilename in list_imagefilename :
                pil_image = self.read_image(dirbreed, imagefilename)
                if pil_image is not None :
                    breedname = dirbreed.split('-')[1]
                    df = df.append(pd.DataFrame([[breedname, label,pil_image]]\
                    , columns=['breed','label','image']))
                else :
                    pass
            label +=1
        
        df.reset_index(drop=True, inplace=True)
        print("Image count = "+str(len(df)))
        return df
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def plot_kpdesc_image(self, is_plot=True) :
        '''
        Plot a single image along with its KP points issued from SIFT features 
        extraction.

        Input : None
            
        Output : 
            * list_breed_kpdesc : list of tuples structured as following :
            (kp,descriptor), for each raw of splitted image.
            * dict_breed_kpdesc_image : dictionary structures as following :
            {breed: list_of_cv2.drawKeypoints}, for each raw of splitted image.
        '''

        raw_new = 0
        list_kpdesc = list()
        list_image_pil = list()

        raw=0
        col=0
        breedname = self.df_pil_image_kpdesc.loc[(raw,col)][1]
        for (raw,col) in self.df_pil_image_kpdesc.index :
            desc = self.df_pil_image_kpdesc.loc[(raw,col)][0]
            kp = self.df_pil_image_kpdesc.loc[(raw,col)][2]
            pil_image = self.df_pil_image_kpdesc.loc[(raw,col)][4]
            list_kpdesc.append((kp, desc))
            list_image_pil.append(pil_image)
        
        dict_pil_image_   = {breedname:list_image_pil}
        dict_breed_kpdesc = {breedname:list_kpdesc}        


        dict_breed_kpdesc_image = dict()
        dict_breed_kp_image = dict()
        count=0
        for (breed, list_breed_kpdesc)\
        , list_image_pil in zip(dict_breed_kpdesc.items(), dict_pil_image_.values()):
            count +=1
            dict_breed_kpdesc_image[breed] \
            = [cv2.drawKeypoints(np.array(image_pil), kp, np.array(image_pil)) \
                for ((kp, desc),image_pil) in zip(list_breed_kpdesc,list_image_pil)]

        raw = self._split_ratio[0]
        col = self._split_ratio[1]

        breedname = list(dict_breed_kpdesc_image.keys())[0]

        arr_= np.array(dict_breed_kpdesc_image[breedname])

        dict_breed_kpdesc_image_raw = dict()
        col_start = 0
        for i_raw in range(0,raw) :
            col_end = col_start+col
            dict_breed_kpdesc_image_raw.update({i_raw:arr_[col_start:col_end,::,::,::]})
            col_start =col_end

        if is_plot is True :
            p7_util.p7_image_pil_show(dict_breed_kpdesc_image_raw\
                                      ,size_x=10,std_image_size=None\
                                      ,is_title=False)

        return list_breed_kpdesc, dict_breed_kpdesc_image_raw
    #---------------------------------------------------------------------------


    #-------------------------------------------------------------------------------
    #
    #-------------------------------------------------------------------------------
    def image_explore(self,breed_name_, image_name, is_show=False\
    , is_squarred=True, std_size=None, is_plot=True) :
    
        # breed_name
        breed_name = get_breedname_from_dirbreed(breed_name_)
        
        oP7_DataBreed_single = P7_DataBreed()
        
        oP7_DataBreed_single.copy(self)
        if False :
            oP7_DataBreed_single.is_squarred = is_squarred
            oP7_DataBreed_single.std_size = std_size
            
            oP7_DataBreed_single._dict_pil_processor = self._dict_pil_processor.copy()
            oP7_DataBreed_single._list_processor_id = self._list_processor_id.copy()

        
        
            oP7_DataBreed_single._dict_breed_sample=self._dict_breed_sample.copy()

        oP7_DataBreed_single.show(is_show=is_show)
        list_restricted_image = [(breed_name,[image_name])]

        oP7_DataBreed_single.list_restricted_image = list_restricted_image
        oP7_DataBreed_single.show(is_show=is_show)
        
        
        oP7_DataBreed_single.build_sift_desc(is_splitted=True)

        oP7_DataBreed_single.show(is_show=is_show)
        if is_plot is True :
            p7_util.p7_image_pil_show(oP7_DataBreed_single.dict_split_pil_image\
                                      ,std_image_size=None, is_title=False)
        return oP7_DataBreed_single
    #-------------------------------------------------------------------------------


    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def predict_image(pil_image) :
        pass
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def dict_classifier_load(self,filename):
        self.dict_classifier = p5_util.object_load(filename)
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def get_image_filename(self, breedname, imagename) :
        '''Build and returns absolute path of image file name using breedname 
        and image name. 
        Images identifiers are stored into dictionary self._dict_breedname_id.
        Directory holding images belonging to a breed is named as following : 
            --> 'id-breedname' such as n02107142-Doberman
        '''
        id_breedname = self._dict_breedname_id[breedname]
        dirbreed = id_breedname+'-'+breedname
        image_filename = self._dir_path+'/'+dirbreed+'/'+imagename
        return image_filename
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def load(self, list_dirbreed=None, imagename=None) :
        '''Read all images from data directory path .
        Images are stored into a dictionary with such structure : 
            --> {breed_directory : list_of_breed_file_names} 
            where :
                -> breed_directory : is the directory containing images for a 
                breed.
                When value is None, then all directories will be scaned.
                
                -> list_of_breed_file_names is the list of file names that 
                reference images from breed.
                When None, then all images files names contained into each 
                directory will be loaded.
        '''
        self._dict_data = dict()
        self._total_image = 0
        
        self._dict_data = p7_util.p7_load_dict_filename(self._dir_path\
        , list_dirbreed=list_dirbreed)
        
        if imagename is not None :
            self._dict_data = {list_dirbreed[0]:[imagename]}

        #-----------------------------------------------------------------------
        # Number of breeds in sample is updated.
        #-----------------------------------------------------------------------
        if list_dirbreed is not None :
            self._sampling_breed_count = len(list_dirbreed)
            
        #-----------------------------------------------------------------------
        # Deactivate random sampling 
        #-----------------------------------------------------------------------
        if len(list_dirbreed) >0 :
            if imagename is not None : 
                self._is_random_sampling_image = False
        
        #-----------------------------------------------------------------------
        # Total number of images files is computed.
        #-----------------------------------------------------------------------
        self._total_images = 0
        for breed in self._dict_data.keys():
            self._total_image += len(self._dict_data[breed])              
            
        if 0 == len(self._dict_img_pil) :
            #-------------------------------------------------------------------
            # Each image is read from data directory and resized.        
            #-------------------------------------------------------------------
            for dirbreedname, list_filename in self._dict_data.items():
                list_image_pil = list()
                for filename in list_filename :
                    #-----------------------------------------------------------
                    # Path file for image access is built
                    #-----------------------------------------------------------
                    pathfilename = self._build_pathname(dirbreedname, filename)
                    image_pil = p7_util.p7_pil_image_load(pathfilename\
                    ,is_verbose=False, std_size=None)

                    #-----------------------------------------------------------
                    #Image is resized and stored in list of images for this breed
                    #-----------------------------------------------------------
                    #list_image_pil.append(image_pil.resize(self._std_size))
                    list_image_pil.append(image_pil)

                #---------------------------------------------------------------
                # List of resized images is stored into dictionary
                #---------------------------------------------------------------
                self._dict_img_pil[dirbreedname] = list_image_pil
        else :
            pass
        #-----------------------------------------------------------------------
        # Update sampling data
        #-----------------------------------------------------------------------
        self._dict_breed_sample = self._dict_data.copy()
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _build_pathname(self, dir_breed, filenamebreed):
        '''Build absolute path name from path dir, and given parameters that are :
        Input : 
            * dir_breed : directory breed name 
            * filenamebreed : a file name located into dir_breed referencing 
            an image.
        Output :
            * absolute file path name.
        '''
        return self._dir_path+'/'+dir_breed+'/'+filenamebreed
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def std_size_build(self) :
        ''' Compute the standard size (weight, height) for the hole images 
        dataset. Standard size is computed from median value.
        
        This standard size will be applied to all images.
        
        '''

        #-----------------------------------------------------------------------
        # A standard image size is computed.
        # This is the most frequent number of pixels X and the most frequent 
        # number of pixels for Y.
        #-----------------------------------------------------------------------
        image_count=0
        df= pd.DataFrame()

        if(0 < self._total_image) :
            for dirbreed, list_imagename in self._dict_data.items():
                for imagename  in list_imagename :
                    imagepathname = self._build_pathname(dirbreed, imagename)
                    pil_image = p7_util.p7_pil_image_load(imagepathname\
                    , is_verbose=False, std_size=None)
                    df[image_count] = (pil_image.size[0],pil_image.size[1])
                    image_count +=1
                    pil_image.close()
                    
            size_x = int(df.iloc[0,:].median())
            size_y = int(df.iloc[1,:].median())
            del(df)        
            self._std_size = (size_x,size_y)
        else :
            print('\n*** ERROR : Empty data structure holding images')        
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def pil_2gray(self, pil_image):
        return pil_image.convert('L')
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def pil_equalize(self, pil_image):
        return ImageOps.equalize(pil_image)
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def pil_resize(self, pil_image):
        return pil_image.resize(self._std_size)
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def split_pil_image(self, pil_image, labelname,ratio=(4,4)):
        '''Split a PIL formated image given the ratio (weight,height).
        
        Input : 
            * pil_image : image to be spitted
            * labelname : image label in order to identify a part of splited 
            image. This label may be relevant for display.
            * ratio : tuple providing number of images for weight and number of 
            images for height.
        Output :
            * dictionary structured as following : {label_i:list_of_pil_image}
            where : 
                --> label_i is a label identifying the ith raw and 
                --> list_of_pil_image is the list of PIL images in the ith raw.

            Such output is usefull for image display.
        '''
        #print("split_pil_image() : ratio= "+str(ratio))
        if self._split_ratio is not None :
            ratio = self._split_ratio
        else :
            pass 

        if self._std_size is None :
            width  = int(pil_image.size[0]/ratio[0])
            height = int(pil_image.size[1]/ratio[1])
        else:        
            width  = int(self._std_size[0]/ratio[0])
            height = int(self._std_size[1]/ratio[1])
        
        dict_pil_image = dict()
        imgwidth, imgheight = pil_image.size
        # Image is resized in order to avoid border when splitting it
        imgwidth  = width*ratio[0]
        imgheight = height*ratio[1]
        pil_image = pil_image.resize((imgwidth,imgheight))
                
        #print("split_pil_image() : (imgwidth, imgheight)= "+str(pil_image.size))
        #print("split_pil_image() :( width, height)= "+str((width, height)))         
        #print("*** split_pil_image() : {} {}".format((imgwidth, imgheight),(width,height)))
        i_row = 0
        for i in range(0,imgheight,height):
            list_pil_image_crop = list()
            for j in range(0,imgwidth,width):
                box = (j, i, j+width, i+height)
                #print("split_pil_image : box= {}".format(box))
                list_pil_image_crop.append(pil_image.crop(box))
            labelname_i = str(i)+'_'+labelname
            dict_pil_image[labelname_i]=  list_pil_image_crop  
            i_row +=1
        #print("split_pil_image : rows per image = "+str(i_row))
        return dict_pil_image
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def build_arr_desc(self):
        '''Build an array (Nx128) where :
        --> N : is the total number of keypoints for the dataset.
        --> 128 is the number of SIFT descriptors per keypoints.
        Arrays of keypoints descriptors are stored into a dictionary.
        Building array of descriptors leads to stack each one of these arrays.
        
        '''
        X_desc = np.zeros(128)
        error=0
        count= 0
        #-----------------------------------------------------------------------
        # Dictionary _dict_breed_kpdesc is structured as {id:(desc,name)}
        # --> desc is an array of key-points descriptors with 128 columns.
        # --> name is the breed name, usefull for classification.
        # --> id is the directory identifier breed images are stored in.
        #-----------------------------------------------------------------------
        rows = len(self._dict_breed_kpdesc)
        if(0 >= rows):
            print("*** ERROR : empty Key-points descriptors! \
            build it with build_sift_desc()")
            return

        for desc in self._df_pil_image_kpdesc.desc.values:
            try :
                X_desc = np.vstack((X_desc,desc))
            except ValueError:
                error +=1
                pass
            count+=1
            if count%1000==0 :
                print("Processed rows= "+str(count)+"/"+str(rows))
        if 0 < error :
            print("\n*** WARN : Nb of exceptions during process ... : "\
            +str(error))
        # Copy result except fist column that matches to initialization.
        self._Xdesc = X_desc[1:].copy()
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _df_pil_image_kpdesc_build(self, array_values):
        '''Reset dataframe holding PIL images along with corresponding 
        KP and descriptors. 
        
        Dataframe is built with a 2-levels indexes. 
            --> 1st level index references rows of spliltted images.
            --> 2nd index references columns for splitted images in a row. 
        Number of rows and columns are issued from split image process. This 
        process is leaded by parameter self.split_ratio.
        
        Input :
            * array_values : array of values : KP, descriptors, image size, 
            breed name.
        Output :
            * dataframe with 2-levels indexes and values issued from array_values.
        '''
        
        #-----------------------------------------------------------------------
        # number of columns is computed based on a square split for any image.
        #-----------------------------------------------------------------------
        #print("*** _df_pil_image_kpdesc_build() :{} ".format(array_values.shape))
        
        if self._is_splitted is False :
            row=1
            col=1
        else :
            if self._is_squarred is True :
                col = int(np.sqrt(array_values.shape[0]))
                row = col 
            else :
                row = self.split_ratio[0]
                col = self.split_ratio[1]
        
        

        row_index =np.arange(0,row*col,1)
        col_index =np.arange(0,row*col,1)

        #-----------------------------------------------------------------------
        # Index for rows and columns initialization
        #-----------------------------------------------------------------------
        row_index[:]=0
        col_index[:]=0

        for i in range(0,row*col, row):
            row_index[i:i+col] = int(i/col)

        for i in range(0,row*col, col):
            col_index[i:i+col] = range(0,col,1)

        list_level_index=[row_index,col_index]
        #print("_df_pil_image_kpdesc_build :")
        #print(list_level_index, array_values.shape)
                
        df_multi_level \
        = pd.DataFrame(array_values\
        , columns=['desc','breed','kp','size','split_image','image_id']\
        , index=list_level_index)

        #print(df_multi_level)

        return df_multi_level
    #---------------------------------------------------------------------------    

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def kpdesc_build(self, dirbreed, pil_image, image_count,imagename) :
        '''Build matrix of key points descriptors where  :
        --> number of rows from this matrix is the number of keypoints
        --> number of columns is the number of features (128) for each keypoint.

        Build is processed following is_splitted flag. When activated, then 
        image is splitted and matrix is built over each one of the splitted 
        images.
        
        Input :
            * dirbreed : directory in which all images lays on.
            * pil_image : PIL image from which KPDESC matrix is built.
            * image_count : this is an incremental value used to build rows 
            for KPDESC matrix. One row represents one image.
            
        Output : 
            * dict_breed_kpdesc : KPDESC matrix for any image stored in a 
            dictionary structured as following : {image_id:(desc,hr_breedname)},
            where :
                --> image_id : is an identifier for current image
                --> desc : this is the descriptor vector (128 sized) for image 
                identified with image_count
                --> hr_breedname : human readable name of the breed.
            * image_count : current number of images.
        
        '''

        dict_breed_kpdesc = dict()
        hr_breedname = get_breedname_from_dirbreed(dirbreed)
        name_id = imagename.split('.')[0]
           
        if self._is_splitted is True :            
            #-------------------------------------------------------------------
            # dict_split_pil_image holds informations from pil_image image. 
            #-------------------------------------------------------------------
            dict_split_pil_image \
            = self.split_pil_image(pil_image,hr_breedname\
            , ratio=self._split_ratio)
            
            
            
            for id_breedname, list_split_pil_image \
            in dict_split_pil_image.items() :
            
                for split_pil_image in list_split_pil_image :
                    kp, desc = get_image_kpdesc(split_pil_image)
                    
                    dict_breed_kpdesc[image_count] \
                    = (desc,hr_breedname,kp,split_pil_image.size\
                    ,split_pil_image,name_id)
                    
                    image_count +=1

            self._dict_breed_kpdesc.update(dict_breed_kpdesc)

        else :
            kp, desc = get_image_kpdesc(pil_image)
            dict_breed_kpdesc[image_count] \
            = (desc,hr_breedname, kp,pil_image.size\
            ,pil_image,name_id)

            self._dict_breed_kpdesc.update(dict_breed_kpdesc)
        
        #-----------------------------------------------------------------------
        # Dataframe with all informations related to PIL images and descriptors 
        #-----------------------------------------------------------------------
        ar = np.array(list(dict_breed_kpdesc.values()))
        #print("*** INFO : kpdesc_build() : {}".format(dict_breed_kpdesc.keys()))
        df_multi_level = self._df_pil_image_kpdesc_build(ar)

        self._df_pil_image_kpdesc  \
        = pd.concat( [self._df_pil_image_kpdesc, df_multi_level])

        return dict_breed_kpdesc, image_count
    #---------------------------------------------------------------------------
    
        
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def apply_pil_processor(self, pil_image, is_verbose=False) :
        
        if is_verbose is True :
            for pil_processor in self._dict_pil_processor.values():
                name_processor = p7_util.p7_get_name_from_function(pil_processor)
                print("Applied filter : "+str(name_processor))
        else :
            pass
        
        for pil_processor in self._dict_pil_processor.values():
            pil_image = pil_processor(pil_image)
        return pil_image
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def build_sift_desc(self, is_splitted=False) :
        '''Build SIFT descriptors from dictionary _dict_breed_sample.
        This dictionary is structured as following : {dirbreed:list_imagename}
        where : 
            --> dirbreed is the directory name for a breed containing all images
            --> list_imagename is the file name of all images under dirbreed.
        Images are not stored in memory. They are loaded from information into 
        _dict_breed_sample.
        
        Filters are applied before SIFT descriptors to be built.
        
        '''
        image_count=0
        error = 0

        ratio = 5/100
        self._dict_breed_kpdesc = dict()
        self._is_splitted = is_splitted
        self._df_pil_image_kpdesc = pd.DataFrame()
        
        print("*** build_sift_desc() ...")
        if is_splitted is True :
            if self._std_size is None :
                print("*** ERROR : Images require square resizing for splitting!")
                return
            else :
                pass
        else :
            pass
        self.build_ser_number_breedname()
        for dirbreed, list_imagename in self._dict_breed_sample.items():
            
            for imagename  in list_imagename :
                #print(imagename)
                
                #---------------------------------------------------------------
                # Load PIL image from file path
                #---------------------------------------------------------------
                imagepathname = self._build_pathname(dirbreed, imagename)
                pil_image = p7_util.p7_pil_image_load(imagepathname\
                , is_verbose=False, std_size=None)

                #---------------------------------------------------------------
                # Resize
                #---------------------------------------------------------------
                try :
                    if self.std_size is not None :
                        pil_image = pil_image.resize(self._std_size)
                        #pil_image = pil_truncate(pil_image,self.std_size)
                    else :
                        pass

                    pil_image = self.apply_pil_processor(pil_image)

                    #-----------------------------------------------------------
                    # Store descriptor along with breed name. This will be 
                    # usefull for classification.
                    #-----------------------------------------------------------
                    dict_breed_kpdesc, image_count\
                    = self.kpdesc_build(dirbreed, pil_image,image_count\
                    ,imagename)
                    
                    #-----------------------------------------------------------
                    # Closing PIL image : all resources of PIL image are released.
                    #-----------------------------------------------------------
                    pil_image.close()

                    #-----------------------------------------------------------
                    # Increase number of processed image in case is_splitted 
                    # flag in not activated.
                    # When activated, number of processed images is increased 
                    # inside kpdesc_build() method.
                    #-----------------------------------------------------------
                    if self._is_splitted is False :
                        image_count +=1     
                    
                except AttributeError :
                    error +=1
                    #print("*** WARNING : attribute error for PIL image ")
                    continue                
        self._df_pil_image_kpdesc.index.names =['raw','col']
        print("\nINFO : Error = "+str(error)\
        +" Total images processed= "+str(image_count))
        self._image_process_count = image_count
    #---------------------------------------------------------------------------



    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def sampling(self, breed_count, image_per_breed_count):
        #-----------------------------------------------------------------------
        # Select randomly a breed directory name; a list of randomly selected
        # breeds directory is built.
        #-----------------------------------------------------------------------
        #self._list_breed_sample = list()
        if True == self._is_random_sampling_image :
            for breed_id in range(0, breed_count,1):
                choice = random.choice(list(self._dict_data.keys()))
                self._list_breed_sample.append(choice)
        else :
            breed_count = len(self._list_breed_sample)
            
        #-----------------------------------------------------------------------
        # For each selected breed, a random list of images is selected when 
        # list_breed_dir is None.
        # Otherwise, 
        #-----------------------------------------------------------------------
        count=0           
        self._dict_breed_sample = dict()

        #-----------------------------------------------------------------------
        # When image_per_breed_count < 0 then all images from breed directory 
        # are loaded.
        #-----------------------------------------------------------------------
        if 0 > image_per_breed_count :
            image_per_breed_count = int(1e6)
        
        for breedname in self._list_breed_sample :
            list_filename = self._dict_data[breedname]
            list_file_sample = list()
            selected_images = min(image_per_breed_count,len(list_filename))
            #print("*** sampling() : {}".format(selected_images))
            if False :
                for file_id in range(0, selected_images,1):
                    list_file_sample.append(random.choice(list_filename))
                    count +=1
            else :
                list_file_sample =list_filename[0:selected_images]
            #-------------------------------------------------------------------
            # Random may lead to duplication; in order to avoid it, 
            # a unique list of names is created using pandas Series object 
            # from dictionary. 
            #-------------------------------------------------------------------
            list_file_sample_unique = list(pd.Series(list_file_sample).unique())
            
            self._dict_breed_sample[breedname] = list_file_sample_unique
            
        self._sampling_breed_count = breed_count
        self._sampling_image_per_breed_count = image_per_breed_count
        self.build_ser_number_breedname()
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def read_image(self, dirbreed, imagename, image_type='PIL') :
        pil_image = None
        imagepathname = self._build_pathname(dirbreed, imagename)
        pil_image = p7_read_image(imagepathname)
        if pil_image is None : 
            breedname = dirbreed
            dirbreed = imagename.split('_')[0]
            dirbreed = dirbreed+'-'+breedname
            imagepathname = self._build_pathname(dirbreed, imagename)
            pil_image = p7_read_image(imagepathname)
            if pil_image is None : 
                print("*** ERROR : re-building breed directory name FAILED!")
            else :
                pass
        return pil_image
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def cluster_model_predict(self, desc) :
        '''This is an abstraction layer to call predict() method depending of 
        clustering model.
        '''
        if 'Hierarchical_clustering' == self._cluster_model_name :
            return self.cluster_model.fit_predict(desc)
        elif 'GMM' == self._cluster_model_name :
            return self.cluster_model.predict(desc)
        elif 'Kmeans' == self._cluster_model_name :
            return self.cluster_model.predict(desc)
        else :
            print("*** ERROR : no cluster_model_predict() method \
            implementation for model= "+str(self._cluster_model_name))
        return None
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def train_test_build(self, size_test=0.1):

        self._X_train, self._X_test, self._y_train,  self._y_test \
        = model_selection.train_test_split(self._df_bof,self._y_label\
        ,test_size=size_test)
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def get_cluster_from_imagedesc(self, desc):
        '''Build histogram of clusters for image descriptors given as parameter
        and convert result into a dataframe.
        
        Input : 
            * desc : image represented as SIFT key points descriptors.
            rows are keypoints, extended over 128 descriptors (columns)
        Output :
            * dataframe containing histogram of clusters representing 
            image bag of visual words.
            dataframe rows : images identifiers
            dataframe columns : descriptors occurencies 
        '''
        
        #-----------------------------------------------------------------------
        # Get current cluster modeler and number of clusters
        #-----------------------------------------------------------------------
        nb_cluster = self.nb_cluster
        df = None
        
        if 0 >= nb_cluster :
            print("\n*** ERROR : No cluster into data model!")
        else:
            cluster_model = self.cluster_model
            df=pd.DataFrame(np.zeros(nb_cluster, dtype=int))
            #-------------------------------------------------------------------
            # Initialization : histogram is placed into a dictionary.
            #-------------------------------------------------------------------
            dict_feature = dict()
            for i in range(0,nb_cluster) :
                dict_feature[i]=0     
            #-------------------------------------------------------------------
            # Get cluster from image represented as Key points descriptors
            #-------------------------------------------------------------------
            try :
                y_label = self.cluster_model_predict(desc)
                #print("get_cluster_from_imagedesc : Label= "+str(y_label))
            except ValueError:
                #print("\n*** get_cluster_from_imagedesc() : Error on desc array")
                return None

            #-------------------------------------------------------------------
            # Build histogram of clusters for this image and convert it into 
            # dataframe.
            #-------------------------------------------------------------------
            for label in set(y_label) :
                dict_feature[label] = np.where(y_label==label)[0].shape[0]
            df_tmp = pd.DataFrame(np.array(list(dict_feature.values())), \
            index=dict_feature.keys())
            df =df_tmp.T
            
            #-------------------------------------------------------------------
            # Selected cluster list is applied to dataframe columns.
            #-------------------------------------------------------------------
            list_cluster = self._list_selected_cluster
            if 0 < len(list_cluster) :
                list_del_col = [col for col in df.columns.tolist() \
                if col not in list_cluster ]
                for col in list_del_col :
                    del(df[col])

        return df
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def get_breedlabel_from_breedname(self,breedname) :
        label = np.where(np.array(self._ser_breed_number.values)==breedname)[0][0]
        return label
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def get_breedname_from_breedlabel(self, breedlabel) :
        return self._ser_breed_number[breedlabel]
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def ylabel_encode(self,dict_label):
        list_tags_ref = list(self._ser_breed_number.keys())
        print("Number of referenced labels= "+str(len(list_tags_ref)))

        list_list_tags = [[tag] for tag in list(dict_label.values())]
        print("Number of labels to be encoded= "+str(len(list_list_tags)))
        y_label = p6_util.p6_encode_target(list_tags_ref, list_list_tags)
        self._y_label =  np.array(y_label).copy()
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def build_datakp_bof(self) :
        '''Build representation of key-points dataset in a bag of features.
        Result is normalized and stored into a pandas data-frame.
        Clustering should have been built before this step.
        Output :
            * dataframe structured as following : 
                --> Rows index are references to images from sampling 
                --> Columns are features occurencies.
        '''
        df = None
        dict_label = dict()
        error =0
        list_index_error = list()
        #-----------------------------------------------------------------------
        # Dataframe index are converted as 2 levels indexes : 
        # index for rows and for each row, index for columns.
        #-----------------------------------------------------------------------
        self._df_pil_image_kpdesc.reset_index(drop=True, inplace=True)

        #-----------------------------------------------------------------------
        # Each descriptor from any row of dataframe is picked, PCA reduction 
        # is applied on it. 
        # index for rows and for each row, index for columns.
        #-----------------------------------------------------------------------        
        for index in range(0, len(self._df_pil_image_kpdesc)) : 
            imagedesc= self._df_pil_image_kpdesc.desc.iloc[index]
            breedname = self._df_pil_image_kpdesc.breed.iloc[index]
            if self._pca is not None :
                if imagedesc is not None :
                    imagedesc \
                    = p3_util.df_get_std_scaled_values(pd.DataFrame(data=imagedesc))
                    imagedesc = self.pca.transform(imagedesc)
                else :
                    pass

            df_tmp = self.get_cluster_from_imagedesc(imagedesc)
            if df_tmp is None :
                error +=1
                list_index_error.append(index)
            else :            
                # Index is matched with index
                df_tmp.rename(index={0:index}, inplace=True)
                if df is None :
                    df = df_tmp.copy()
                else:
                    df = pd.concat([df, df_tmp])
                # Used for Y label
                breedlabel = self.get_breedlabel_from_breedname(breedname)
                
                # Assign a label for each image.
                dict_label[index] = breedlabel

        print("\n***Nb of errors..............= "+str(error))
        print("\n***Nb of labelized images ...= "+str(len(dict_label)))

        
        #-----------------------------------------------------------------------
        # Labels are encoded in a multi-class way
        #-----------------------------------------------------------------------
        self.ylabel_encode(dict_label) 
        
        #-----------------------------------------------------------------------
        # L2 Normalization with :
        # * L2 per column is computed summing all terms for any column
        # * Normalization is applied on all terms or any column 
        #-----------------------------------------------------------------------

        ser_sqrt = df.apply(lambda x: np.sqrt(x.dot(x.T)), axis=0)
        ser_sqrt
        df = df/ser_sqrt
    
        self._df_bof = df.copy()
        
        #-----------------------------------------------------------------------
        # Reset index removing gaps due to errors from cluster assignement.
        #-----------------------------------------------------------------------
        self._df_bof.reset_index(drop=True, inplace=True)
        
        if 0 < error:
            print("\n***WARN : build_datakp_bof() : errors= "+str(error))
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def build_datakp_bof_deprecated(self) :
        '''Build representation of key-points dataset in a bag of features.
        Result is normalized and stored into a pandas data-frame.
        Clustering should have been built before this step.
        Output :
            * dataframe structured as following : 
                --> Rows index are references to images from sampling 
                --> Columns are features occurencies.
        '''
        df = None
        dict_label = dict()
        error =0
        list_image_id_error = list()
        self._df_pil_image_kpdesc.reset_index(drop=True, inplace=True)

        for image_id in self._df_pil_image_kpdesc.desc.index: 
            breedname = self._df_pil_image_kpdesc.breed[image_id]       
            imagedesc = self._df_pil_image_kpdesc.desc[image_id]
            df_tmp = self.get_cluster_from_imagedesc(imagedesc)
            if df_tmp is None :
                error +=1
                list_image_id_error.append(image_id)
            else :            
                # Index is matched with image_id
                df_tmp.rename(index={0:image_id}, inplace=True)
                if df is None :
                    df = df_tmp.copy()
                else:
                    df = pd.concat([df, df_tmp])
                # Used for Y label
                breedlabel = self.get_breedlabel_from_breedname(breedname)
                
                # Assign a label for each image.
                dict_label[image_id] = breedlabel

        print("\n***Nb of errors..............= "+str(error))
        print("\n***Nb of labelized images ...= "+str(len(dict_label)))

        if False :
            #-----------------------------------------------------------------------
            # Errors recorded during BOF construction are removed from dictionary.
            #-----------------------------------------------------------------------
            for image_id_error in list_image_id_error :
                del(self._dict_breed_kpdesc[image_id_error] )
        
        #-----------------------------------------------------------------------
        # Labels are encoded in a multi-class way
        #-----------------------------------------------------------------------
        self.ylabel_encode(dict_label) 
        
        #-----------------------------------------------------------------------
        # L2 Normalization with :
        # * L2 per column is computed summing all terms for any column
        # * Normalization is applied on all terms or any column 
        #-----------------------------------------------------------------------

        ser_sqrt = df.apply(lambda x: np.sqrt(x.dot(x.T)), axis=0)
        ser_sqrt
        df = df/ser_sqrt
    
        self._df_bof = df.copy()
        
        #-----------------------------------------------------------------------
        # Reset index removing gaps due to errors from cluster assignement.
        #-----------------------------------------------------------------------
        self._df_bof.reset_index(drop=True, inplace=True)
        
        if 0 < error:
            print("\n***WARN : build_datakp_bof() : errors= "+str(error))
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def breed_show(self) :
        print("")
        self.build_ser_number_breedname()

        for name,id in self._dict_breedname_id.items() :
            print("Breed directory= {:30}  Breed name= {}"\
            .format(str(id)+'-'+name,name))
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def build_ser_number_breedname(self) :
        '''Build, from sampling dataset, a pandas Series with index 
        as breed labels and human readable breed names as values.
        
        Series is strictured as following : 
        --> keys : [list_of_labels]
        --> values : [array_of_breed_name]
        '''
        if 0 >= len(self._dict_breedname_id) :
            pass
        else :
            print("\nINFO : Series already built!")

        dict_breedname_id=dict()
        if 0 < len(self._dict_breed_sample) :
            print("Building...")
            for dirbreedname in self._dict_breed_sample.keys() :
                id  = dirbreedname.split('-')[0]
                breedname = dirbreedname.split('-')[1]
                dict_breedname_id[breedname] = id
        else:
            print("\n*** WARN : empty breed in list!")
        
        self._dict_breedname_id = dict_breedname_id.copy()

        #-----------------------------------------------------------------------
        # Build pandas Series with labeled breed names.
        #-----------------------------------------------------------------------
        label = 0
        dict_breed_number = dict()
        for breed_name in self._dict_breedname_id.keys() :
            dict_breed_number[label] = breed_name
            label+=1
        self._ser_breed_number = pd.Series(dict_breed_number).copy()
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def show_image_name(self, breedname, is_sample_show=False):
        '''Display images files names issued from sampling from a human readable 
        breedname.
        
        Directory name of breeds images is built from human readable breedname.
        List of images files is built from this directory and is displayed.
        
        Input :
            * breedname : human readable breed name.
            * is_sample_show : when fixed to False, then images stored in sample 
            are not displayed. Then only images names out of sample are displayed.
            This may be relevant when testing image out of sample.
            
            Images from sample are those used for training a ML or DL algorithm.
        '''

        ser = pd.Series(self._dict_breedname_id)
        
        tuple_array = np.where(ser.keys()==breedname)
        
        if 0 == len(tuple_array[0]) :
            breedname = breedname.split('-')[1]
            tuple_array = np.where(ser.keys()==breedname)
        
        if 0 < len(tuple_array[0]):
            index = tuple_array[0][0]
            id = list(self._dict_breedname_id.values())[index]
            breedid = str(id)+'-'+breedname
            dirbreed = self._dir_path+'/'+breedid
            
            list_image_name = os.listdir(dirbreed)
            
            print("Directory breed name = "+dirbreed)
            
            #-----------------------------------------------------------------------
            # Do not show images stored in sampling when is_sample_show is False
            #-----------------------------------------------------------------------
            list_sample_breed_image = list()
            if is_sample_show is False :
                list_sample_breed_image = self._dict_breed_sample[breedid]


            print("")
            count_image_sample_show = len(list_sample_breed_image)
            print("Number of images ="\
            +str(len(list_image_name)-count_image_sample_show))
            for image_name in list_image_name :
                if image_name not in list_sample_breed_image :
                    print("Image name= {}".format(image_name))
        else : 
            print("\n*** ERROR : breadname= "+str(breedname)+" not found into sample! Enter show_breed_name()\n")
            
        
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def predict(self,dirbreed,imagename, top=3):
        self.load(list_dirbreed=[dirbreed], imagename=imagename)
        
        #-----------------------------------------------------------------------
        # SIFt descriptors are built
        #-----------------------------------------------------------------------
        self.build_sift_desc(is_splitted=True)
        self.build_arr_desc()


        #-----------------------------------------------------------------------
        # Bag Of Feature is built
        #-----------------------------------------------------------------------
        self.build_datakp_bof()
        
        #-----------------------------------------------------------------------
        # Classification take place
        #-----------------------------------------------------------------------
        classifier = self.classifier
        result = classifier.predict(self.df_bof)
        #print(classifier.predict_proba(self.df_bof))

        #-----------------------------------------------------------------------
        # Sum over each column is computed; result is sorted.
        #-----------------------------------------------------------------------
        ser = pd.DataFrame(result).apply(lambda x:x.sum())
        ser.sort_values(ascending=False, inplace=True)

        #-----------------------------------------------------------------------
        # Get breed label
        #-----------------------------------------------------------------------
        list_predicted = list()
        for breedlabel, value in ser[:top].items():
            #-------------------------------------------------------------------
            # Get breed name
            #-------------------------------------------------------------------
            breedname = self.get_breedname_from_breedlabel(breedlabel)
            list_predicted.append(breedname)

        #-----------------------------------------------------------------------
        # Get breed name
        #-----------------------------------------------------------------------
        breedname = get_breedname_from_dirbreed(dirbreed)
        return breedname, list_predicted

    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def show_breed_name(self) :
        for breedname, breedid in self._dict_breedname_id.items():
            dirbreed = str(breedid)+'-'+str(breedname)
            print("{0:10} ..... : {1:30}".format(breedname,dirbreed))
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def kp_filter(self):
        '''Process filtering of KP descriptors issued from build_sif_desc() 
        method.
        Splitted image are filtered based of occurency of KP.
        For each splitted image, distribution of KP is computed.
        Images from which KP occurency is outside the quartile [q1,q3] are 
        ignored.
        '''
        #-----------------------------------------------------------------------
        # If splitted flag is not activated, then KP filtering is deactivated
        #-----------------------------------------------------------------------
        if self._is_splitted is False :
            self._is_kp_filtered = False
        
        if self._is_kp_filtered is False :
            return
        
        df_pil_image_kpdesc = self.df_pil_image_kpdesc

        #-----------------------------------------------------------------------
        # List of unique images identifiers is extracted.
        # All splitted images have the same marker, image_id. Thsi marker is  
        # the identifier of the original image (the one before splitting)
        #-----------------------------------------------------------------------
        list_name_id_unique = list(df_pil_image_kpdesc.image_id.unique())

        #-----------------------------------------------------------------------
        # Rows with KP array values fixed to None are erased
        #-----------------------------------------------------------------------
        print("*** Before filtering       : "+str(df_pil_image_kpdesc.shape))
        df_pil_image_kpdesc['desc'].replace([None],'-', inplace=True)
        df_pil_image_kpdesc = df_pil_image_kpdesc[df_pil_image_kpdesc['desc']!='-']
        print("*** After 1st filter level : "+str(df_pil_image_kpdesc.shape))

        df_pil_image_kpdesc_filtered = pd.DataFrame()

        for name_id_unique in list_name_id_unique :
            #-------------------------------------------------------------------
            # Select a set of splitted images from dataframe 
            #-------------------------------------------------------------------
            df = df_pil_image_kpdesc[df_pil_image_kpdesc.image_id==name_id_unique]
        
            #-------------------------------------------------------------------
            # Build dictionary with occurencies of KP for each splitted image.
            #-------------------------------------------------------------------
            dict_kp_occurency = dict()
            range_list = range(0,df.shape[0])
            i_raw = 0
            for (raw, col), list_kp in df.kp.items() :
                dict_kp_occurency[i_raw] = len(list_kp)
                i_raw += 1
        
        
        
            #-------------------------------------------------------------------
            # Occurencies dictionary is converted into a Dataframe 
            # allowing easiest operations.
            #-------------------------------------------------------------------
            
            ser = pd.Series(dict_kp_occurency)
            df_kp = pd.DataFrame([ser]).T.rename(columns={0:'count'})

            #-------------------------------------------------------------------
            # Threashold are computed 
            #-------------------------------------------------------------------
            q1,q3,zmin,zmax = p3_util.df_boxplot_limits(df_kp , 'count')

            #-------------------------------------------------------------------
            # Filtering is applied
            #-------------------------------------------------------------------
            if True :
                df_kp_filtered = df_kp[df_kp['count']<=q3]
                df_kp_filtered = df_kp_filtered[df_kp_filtered['count']>=q1]
                df.reset_index(inplace=True)
            else :
                pass

            #-------------------------------------------------------------------
            # Dataframe rows from outside filter are droped : index list is 
            # firstly built. Then this filtered indexes are applied to dataframe.
            #-------------------------------------------------------------------
            list_index_drop = list()
            df_to_filter = df.copy()
            for id  in df.index:
                if id not in  df_kp_filtered.index :
                    list_index_drop.append(id)
                else:
                    pass

            df.drop(list_index_drop, inplace=True)
        
            #-------------------------------------------------------------------
            # Filtered dataset is concatened with previous one.
            #-------------------------------------------------------------------
            df_pil_image_kpdesc_filtered \
            = pd.concat([df_pil_image_kpdesc_filtered, df],ignore_index=True)
            
        #-------------------------------------------------------------------
        # Dataframe issue from filtering replace previous dataframe. 
        #-------------------------------------------------------------------
        self._df_pil_image_kpdesc = df_pil_image_kpdesc_filtered.copy()    
        print("*** After KP filter        : "\
        +str(self._df_pil_image_kpdesc.shape))
    #---------------------------------------------------------------------------
    
    
#-------------------------------------------------------------------------------

