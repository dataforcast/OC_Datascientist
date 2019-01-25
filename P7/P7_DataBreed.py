import os
import random

import pandas as pd
import numpy as np

import cv2
from PIL import ImageOps
from PIL import Image
from  sklearn import model_selection

import p5_util
import p6_util
import p7_util
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import P7_DataBreed
import p5_util


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
def p7_keras_X_y_build(ser_pil_image, ser_label, square=None):
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
    # Initialization : convert all PIL images into Keras array of images.
    #---------------------------------------------------------------------------
    image_id = 0
    list_image_error = list()
    list_label_error = list()
    
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

    
        pil_image_truncated = pil_square(pil_image, square=square)
        arr_keras = np.array(pil_image_truncated)
        
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
, test_size=0.2, square=None):
    '''Build train and test arrays based on given data arrays.
    Input : 
        * ser_pil_image : Series containing PIL images.
        * ser_label : Series of labels related to PIL images.
        * test_size : percentage of test data-set.
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
    print("Images for training = "+str(len(ser_pil_image)))
    arr_keras_image, arr_label \
    = p7_keras_X_y_build(ser_pil_image, ser_label, square=square)
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
def p7_keras_X_train_test_build_deprected(ser_pil_image, ser_label\
, test_size=0.2, square=None):
    '''Build train and test arrays based on given data arrays.
    Input : 
        * ser_pil_image : Series containing PIL images.
        * ser_label : Series of labels related to PIL images.
        * test_size : percentage of test data-set.
    Output :
        * arr_X_train : array of pixels containing train dataset
        * arr_X_test : array of pixels containing test dataset
        * arr_y_train : array of labels related to images from train dataset.
        * arr_y_test : array of labels related to images from test dataset.
        
    '''
    #---------------------------------------------------------------------------
    # Get train and test dataset returned as Series of PIL images
    #---------------------------------------------------------------------------
    ser_pil_image_train, ser_pil_image_test, ser_label_train,  ser_label_test \
    = model_selection.train_test_split(ser_pil_image, ser_label\
    , test_size=test_size)
    
    print(ser_pil_image_train.shape,ser_pil_image_test.shape\
    ,ser_label_train.shape, ser_label_test.shape)
    
    #---------------------------------------------------------------------------
    # Convert train dataset Series into numpy array allowing to feed Keras 
    # dense layer.
    #---------------------------------------------------------------------------
    print("Images for training = "+str(len(ser_label_train)))
    arr_X_train, arr_y_train \
    = p7_keras_X_y_build(ser_pil_image_train, ser_label_train, square=square)
    print(arr_X_train.shape, arr_y_train.shape)
    
    #---------------------------------------------------------------------------
    # Convert test dataset Series into numpy array 
    #---------------------------------------------------------------------------
    print()
    print("Images for testing = "+str(len(ser_pil_image_test)))
    arr_X_test, arr_y_test \
    = p7_keras_X_y_build(ser_pil_image_test, ser_label_test,square=square)
    print(arr_X_test.shape, arr_y_test.shape)

    return arr_X_train, arr_X_test, arr_y_train, arr_y_test
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def pil_square(pil_image,square=None):
    '''Truncate image with a margin for having same size for height aimagend weight.
    Input :
        * Rectangular PIL image 
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
def pil_truncate(pil_image,std_size) :
    ''' 
        Input :
            * pil_image : PIL image format to be truncated 
            * std_size : tuple for size truncation
        Output : 
            * truncated PIL image.
    '''
    arr_image = np.array(pil_image)
    #print(std_size[0],std_size[1])
    #print(arr_image.shape)
    #print(arr_image.shape[0],arr_image.shape[1])
    
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
def show_pil_image_and_kp(pil_image,breedname) :
    kp, desc = get_image_kpdesc(pil_image)
    print("KP= "+str(len(kp)))
    print("DESC= "+str(desc.shape))
    dict_breed_kpdesc = {breedname:[(kp,desc)]}
    dict_pil_image = {breedname : [pil_image] }
    dict_breed_kpdesc_image = dict()
    for (breed, list_breed_kpdesc), list_image_pil in zip(dict_breed_kpdesc.items(), dict_pil_image.values()):
        dict_breed_kpdesc_image[breed] = [cv2.drawKeypoints(np.array(image_pil), kp, np.array(image_pil)) \
                                 for ((kp, desc),image_pil) in zip(list_breed_kpdesc,list_image_pil)]
    p7_util.p7_image_pil_show(dict_breed_kpdesc_image,std_image_size=None)

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
    return dirbreed.split('-')[1]
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
def p7_read_image(imagepathname, image_type='PIL') :
    
    pil_image = p7_util.p7_pil_image_load(imagepathname\
            , is_verbose=False, std_size=None)
    
    return pil_image
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
            |   +--> p7_util.p7_load_data()
            |   
            +-->sampling()
            |   |
            |   +--> build_ser_number_breedname()
            |
            +-->build_sift_desc()
            |   |
            |   +--> p7_util.p7_pil_image_load()
            |   |
            |   +--> kpdesc_build()
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
        self.strprint("Number of breeds .............. : "\
        +str(len(self._dict_data)))
        self.strprint("Total number of images ........ : "\
        +str(self._total_image))       
        self.strprint("Standard images size .......... : "\
        +str(self._std_size))
        self.strprint("SIFT Image descriptors count .. : "\
        +str(len(self._dict_breed_kpdesc)))


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
        self.strprint("Labels from dataset ........... : "\
        +str(self._y_label.shape))
        self.strprint("Number of breeds .............. : "\
        +str(len(self._dict_breedname_id)))
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
        #self.strprint("Number of splitted images ..... : "\
        #+str(len(self.dict_split_pil_image)))
        self.strprint("Splitted parts ................ : "\
        +str(self._split_ratio))
        self.strprint("Dataframe images descriptors .. : "\
        +str(self._df_pil_image_kpdesc.shape))
        
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
        print("\n*** WARN : cluster model does not exists in clusters dictionary !\n")

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
        #self._dict_split_pil_image = dict_split_pil_image.copy()
        print("\n*** WARN : assignement is not authorized !\n")

    def _get_split_ratio(self) :
      return self._split_ratio
    def _set_split_ratio(self, split_ratio) :
        self._split_ratio = split_ratio
    
    def _get_df_pil_image_kpdesc(self) :
      return self._df_pil_image_kpdesc
    def _set_df_pil_image_kpdesc(self, Xdesc) :
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
    


    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def df_build(self) :
        '''Build dataframe from _dict_data. This dictionary is structured as 
        following : {dirbreed:list_of_image_name}
        where  : 
            * dirbreed is the directory name for a breed
            * list_of_image_name is the list of all image files under dirbreed

        and built dataframe is structured as following : 
            * columns : breed, label, image
                --> breed : human readable breed name
                --> label : encoded label for breed name.
                --> image : content of image in a given format.
        '''
        dict_image = dict()
        label = 0
        df = pd.DataFrame()
        new_breedname = str()
        for dirbreed, list_imagefilename in self._dict_data.items():
            for imagefilename in list_imagefilename :
                pil_image = self.read_image(dirbreed, imagefilename)
                breedname = dirbreed.split('-')[1]
                df = df.append(pd.DataFrame([[breedname, label,pil_image]]\
                , columns=['breed','label','image']))

            label +=1
        
        df.reset_index(drop=True, inplace=True)
        print("Image count = "+str(len(df)))
        return df
    #---------------------------------------------------------------------------

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
        id = self._dict_breedname_id[breedname]
        breedname = id+'-'+breedname
        image_filename = self._dir_path+'/'+breedname+'/'+imagename
        return image_filename
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def load(self, dirbreed=None, imagename=None) :
        '''Read all images from data directory path .
        Imges are stored into a dictionary with such structure : 
        {breed_directory : list_of_breed_file_names} 
        where list_of_breed_file_names is the list of file names that 
        reference images from breed.
        '''
        self._dict_data = p7_util.p7_load_data(self._dir_path, dirbreed=dirbreed)
        if imagename is not None :
            self._dict_data = {dirbreed:[imagename]}
        
        #-----------------------------------------------------------------------
        # Total number of files
        #-----------------------------------------------------------------------
        self._total_images = 0
        for breed in self._dict_data.keys():
            self._total_image += len(self._dict_data[breed])              
            
        if False :
            #-----------------------------------------------------------------------
            # Each image is read from data directory and resized.        
            #-----------------------------------------------------------------------
            self._dict_img_pil = dict() 
            for dirbreedname, list_filename in self._dict_data.items():
                list_image_pil = list()
                for filename in list_filename :
                    #---------------------------------------------------------------
                    # Path file for image access is built
                    #---------------------------------------------------------------
                    pathfilename = self._build_pathname(dirbreedname, filename)
                    image_pil = p7_util.p7_pil_image_load(pathfilename\
                    ,is_verbose=False, std_size=None)

                    #---------------------------------------------------------------
                    #Image is resized and stored in list of images for this breed
                    #---------------------------------------------------------------
                    #list_image_pil.append(image_pil.resize(self._std_size))
                    list_image_pil.append(image_pil)

                #-------------------------------------------------------------------
                # List of resized images is stored into dictionary
                #-------------------------------------------------------------------
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
        '''Build path name from path dir, and given parameters that are :
        Input : 
            * dir_breed : directory breed name 
            * filenamebreed : a file name located into dir_breed
        Output :
            * file path name.
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
        '''Split a PIL formated image given a ratio for weight and given a ratio 
        for height.
        
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
        if self._std_size is None :
            width  = int(pil_image.size[0]/ratio[0])
            height = int(pil_image.size[1]/ratio[1])
        else:        
            width  = int(self._std_size[0]/ratio[0])
            height = int(self._std_size[1]/ratio[1])
        dict_pil_image = dict()
        imgwidth, imgheight = pil_image.size
        for i in range(0,imgheight,height):
            list_pil_image_crop = list()
            for j in range(0,imgwidth,width):
                box = (j, i, j+width, i+height)
                list_pil_image_crop.append(pil_image.crop(box))
            labelname_i = str(i)+'_'+labelname
            dict_pil_image[labelname_i]=  list_pil_image_crop  
        
        return dict_pil_image
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def build_arr_desc(self):
        '''Build an array (Nx128) where :
        --> N : is the total number of keypoints for the dataset.
        --> 128 is the number of descriptors per keypoints.
        Arrays of keypoints descriptors are stored into a dictionary.
        Building array of descriptors leads to stack eacu one of these arrays.
        
        TBD : to store all descriptors into a dataframe for which : 
        indexes of raws allow to identify image ==> 2 levels index dataframe.
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
        raws = len(self._dict_breed_kpdesc)
        if(0 >= raws):
            print("*** ERROR : empty Key-points descriptors! \
            build it with build_sift_desc()")
            return

        for id, (desc,breedname) in self._dict_breed_kpdesc.items():
            try :
                X_desc = np.vstack((X_desc,desc))
            except ValueError:
                error +=1
                pass
            count+=1
            if count%1000==0 :
                print("Processed raws= "+str(count)+"/"+str(raws))
        if 0 < error :
            print("\n*** WARN : Nb of exceptions during process ... : "\
            +str(error))
        self._Xdesc = X_desc[1:].copy()
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _df_pil_image_kpdesc_build(self, array_values):
        '''Reset dataframe holding PIL images along with corresponding 
        KP and descriptors.
        Dataframe is built with a multi-level indexes. 
            --> 1st level index references raws of spillted images.
            --> 2nd index references columns for splitted images in a raw. 
        Number of raws and columns are issued from split image process. This 
        process is leaded by parameter self.split_ratio.
        
        Input :
            * array_values : array of values : KP, descriptors, image size, 
            breed name.
        Output :
            * dataframe with multi-level indexes and values contained in arr.
        '''
        raw = self.split_ratio[0]
        col = self.split_ratio[1]
        
        col = int(np.sqrt(array_values.shape[0]))
        raw = col 
        
        raw_index =np.arange(0,raw*col,1)
        col_index =np.arange(0,raw*col,1)

        #-----------------------------------------------------------------------
        # Index for raws and colulns initialization
        #-----------------------------------------------------------------------
        raw_index[:]=0
        col_index[:]=0

        for i in range(0,raw*col, col):
            raw_index[i:i+col] = int(i/col)

        for i in range(0,raw*col, col):
            col_index[i:i+col] = range(0,col,1)

        list_level_index=[raw_index,col_index]
                
        
        df_multi_level \
        = pd.DataFrame(array_values\
        , columns=['desc','breed','kp','size','split_image','image_id']\
        , index=list_level_index)

        return df_multi_level
    #---------------------------------------------------------------------------    

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def kpdesc_build(self, dirbreed, pil_image, image_count,imagename) :
        '''Build matrix of key points descriptors where  :
        --> number of raws from this matrix is the number of keypoints
        --> number of columns is the number of features (128) for each keypoint.

        Build is processed following is_splitted falg. When activated, then 
        image is splitted and matrix is built over each one of the splitted 
        images.
        
        Input :
            * dirbreed : directory in which all images lays on.
            * pil_image : PIL image from which KPDESC matrix is built.
            * image_count : this is an incremental value used to build raws of 
            the KPDESC matrix.
            
        Output : 
            * dict_breed_kpdesc : KPDESC matrix stored in a dictionary 
            strustured as following : {image_count:(desc,hr_breedname)}, where :
                --> desc : this is the descriptor vector (128 sized) for image 
                identified with image_count
                --> hr_breedname : human readable name of the breed.
            * image_count : current number of images.
        
        '''

        dict_breed_kpdesc = dict()
        hr_breedname = get_breedname_from_dirbreed(dirbreed)
        name_id = imagename.split('.')[0]
        if name_id == 'n02113186_11037' :
            print(name_id)            
        if self._is_splitted is True :
            
            dict_split_pil_image = self.split_pil_image(pil_image,hr_breedname)
            
            for id_breedname, list_split_pil_image \
            in dict_split_pil_image.items() :
                for split_pil_image in list_split_pil_image :
                    kp, desc = get_image_kpdesc(split_pil_image)
                    dict_breed_kpdesc[image_count] \
                    = (desc,hr_breedname,kp,split_pil_image.size,split_pil_image,name_id)
                    image_count +=1

            self._dict_breed_kpdesc.update(dict_breed_kpdesc)
        else :            
            kp, desc = get_image_kpdesc(pil_image)
            dict_breed_kpdesc[image_count] \
            = (desc,hr_breedname, kp,split_pil_image.size,split_pil_image,name_id)
            self._dict_breed_kpdesc.update(dict_breed_kpdesc)

        #-----------------------------------------------------------------------
        # Dictionary of splitted images is updated.
        #-----------------------------------------------------------------------
        #self._dict_split_pil_image.update(dict_split_pil_image)
        #print("*** kpdesc_build() ... "+str(len(self._dict_split_pil_image)))
        
        #-----------------------------------------------------------------------
        # Dataframe with all informations related to PIL images and descriptors 
        #-----------------------------------------------------------------------
        ar = np.array(list(dict_breed_kpdesc.values()))

        #print("*** kpdesc_build() ... "+str(ar.shape))
        df_multi_level = self._df_pil_image_kpdesc_build(ar)

        #print("***"+str(df_multi_level.shape))

        self._df_pil_image_kpdesc  \
        = pd.concat( [self._df_pil_image_kpdesc, df_multi_level])

        if False :
            self._df_pil_image_kpdesc \
            = self._df_pil_image_kpdesc.append(df_multi_level)

        return dict_breed_kpdesc, image_count
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
                    #-----------------------------------------------------------
                    # Resize of PIL image
                    #-----------------------------------------------------------
                    #pil_image.resize(self._std_size)

                    #-----------------------------------------------------------
                    # Gray transformation : removing channel dimension.
                    #-----------------------------------------------------------
                    pil_image = pil_2gray(pil_image)

                    #-----------------------------------------------------------
                    # Image is rendered with same weight and heigth 
                    #-----------------------------------------------------------
                    pil_image = pil_square(pil_image)
                    
                    #-----------------------------------------------------------
                    # Image may be truncated to render std_size
                    #-----------------------------------------------------------
                    if self.std_size is not None :
                        pil_image = pil_truncate(pil_image,self.std_size)
                    else :
                        pass
                    
                    #-----------------------------------------------------------
                    # Median filter is applied
                    #-----------------------------------------------------------
                    filename, pil_image = p7_util.p7_filter_median(pil_image)

                    #-----------------------------------------------------------
                    # AUto-contrast filter is applied: this allows to provide 
                    # mode contrast for finding SIFT descriptors.
                    # It is supposed here that more an imageis contrasted, 
                    # the easiest SIFT descriptors will be extracted from 
                    # shape. 
                    #-----------------------------------------------------------
                    pil_image = pil_autocontrast(pil_image)

                    #-----------------------------------------------------------
                    # Equalization
                    #-----------------------------------------------------------
                    pil_image = pil_equalize(pil_image)
                    
                    #-----------------------------------------------------------
                    # Store descriptor along with breed name. This will be usefull
                    # for classification.
                    #-----------------------------------------------------------
                    dict_breed_kpdesc, image_count\
                    = self.kpdesc_build(dirbreed, pil_image,image_count,imagename)
                    
                    #print("Images processed= "+str(image_count))             

                    #self._dict_breed_kpdesc = dict_breed_kpdesc.copy()
                    
                    #-----------------------------------------------------------
                    # Closing PIL image : all resources of PIL image are released.
                    #-----------------------------------------------------------
                    pil_image.close()

                    #-----------------------------------------------------------
                    # Display progress
                    #-----------------------------------------------------------
                    if False :
                        if(0 == (image_count)%500 ) :
                            print("Images processed= "\
                            +str(image_count)+"/"+str(self._total_image))

                    if self._is_splitted is False :
                        image_count +=1     
                    
                except AttributeError :
                    error +=1
                    #print("*** WARNING : attribute error for PIL image ")
                    continue                
        self._df_pil_image_kpdesc.index.names =['raw','col']
        print("\nINFO : Error = "+str(error)+" Total images processed= "+str(image_count))        
    #---------------------------------------------------------------------------
        
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def sampling(self, breed_count, image_per_breed_count):
        #-----------------------------------------------------------------------
        # Select randomly a breed directory name
        #-----------------------------------------------------------------------
        self._list_breed_sample = list()
        for breed_id in range(0, breed_count,1):
            choice = random.choice(list(self._dict_data.keys()))
            self._list_breed_sample.append(choice)
            
        #-----------------------------------------------------------------------
        # For each selected breed, a random list of images is selected
        #-----------------------------------------------------------------------
        count=0           
        self._dict_breed_sample = dict()
        for breedname in self._list_breed_sample :
            list_filename = self._dict_data[breedname]
            list_file_sample = list()
            for file_id in range(0, image_per_breed_count,1):
                list_file_sample.append(random.choice(list_filename))
                count +=1
            self._dict_breed_sample[breedname] = list_file_sample
        self._sampling_breed_count = breed_count
        self._sampling_image_per_breed_count = image_per_breed_count
        self.build_ser_number_breedname()
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def read_image(self, dirbreed, imagename, image_type='PIL') :
    
        imagepathname = self._build_pathname(dirbreed, imagename)
        pil_image = p7_read_image(imagepathname)
        
        return pil_image
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
        and convert if into a dataframe.
        
        Input : 
            * desc : image represented as SIFT key points descriptors.
            Raws are keypoints, extended over 128 descriptors (columns)
        Output :
            * dataframe containing histogram of clusters representing 
            image bag of visual words.
            dataframe raws : images identifiers
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
            # Initialization
            #-------------------------------------------------------------------
            dict_feature = dict()
            for i in range(0,nb_cluster) :
                dict_feature[i]=0     
            #-------------------------------------------------------------------
            # Get cluster from image represented as Key points descriptors
            #-------------------------------------------------------------------
            try :
                y_label = cluster_model.predict(desc)
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
        '''Build representation of keypoints dataset in a bag of features.
        Result is normalized and stored into a pandas data-frame.
        Clustering should have been built before this step.
        Output :
            * dataframe structures as following : 
                Raws index are references to images from sampling 
                Columns are features occurencies.
        '''
        df = None
        dict_label = dict()
        error =0
        list_image_id_error = list()
        for image_id, (imagedesc, breedname) in self._dict_breed_kpdesc.items():
            df_tmp = self.get_cluster_from_imagedesc(imagedesc)
            if df_tmp is None :
                error +=1
                list_image_id_error.append(image_id)
            else :            
                # Index is matched with image id
                df_tmp.rename(index={0:image_id}, inplace=True)
                if df is None :
                    df = df_tmp.copy()
                else:
                    df = pd.concat([df, df_tmp])
                # Used for Y label
                breedlabel = self.get_breedlabel_from_breedname(breedname)
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
        # Label are encoded in a multi-class way
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
        for name,id in self._dict_breedname_id.items() :
            print("Identifier= {}  Breed name= {}".format(str(id)+'-'+name,name))
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def build_ser_number_breedname(self) :
        '''Build a Series with index as breed numbers and values as human 
        readable breed names.
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
        # Also build dictionary for label identifiers
        #-----------------------------------------------------------------------
        index = 0
        dict_breed_number = dict()
        for breed_name in self._dict_breedname_id.keys() :
            dict_breed_number[index] = breed_name
            index+=1
            #print(dict_breed_number)
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
            * is_sample_show : when fixed to True, then....?
        '''

        ser = pd.Series(self._dict_breedname_id)
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
        self.load(dirbreed=dirbreed, imagename=imagename)
        
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
            print("{0} ..... : {1}".format(breedname,dirbreed))
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    #def single_image(self,list_restricted_image):
    #    self.list_restricted_image = list_restricted_image.copy()
    #---------------------------------------------------------------------------
    
    
    
#-------------------------------------------------------------------------------

