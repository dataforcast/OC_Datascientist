import cv2
import os

import pandas as pd
import numpy as np
import random


from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
import seaborn as sns

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import p3_util_plot
import p3_util
import p5_util
import p7_util

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_get_name_from_function(_function) :
    splitted_information = str(_function).split(' ')
    return splitted_information[1]
#-------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def p7_plot_cnn_history(cnn_model ,X_test, y_test, history=None) :
    '''Display loss and accuracy curves along with steps history.
    '''
    [test_loss, test_acc] = cnn_model.evaluate(X_test, y_test)
    print("Evaluation result on Test Data : Loss = {0:1.2F}, accuracy = {1:1.2F}"\
    .format(test_loss, test_acc))

    if history is not None :
        #Plot the Loss Curves
        plt.figure(figsize=[8,6])
        plt.plot(history.history['loss'],'r',linewidth=1.0)
        plt.plot(history.history['val_loss'],'b',linewidth=1.0)
        plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.title('Loss Curves ',fontsize=16)

        #Plot the Accuracy Curves
        plt.figure(figsize=[8,6])
        plt.plot(history.history['acc'],'r',linewidth=1.0)
        plt.plot(history.history['val_acc'],'b',linewidth=1.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Accuracy',fontsize=16)
        plt.title('Accuracy Curves',fontsize=16)
    return test_loss, test_acc
    
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def p7_filter_convolutional(image, kernel, size=(3,3), title=str()\
, xlabel=str(), ylabel=str(), is_show=False, extension='conv'):
    filename = str()
    #---------------------------------------------------------------------------
    # Les filtres par convolution ne supportent que les formats RGB et L 
    # d'encodage des pixels. L'image est réencodée en Ldict_split_pil_image = dict()
    #---------------------------------------------------------------------------
    image_L =Image.fromarray(np.array(image)).convert('L')

    #---------------------------------------------------------------------------
    # Construction du filtre avec le notau pré-définie
    #---------------------------------------------------------------------------
    image_filtered = ImageFilter.Kernel(size, kernel.flatten(), scale=None\
    , offset=0)


    #---------------------------------------------------------------------------
    # Filtrage appliqué a l'image
    #---------------------------------------------------------------------------
    image_filtered = image_L.filter(image_filtered)


    if is_show is True :
        #---------------------------------------------------------------------------
        # L'histograme des pixels et des pixels cumulés est affiché
        #---------------------------------------------------------------------------
        p7_image_hist(image_filtered, title=title, xlabel=xlabel, ylabel=ylabel)
        
        p7_image_hist(image_filtered\
                   , title=title+' cumulé'\
                   , xlabel=xlabel\
                   , ylabel=ylabel+" cumulés"\
                  ,cumulative=True)
        #---------------------------------------------------------------------------
        # Sauvegarde de l'image filtree dans un fichier
        #---------------------------------------------------------------------------
        filename = "./data/image_filtered_conv_"+extension+".png"
        image_filtered.save(filename)
        image_filtered.show()
    
    return filename, image_filtered
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def p7_filter_median(image, size=3, title=str(), xlabel=str(), ylabel=str()\
,is_show=False) :
    image_filtered = image.filter(ImageFilter.MedianFilter(size=size))
    filename = str()
    if is_show is True :
        p7_image_hist(image_filtered, title=title, xlabel=xlabel, ylabel=ylabel)
        p7_image_hist(image_filtered\
                   , title=title\
                   , xlabel=xlabel\
                   , ylabel=ylabel\
                  ,cumulative=True)
        filename = "./data/image_filtered_median_"+str(size)+".png"
        image_filtered.save(filename)    
        image_filtered.show()
    return filename, image_filtered
#------------------------------------------------------------------------------
    

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def p7_filter_gaussian(image, size=3, title=str(), xlabel=str(), ylabel=str()\
, is_show=False) :
    filename = str()
    image_filtered = image.filter(ImageFilter.GaussianBlur(size))

    if is_show is True:
        p7_image_hist(image_filtered, title=title, xlabel=xlabel, ylabel=ylabel)

        p7_image_hist(image_filtered, title=title, xlabel=xlabel, ylabel=ylabel\
                  ,cumulative=True)
        filename = "./data/image_filtered_gaussian_"+str(size)+".png"
        image_filtered.save(filename)
        image_filtered.show()
    return filename, image_filtered
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_image_hist(image, title=None, xlabel=None, ylabel=None, cumulative=False):
    arr_img = np.array(image)

    n, bins, patches = plt.hist(arr_img.flatten(), bins=range(256)\
    , cumulative=cumulative)
    if title is not None:
        plt.title(title)
    if xlabel is not None :
        plt.xlabel(xlabel)
    if ylabel is not None :
        plt.ylabel(ylabel)

    plt.show()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_pil_image_load(filename, is_verbose=True, std_size=None) :
    '''Load an image from a file using PIL package and returns it.
    '''
    pil_image_copy = None
    pil_image = None
    try :
        pil_image = Image.open(filename) 
    except FileNotFoundError :
        print("*** ERROR : image from path= "+filename+" NOT FOUND!")
        pass   
    
    if pil_image is None :
        return pil_image

    pil_image.load_end()
    try :
        pil_image_copy = pil_image.copy()
        pil_image.close()
    except AttributeError :
        pass
    
    if is_verbose is True:
        print("Format des pixels : {}".format(pil_image_copy.mode))
    if std_size is not None :
        pil_image_copy = pil_image_copy.resize(std_size)

    return pil_image_copy
#-------------------------------------------------------------------------------
 
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_image_load(filename, is_verbose=True) :
    '''Load an image from a file and returns it.
    '''
    with open(filename,"rb") as imageFile:
        image = imageFile.read() 
    return image
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def filter_convolutional(image, kernel, size=(3,3), title=str()\
, xlabel=str(), ylabel=str(), is_show=False, extension='conv'):
    #---------------------------------------------------------------------------
    # Les filtres par convolution ne supportent que les formats RGB et L 
    # d'encodage des pixels. L'image est réencodée en L
    #---------------------------------------------------------------------------
    image_L =Image.fromarray(np.array(image)).convert('L')

    #---------------------------------------------------------------------------
    # Construction du filtre avec le notau pré-définie
    #---------------------------------------------------------------------------
    image_filtered = ImageFilter.Kernel(size, kernel.flatten(), scale=None, offset=0)


    #---------------------------------------------------------------------------
    # Filtrage appliqué a l'image
    #---------------------------------------------------------------------------
    image_filtered = image_L.filter(image_filtered)

    #---------------------------------------------------------------------------
    # L'histograme des pixels et des pixels cumulés est affiché
    #---------------------------------------------------------------------------
    image_hist(image_filtered, title=title, xlabel=xlabel, ylabel=ylabel)
    
    image_hist(image_filtered\
               , title=title+' cumulé'\
               , xlabel=xlabel\
               , ylabel=ylabel+" cumulés"\
              ,cumulative=True)
    #---------------------------------------------------------------------------
    # Sauvegarde de l'image filtree dans un fichier
    #---------------------------------------------------------------------------
    filename = "./data/image_filtered_conv_"+extension+".png"
    image_filtered.save(filename)

    #---------------------------------------------------------------------------
    # L'histograme des pixels et des pixels cumulés est affiché
    #---------------------------------------------------------------------------
    if is_show is True :
        image_filtered.show()
    
    return filename, image_filtered
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def image_hist(image, title=None, xlabel=None, ylabel=None, cumulative=False):
    arr_img = np.array(image)

    n, bins, patches = plt.hist(arr_img.flatten(), bins=range(256)\
    , cumulative=cumulative)
    if title is not None:
        plt.title(title)
    if xlabel is not None :
        plt.xlabel(xlabel)
    if ylabel is not None :
        plt.ylabel(ylabel)

    plt.show()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    
    # kp are the keypoints
    
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_show_sift_features(gray_img, color_img, kp):
    
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_image_pil_show(dict_image_pil, std_image_size=(200,200),size_x=10, is_title=True) :
    '''Plot images in the dictionary given as parameter.
    Input :
        * dict_image_pil : dicitonay structures as following : {tuple_name:list_PIL_image}
        * std_image_size : ( weight, height) values for resizing image before plot.
        * size_x :
        * is_title : title to be displayed along with plotted image.
    '''
    
    for tuple_breed in  dict_image_pil.keys():
        list_image_pil = dict_image_pil[tuple_breed]

        image_count = len(list_image_pil)

        size_y = int(size_x/image_count)
        size_y = size_x
        f, axs = plt.subplots(1, image_count, figsize=(size_x,size_y))

        if( 1 < len(list_image_pil)) :
            for index in range(0,len(list_image_pil)) :
                image_pil = list_image_pil[index].copy()
                axs[index].axis('off')
                if std_image_size is not None :
                    axs[index].imshow(image_pil.resize(std_image_size))
                else :
                    axs[index].imshow(image_pil)
                if is_title is True :
                    breed = tuple_breed[index]
                    axs[index].set_title(breed)
        else :
            for index in range(0,len(list_image_pil)) :
                image_pil = list_image_pil[index].copy()
                axs.axis('off')
                if std_image_size is not None :
                    axs.imshow(image_pil.resize(std_image_size))
                else :
                    axs.imshow(image_pil)
                if is_title is True :
                    breed = tuple_breed[index]
                    axs.set_title(breed)
    #plt.tight_layout(pad=-2)
    plt.show()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_image_pil_show_deprecated(dict_image_pil\
, std_image_size=(200,200),size_x=10, is_title=True) :
    '''Plot images in the dictionary given as parameter.
    Input :
        * dict_image_pil : dicitonay structures as following : {list_name:list_PIL_image}
        * std_image_size : ( weight, height) values for resizing image before plot.
        * size_x :
        * is_title : title to be displayed along with plotted image.
    '''
    
    for breed in  dict_image_pil.keys():
        list_image_pil = dict_image_pil[breed]
        image_count = len(list_image_pil)

        size_y = int(size_x/image_count)
        size_y = size_x
        f, axs = plt.subplots(1, image_count, figsize=(size_x,size_y))

        if( 1 < len(list_image_pil)) :
            for index in range(0,len(list_image_pil)) :
                image_pil = list_image_pil[index].copy()
                axs[index].axis('off')
                if std_image_size is not None :
                    axs[index].imshow(image_pil.resize(std_image_size))
                else :
                    axs[index].imshow(image_pil)
                if is_title is True :
                    axs[index].set_title(breed)
        else :
            for index in range(0,len(list_image_pil)) :
                image_pil = list_image_pil[index].copy()
                axs.axis('off')
                if std_image_size is not None :
                    axs.imshow(image_pil.resize(std_image_size))
                else :
                    axs.imshow(image_pil)
                if is_title is True :
                    axs.set_title(breed)
    #plt.tight_layout(pad=-2)
    plt.show()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_dict_image_pil_resize(dict_img_pil, resize):
    dict_img_pil_tmp = dict() 
    for breed in dict_img_pil.keys():
        list_img_pil = [img_pil.resize(resize) for img_pil \
        in dict_img_pil[breed]]
        
        dict_img_pil_tmp[breed] = list_img_pil
    return dict_img_pil
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_get_image_name(filename) :
    list_split = filename.split('/')
    pos = len(list_split)
    return list_split[pos-1]
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_get_std_size(dict_breed_filename):
    image_count=0
    df= pd.DataFrame()
    for breed_ref_name in dict_breed_filename.keys():
        list_breed_filename = dict_breed_filename[breed_ref_name]
        for breed_filename in list_breed_filename :
            image = p7_pil_image_load(breed_filename\
            ,is_verbose=False, std_size=None)
            
            df[image_count] = (image.size[0],image.size[1]) 
            image_count +=1
    size_x = int(df.iloc[0,:].median())
    size_y = int(df.iloc[1,:].median())
    return(size_x,size_y), df
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_load_breed_name(directory_name) :
    #---------------------------------------------------------------------------
    # List of all directories, each directory contains a list of all 
    # images from breed.
    #---------------------------------------------------------------------------
    list_dir_breed = os.listdir(directory_name)
        
    #---------------------------------------------------------------------------
    # For each breed directory, list of all images files is loaded into a 
    # dictionary
    #---------------------------------------------------------------------------
    dict_breed_list_filename = dict()
    for dir_breed in list_dir_breed :
        dict_breed_list_filename[dir_breed] \
        = os.listdir(directory_name+'/'+dir_breed)
    return dict_breed_list_filename
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_load_dict_filename(root_directory_path, list_dirbreed=None) :
    '''Load all images file names contained in directories. 
    These directories are under path of a root directory.
    
    Input :
        * root_directory_path : directory path under which all sub-directories 
        containing files of images lies.
        
        * list_dirbreed : list of directories names containing images files.
        When None, then all directories under root directory are scaned in order 
        to load images files names from them.

    Output : 
        * dictionary structured as following : {dirbreed: list_of_file_name }
    where list_of_file_name is the list of files names under directory dirbreed.
    '''
    #---------------------------------------------------------------------------
    # List of all directories, each directory contains a list of all 
    # images from breed.
    #---------------------------------------------------------------------------
    if list_dirbreed is None :
        list_dirbreed = os.listdir(root_directory_path)
    else :
        pass

    #---------------------------------------------------------------------------
    # For each breed directory, list of all images files is loaded into a 
    # dictionary
    #---------------------------------------------------------------------------
    dict_filename = dict()
    for dirbreed in list_dirbreed :
        dict_filename[dirbreed] = os.listdir(root_directory_path+'/'+dirbreed)
    return dict_filename
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_load_dict_breed_imagename(dict_breed_filename):
    dict_breed_imagename = dict()
    for breed, list_filename in dict_breed_filename.items():
        for filename in list_filename:
            imagename = p7_get_image_name(filename)
            dict_breed_imagename[imagename]=breed
    return dict_breed_imagename
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_load_dict_image(image_directory, dict_breed_name, list_breed_sample\
, dog_breed_count) :
    #---------------------------------------------------------------------------
    # From any of the dogs breeds, a sample of dogs is selected.
    # Means, images from these sampling are read from files.
    #---------------------------------------------------------------------------

    dict_breed_image = dict()
    dict_breed_filename = dict()
    dict_image_pil = dict()
    list_breed_all =[ breed for breed in dict_breed_name.values()]
    #-------------------------------------------------------------------------------
    # List of name containing directories is built and duplicated names are removed
    #-------------------------------------------------------------------------------
    list_breed_sample_name =[list_breed_all[index] for index in list_breed_sample]
    list_breed_sample_name = list(set(list_breed_sample_name))
    list_breed_sample_name

    
    #-------------------------------------------------------------------------------
    # A random list of directories are selected among dogs breeds
    # list_breed_sample contains the sampled list of dogs breeds.
    #-------------------------------------------------------------------------------
    for directory, breed in dict_breed_name.items() :
        if directory.split('-')[1] in list_breed_sample_name :
            #print(directory)
            list_image = list()
            list_filename = list()
            list_image_pil = list()
            list_image_file = os.listdir(image_directory+'/'+directory)
            for index_image_file in range(0,dog_breed_count,1) :
                filename = image_directory+'/'+directory+'/'+str(list_image_file[index_image_file])
                list_filename.append(filename)
                list_image.append(p7_image_load(filename, is_verbose=False))
                list_image_pil.append(p7_pil_image_load(filename, is_verbose=False))
            dict_breed_image[breed] = list_image
            dict_breed_filename[breed] = list_filename
            dict_image_pil[breed] = list_image_pil
    print(len(dict_breed_image))
    return dict_breed_image, dict_breed_filename,dict_image_pil 
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_breed_sampling_index(dict_breed_all, breed_count, dog_breed_count):
    '''Returns a random list of indexes for breeds.
    '''
    list_breed_sample = list()
    
    if breed_count == -1 :
        for index in range(0, len(dict_breed_all)):
            list_breed_sample.append(index)
    else : 
        for sample in range(0, breed_count,1):
            list_breed_sample.append(random.randrange(0, len(dict_breed_all),breed_count))

        
    # List is rendered unique
    list_breed_sample = list(set(list_breed_sample))
    return list_breed_sample

#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_pil_to_keras_image(pil_image, is_show=True) :
    '''Convert PIL image into a Keras image.
    PIL image must be extended with an additional dimension for batch size.
    Also Keras image required PIL image to be resized in (224,224)
    '''
    pil_image =pil_image.resize((224,224)) 
    if is_show is True :
        plt.imshow(pil_image)
        plt.show()
    numpy_image = img_to_array(pil_image)
    batch_image = np.expand_dims(numpy_image, axis=0)
    if is_show is True :
        plt.imshow(np.uint8(batch_image[0]))
        plt.show()
    else : 
        pass
    return batch_image    
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_list_filtered_index(list_breed_kpdesc, is_verbose=False, style='box') :
    dict_kp_occurency = dict()
    range_list = range(0,len(list_breed_kpdesc))

    for i_raw, tuple_kp_image in zip(range_list, list_breed_kpdesc) :
        dict_kp_occurency[i_raw] = len(tuple_kp_image[0])
            
    ser = pd.Series(dict_kp_occurency)
    df_kp = pd.DataFrame([ser]).T.rename(columns={0:'count'})


    q1,q3,zmin,zmax = p3_util.df_boxplot_limits(df_kp , 'count')
    min_abs = df_kp['count'].min()
    max_abs = df_kp['count'].max()
    
    if is_verbose is True :
        print("Q1   = "+str(q1))
        print("Q3   = "+str(q3))
        print("Zmin = "+str(zmin))
        print("Zmax = "+str(zmax))
        print("Min  = "+str(min_abs))
        print("Max  = "+str(max_abs))
        kp_count = df_kp.apply(lambda x: sum(x))
        print("Number of KP= "+str(kp_count))
    
    if style=='box':    
        p3_util_plot.df_boxplot_display(df_kp, 'count')
    elif style=='violin' :
        sns.violinplot( y=None, x='count', data=df_kp, idth=0.8, inner='quartile', palette="colorblind")
    else :
        pass
    #---------------------------------------------------------------------------
    # Filtering is applied
    #---------------------------------------------------------------------------
    min_limit = q1
    #max_limit = df_kp['count'].max()
    max_limit = q3
    
    if is_verbose is True :
        print("Min limit= "+str(min_limit))
        print("Max limit= "+str(max_limit))
    
    df_kp_filtered = df_kp[df_kp['count']<=max_limit]
    df_kp_filtered = df_kp_filtered[df_kp_filtered['count']>=min_limit]
    #df_kp_filtered = df_kp_filtered[df_kp_filtered['count']>1]

    list_filtered_index = list(df_kp_filtered.index)
    return list_filtered_index
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def plot_filtered_kpdesc_image(list_breed_kpdesc, dict_breed_kpdesc_image\
, is_verbose=False, style = 'box') :
    '''Plot a splitted image that has been filtered based on KP distribution
    in each splitted image.
    
    Input :
        * list_breed_kpdesc : list of tuples; tuples are structured as following :
            (kp_array, descriptors)
        * dict_breed_kpdesc_image : dictionary 
    '''

    #---------------------------------------------------------------------------
    # Filtering is applied
    #---------------------------------------------------------------------------
    list_filtered_index = get_list_filtered_index(list_breed_kpdesc\
    , is_verbose=is_verbose, style=style)
    
    
    #---------------------------------------------------------------------------
    # For splitted images out of filtered indexes, they are turned in black.
    #---------------------------------------------------------------------------
    index=0
    #for i_raw in range(0,raw):
    print(list_filtered_index)
    #if True :
    for i_raw in range(0,len(dict_breed_kpdesc_image)) :
        col = dict_breed_kpdesc_image[i_raw].shape[0]
        
        for i_col in range(0,col):
            if index in list_filtered_index :
                pass
            else :
                # Image index out of filter is erased 
                arr_ = dict_breed_kpdesc_image[i_raw][i_col]
                dict_breed_kpdesc_image[i_raw][i_col] \
                = np.zeros((arr_.shape[0],arr_.shape[1],3))
            index += 1
    
    p7_util.p7_image_pil_show(dict_breed_kpdesc_image\
                              ,size_x=10,std_image_size=None,is_title=False)
    return dict_breed_kpdesc_image
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def plot_filtered_kpdesc_image_deprecated(list_breed_kpdesc, dict_breed_kpdesc_image) :
    '''Plot a splitted image that has been filtered based on KP distribution
    in each splitted image.
    
    Input :
        * list_breed_kpdesc : list of tuples; tuples are structured as following :
            (kp_array, descriptors)
        * dict_breed_kpdesc_image : dictionary 
    '''

    #---------------------------------------------------------------------------
    # Building KP occurancies for each splitted image 
    #---------------------------------------------------------------------------
    dict_kp_occurency = dict()
    range_list = range(0,len(list_breed_kpdesc))

    for i_raw, tuple_kp_image in zip(range_list, list_breed_kpdesc) :
        dict_kp_occurency[i_raw] = len(tuple_kp_image[0])
            
    ser = pd.Series(dict_kp_occurency)
    df_kp = pd.DataFrame([ser]).T.rename(columns={0:'count'})

    p3_util_plot.df_boxplot_display(df_kp, 'count')

    q1,q3,zmin,zmax = p3_util.df_boxplot_limits(df_kp , 'count')
    
    if True :
        print("Q1   = "+str(q1))
        print("Q3   = "+str(q3))
        print("Zmin = "+str(zmin))
        print("Zmax = "+str(zmax))
        print("Min  = "+str(df_kp['count'].min()))
        print("Max  = "+str(df_kp['count'].max()))
    
    #---------------------------------------------------------------------------
    # Filtering is applied
    #---------------------------------------------------------------------------
    min_limit = q1
    max_limit = q3
    
    if True :
        print("Min limit= "+str(min_limit))
        print("Max limit= "+str(max_limit))
    
    df_kp_filtered = df_kp[df_kp['count']<=max_limit]
    df_kp_filtered = df_kp_filtered[df_kp_filtered['count']>=min_limit]

    list_filtered_index = list(df_kp_filtered.index)
    
    index=0
    #for i_raw in range(0,raw):
    print(list_filtered_index)
    #if True :
    for i_raw in range(0,len(dict_breed_kpdesc_image)) :
        col = dict_breed_kpdesc_image[i_raw].shape[0]
        
        for i_col in range(0,col):
            if index in list_filtered_index :
                pass
            else :
                # Image index out of filter is erased 
                arr_ = dict_breed_kpdesc_image[i_raw][i_col]
                dict_breed_kpdesc_image[i_raw][i_col] \
                = np.zeros((arr_.shape[0],arr_.shape[1],3))
            index += 1
    
    p7_util.p7_image_pil_show(dict_breed_kpdesc_image\
                              ,size_x=10,std_image_size=None,is_title=False)  
    return df_kp
#-------------------------------------------------------------------------------
    
