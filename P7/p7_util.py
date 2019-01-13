import cv2
import os

import pandas as pd
import random

from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import p5_util

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def p7_filter_convolutional(image, kernel, size=(3,3), title=str()\
, xlabel=str(), ylabel=str(), is_show=False, extension='conv'):
    #---------------------------------------------------------------------------
    # Les filtres par convolution ne supportent que les formats RGB et L 
    # d'encodage des pixels. L'image est réencodée en L
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

    #---------------------------------------------------------------------------
    # L'histograme des pixels et des pixels cumulés est affiché
    #---------------------------------------------------------------------------
    if is_show is True :
        image_filtered.show()
    
    return filename, image_filtered
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def p7_filter_median(image, size=3, title=str(), xlabel=str(), ylabel=str()\
,is_show=False) :
    image_filtered = image.filter(ImageFilter.MedianFilter(size=size))
    p7_image_hist(image_filtered, title=title, xlabel=xlabel, ylabel=ylabel)

    p7_image_hist(image_filtered\
               , title=title\
               , xlabel=xlabel\
               , ylabel=ylabel\
              ,cumulative=True)
    filename = "./data/image_filtered_median_"+str(size)+".png"
    image_filtered.save(filename)
    
    if is_show is True :
        image_filtered.show()
    return filename, image_filtered
#------------------------------------------------------------------------------
    

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def p7_filter_gaussian(image, size=3, title=str(), xlabel=str(), ylabel=str()\
, is_show=False) :
    image_filtered = image.filter(ImageFilter.GaussianBlur(size))
    p7_image_hist(image_filtered, title=title, xlabel=xlabel, ylabel=ylabel)

    p7_image_hist(image_filtered, title=title, xlabel=xlabel, ylabel=ylabel\
              ,cumulative=True)
    filename = "./data/image_filtered_gaussian_"+str(size)+".png"
    image_filtered.save(filename)

    if is_show is True:
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
def p7_pil_image_load(filename, is_verbose=True, std_size=(200,200)) :
    '''Load an image from a file using PIL package and returns it.
    '''
    image = Image.open(filename) 
    image.load_end()
    if is_verbose is True:
        print("Format des pixels : {}".format(image.mode))
    if std_size is not None :
        image = image.resize(std_size)
    return image
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
def p7_image_pil_show(dict_image_pil, std_image_size=(200,200),size_x=10) :
    
    for breed in  dict_image_pil.keys():
        list_image_pil = dict_image_pil[breed]
        image_count = len(list_image_pil)
        size_y = int(size_x/image_count)
        f, axs = plt.subplots(1, image_count, figsize=(size_x,size_y))

        if( 1 < len(list_image_pil)) :
            for index in range(0,len(list_image_pil)) :
                image_pil = list_image_pil[index].copy()
                axs[index].axis('off')
                if std_image_size is not None :
                    axs[index].imshow(image_pil.resize(std_image_size))
                else :
                    axs[index].imshow(image_pil)
                axs[index].set_title(breed)
        else :
            for index in range(0,len(list_image_pil)) :
                image_pil = list_image_pil[index].copy()
                axs.axis('off')
                if std_image_size is not None :
                    axs.imshow(image_pil.resize(std_image_size))
                else :
                    axs.imshow(image_pil)
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
def p7_load_data(directory_name, dirbreed=None) :
    #---------------------------------------------------------------------------
    # List of all directories, each directory contains a list of all 
    # images from breed.
    #---------------------------------------------------------------------------
    list_dir_breed = None
    list_dir_breed_image = None        
    if dirbreed is None :
        list_dir_breed = os.listdir(directory_name)
    else :
        list_dir_breed_image = os.listdir(directory_name+'/'+dirbreed)    

    #---------------------------------------------------------------------------
    # For each breed directory, list of all images files is loaded into a 
    # dictionary
    #---------------------------------------------------------------------------
    dict_breed_list_filename = dict()
    if list_dir_breed is not None :
        for dirbreed in list_dir_breed :
            dict_breed_list_filename[dirbreed] \
            = os.listdir(directory_name+'/'+dirbreed)
    else : 
        list_dirbreed = os.listdir(directory_name+'/'+dirbreed)
        dict_breed_list_filename={dirbreed : [ filename for filename in list_dirbreed ]}
    return dict_breed_list_filename
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


    
