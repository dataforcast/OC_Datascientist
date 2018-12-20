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
    #---------------------------------------------------------------------------------------------
    # Les filtres par convolution ne supportent que les formats RGB et L d'encodage des pixels.
    # L'image est réencodée en L
    #---------------------------------------------------------------------------------------------
    image_L =Image.fromarray(np.array(image)).convert('L')

    #---------------------------------------------------------------------------------------------
    # Construction du filtre avec le notau pré-définie
    #---------------------------------------------------------------------------------------------
    image_filtered = ImageFilter.Kernel(size, kernel.flatten(), scale=None, offset=0)


    #---------------------------------------------------------------------------------------------
    # Filtrage appliqué a l'image
    #---------------------------------------------------------------------------------------------
    image_filtered = image_L.filter(image_filtered)

    #---------------------------------------------------------------------------------------------
    # L'histograme des pixels et des pixels cumulés est affiché
    #---------------------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def p7_filter_median(image, size=3, title=str(), xlabel=str(), ylabel=str()\
,is_show=False) :
    image_filtered = image.filter(ImageFilter.MedianFilter(size=size))
    image_hist(image_filtered, title=title, xlabel=xlabel, ylabel=ylabel)

    image_hist(image_filtered\
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
    image_hist(image_filtered, title=title, xlabel=xlabel, ylabel=ylabel)

    image_hist(image_filtered, title=title, xlabel=xlabel, ylabel=ylabel\
              ,cumulative=True)
    filename = "./data/image_filtered_gaussian_"+str(size)+".png"
    image_filtered.save(filename)

    if is_show is True:
        image_filtered.show()
    return filename, image_filtered
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p7_image_hist(image, title=None, xlabel=None, ylabel=None, cumulative=False):
    arr_img = np.array(image)

    n, bins, patches = plt.hist(arr_img.flatten(), bins=range(256), cumulative=cumulative)
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
def p7_image_load_deprecated(filename, is_verbose=True) :
    '''Load an image from a file and returns it.
    '''
    image = Image.open(filename) 
    if is_verbose is True:
        print("Format des pixels : {}".format(image.mode))
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

    if is_verbose is True:
        print("Format des pixels : {}".format(image.mode))
    return image
#------------------------------------------------------------------------------


