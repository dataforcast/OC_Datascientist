import pandas as pd
from PIL import ImageOps
import numpy as np
import random

import p7_util

#---------------------------------------------------------------------------
#
#---------------------------------------------------------------------------
def get_breedname_from_dirbreed(dirbreed):
    return dirbreed.split('-')[1]
#---------------------------------------------------------------------------
        

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
class P7_DataBreed() :
    '''This class implements breeds data structure.
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
        self._std_size = (0,0)
        self._dict_img_pil = dict() 
        self._dict_breed_kpdesc = dict()
        self._dict_breed_sample = dict()
        self._list_breed_sample = list()
        
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
    def show(self):
        '''Show classes attributes
        '''
        self.strprint("Path to data directory ........ : "+str(self._dir_path))
        self.strprint("Number of breeds .............. : "\
        +str(len(self._dict_data)))
        self.strprint("Total number of images ........ : "\
        +str(self._total_image))       
        self.strprint("Standard images size .......... : "\
        +str(self._std_size))       
        self.strprint("SIFT Descriptors count ........ : "\
        +str(len(self._dict_breed_kpdesc)))
        
    #---------------------------------------------------------------------------
        

    #---------------------------------------------------------------------------
    # Properties
    #---------------------------------------------------------------------------
    def _get_dir_path(self) :
      return self._dir_path
    
    def _set_dir_path(self,dir_path) :
        self._dir_path = dir_path.copy()

    dir_path = property(_get_dir_path,_set_dir_path)
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def load(self) :
        '''Read all images from data directory path .
        Imges are stored into a dictionary with such structure : 
        {breed_directory : list_of_breed_file_names} 
        where list_of_breed_file_names is the list of file names that 
        reference images from breed.
        '''
        self._dict_data = p7_util.p7_load_data(self._dir_path)
        
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
    def resize(self) :
        ''' 
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
    def build_sift_desc(self) :
        image_count=0
        ratio = 5/100
        self._dict_breed_kpdesc = dict()
        for dirbreed, list_imagename in self._dict_breed_sample.items():
            image_count_ = 0
            for imagename  in list_imagename :
                #---------------------------------------------------------------
                # Load PIL image from file path
                #---------------------------------------------------------------
                imagepathname = self._build_pathname(dirbreed, imagename)
                pil_image = p7_util.p7_pil_image_load(imagepathname\
                , is_verbose=False, std_size=None)

                #---------------------------------------------------------------
                # Resize
                #---------------------------------------------------------------
                #pil_image = self.pil_resize(pil_image)
                try :
                    pil_image.resize(self._std_size)
                except AttributeError :
                    print("*** WARNING : attribute error for PIL image ")
                    continue
                #---------------------------------------------------------------
                # Gray transformation
                #---------------------------------------------------------------
                pil_image = self.pil_2gray(pil_image)
                
                #---------------------------------------------------------------
                # Equalization
                #---------------------------------------------------------------
                pil_image = self.pil_equalize(pil_image)
                
                #---------------------------------------------------------------
                # Store descriptor along with breed name. This will be usefull
                # for classification.
                #---------------------------------------------------------------
                breedname = get_breedname_from_dirbreed(dirbreed)
                kp, desc = p7_util.p7_gen_sift_features(np.array(pil_image))
                self._dict_breed_kpdesc[image_count] = (desc,breedname)
                
                #---------------------------------------------------------------
                # Closing PIL image
                #---------------------------------------------------------------
                pil_image.close()

                #---------------------------------------------------------------
                # Display progress
                #---------------------------------------------------------------
                if(0 == (image_count+1)%500 ) :
                    print("Images processed= "\
                    +str(image_count)+"/"+str(self._total_image))
                image_count +=1     
                image_count_ +=1

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
            
    #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------

