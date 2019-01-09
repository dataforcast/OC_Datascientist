import pandas as pd
from PIL import ImageOps
import numpy as np
import random
from  sklearn import model_selection
import p7_util

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def process_breed_sample(dirbreed, list_image_name, resize) :
    dict_pil_image = dict()
    dict_pil_image['resize']   = list()
    dict_pil_image['orig']     = list()
    dict_pil_image['2gray']    = list()
    dict_pil_image['equalize'] = list()
    for image_name in list_image_name :
        image_path_name = dirbreed+'/'+image_name
        pil_image = p7_read_image(image_path_name)
        dict_pil_image['orig'].append([pil_image])

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
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None
        self._sampling_breed_count = 0
        self._sampling_image_per_breed_count = 0
        self._dict_cluster_model = dict()
        
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
    #---------------------------------------------------------------------------
        
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def copy(self, object, is_new_attribute=True) :
        '''' Copies attributes from object given as parameter into 
        this object.'''
        self._dir_path = object.dir_path

        self._dict_data = object._dict_data.copy()
        self._total_image = object._total_image
        self.is_verbose = object.is_verbose
        self._std_size = object._std_size
        self._dict_img_pil = object._dict_img_pil.copy()
        self._dict_breed_kpdesc = object._dict_breed_kpdesc.copy()
        self._dict_breed_sample = object._dict_breed_sample.copy()
        self._list_breed_sample = object._list_breed_sample.copy()
        if object._X_train is not None :
            self._X_train = object._X_train.copy()
        if object._y_train is not None :
            self._y_train = object._y_train.copy()
        if object._X_test is not None :
            self._X_test = object._X_test.copy()
        if object._y_test is not None :
            self._y_test = object._y_test.copy()
        self._sampling_breed_count =object._sampling_breed_count
        self._sampling_image_per_breed_count \
        = object._sampling_image_per_breed_count

        if is_new_attribute is True :
            self._dict_cluster_model = object._dict_cluster_model.copy()
            #pass
        else :
            print("\n*** WARN : new attributes from object are not copied on target!\n")
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
      return pd.DataFrame(self._X_train)
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
      self._dict_cluster_model = dict_cluster_model.copy()


    dir_path = property(_get_dir_path,_set_dir_path)
    std_size = property(_get_std_size,_set_std_size)
    df_desc  = property(_get_df_desc, _set_df_desc)
    
    sampling_breed_count  = property(_get_sampling_breed_count\
    , _set_sampling_breed_count)
    
    sampling_image_per_breed_count=property(_get_sampling_image_per_breed_count\
    , _set_sampling_image_per_breed_count)
    
    dict_cluster_model  = property(_get_dict_cluster_model\
    , _set_dict_cluster_model)
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
                pil_image = pil_2gray(pil_image)
                
                #---------------------------------------------------------------
                # Equalization
                #---------------------------------------------------------------
                pil_image = pil_equalize(pil_image)
                
                #---------------------------------------------------------------
                # Store descriptor along with breed name. This will be usefull
                # for classification.
                #---------------------------------------------------------------
                breedname = get_breedname_from_dirbreed(dirbreed)
                kp, desc = get_image_kpdesc(pil_image)
                #p7_util.p7_gen_sift_features(np.array(pil_image))
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
        self._sampling_breed_count = breed_count
        self._sampling_image_per_breed_count = image_per_breed_count
        
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
    def train_test_build(self):
        X = np.zeros(128)
        y = list()
        
        for id, (desc,breedname) in self._dict_breed_kpdesc.items():
            X = np.vstack((X,desc))
            for k in range(0,desc.shape[0]):
                y.append(breedname)
        y = np.array(y)
        X = X[1:,:].copy()    

        self._X_train, self._X_test, self._y_train,  self._y_test \
        = model_selection.train_test_split(X,y,test_size=0.1)
    #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------

