import os
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
        self._cluster_model_name = str()
        self._df_bof = pd.DataFrame()
        self._y_label = np.array(0)
        self._dict_breedname_id = dict()
        self._is_splitted = False
        self._Xdesc = np.zeros(128)
        
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
    def show(self, legend=str()):
        '''Show classes attributes
        '''
        self.strprint("\n "+str(legend))
        
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
        
        self.strprint("Current cluster model  ........ : "\
        +self._cluster_model_name)
        self.strprint("Bag of features dataframe ..... : "\
        +str(self._df_bof.shape))
        self.strprint("Labels from dataframe ......... : "\
        +str(self._y_label.shape))
        self.strprint("Number of breeds .............. : "\
        +str(len(self._dict_breedname_id)))
        self.strprint("Image splitted ................ : "\
        +str(self._is_splitted))
        self.strprint("Key point descriptors ......... : "\
        +str(self._Xdesc.shape))
        
        self.strprint("")

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
        self._dict_cluster_model = object._dict_cluster_model.copy()
        self._cluster_model_name = object._cluster_model_name
        self._df_bof = object._df_bof.copy()
        self._y_label = object._y_label.copy()
        self._dict_breedname_id = object._dict_breedname_id.copy()
        self._is_splitted = object._is_splitted
        self._Xdesc = object._Xdesc.copy()
        
        if is_new_attribute is True :
            pass
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
      self._dict_cluster_model = dict_cluster_model.copy()


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
            print("\n*** WARN : method not authorized !\n")

    def _get_nb_cluster(self) :
        nb_cluster = 0
        if 'GMM' == self._cluster_model_name :
            nb_cluster \
            = self._dict_cluster_model[self._cluster_model_name].n_components
        else:
            print("\n*** WARN : cluster model does not exists in clusters dictionary !\n")
        return nb_cluster
   
    def _set_nb_cluster(self, nb_cluster) :
        print("\n*** WARN : method not authorized !\n")

    def _get_df_bof(self) :
      return self._df_bof
    def _set_df_bof(self, df_bof) :
      self._df_bof = df_bof.copy()

    def _get_X_train(self) :
      return self._X_train
    def _set_X_train(self, X_train) :
        print("\n*** WARN : method not authorized !\n")

    def _get_y_train(self) :
      return self._y_train
    def _set_y_train(self, y_train) :
        print("\n*** WARN : method not authorized !\n")

    def _get_X_test(self) :
      return self._X_test
    def _set_X_test(self, X_test) :
        print("\n*** WARN : method not authorized !\n")

    def _get_y_test(self) :
      return self._y_test
    def _set_y_test(self, y_test) :
        print("\n*** WARN : method not authorized !\n")

    def _get_ylabel(self) :
      return self._y_label
    def _set_ylabel(self, ylabel) :
        print("\n*** WARN : method not authorized !\n")

    def _get_dict_breedname_id(self) :
      return self._dict_breedname_id
    def _set_dict_breedname_id(self, dict_breedname_id) :
        print("\n*** WARN : method not authorized !\n")

    def _get_is_splitted(self) :
      return self._is_splitted
    def _set_is_splitted(self, is_splitted) :
        print("\n*** WARN : method not authorized !\n")

    def _get_Xdesc(self) :
      return self._Xdesc
    def _set_Xdesc(self, Xdesc) :
        print("\n*** WARN : method not authorized !\n")

    dir_path = property(_get_dir_path,_set_dir_path)
    std_size = property(_get_std_size,_set_std_size)
    df_desc  = property(_get_df_desc, _set_df_desc)
    
    sampling_breed_count  = property(_get_sampling_breed_count\
    , _set_sampling_breed_count)
    
    sampling_image_per_breed_count=property(_get_sampling_image_per_breed_count\
    , _set_sampling_image_per_breed_count)
    
    dict_cluster_model  = property(_get_dict_cluster_model\
    , _set_dict_cluster_model)
    cluster_model_name = property(_get_cluster_model_name,_set_cluster_model_name)
    
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
    def split_pil_image(self, pil_image, classname):
        width = int(self._std_size[0]/4)
        height = int(self._std_size[1]/4)
        dict_pil_image = dict()
        imgwidth, imgheight = pil_image.size
        for i in range(0,imgheight,height):
            list_pil_image_crop = list()
            for j in range(0,imgwidth,width):
                box = (j, i, j+width, i+height)
                list_pil_image_crop.append(pil_image.crop(box))
            classname_i = str(i)+'_'+classname
            dict_pil_image[classname_i]=  list_pil_image_crop  
        
        return dict_pil_image
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def build_sift_desc(self, is_splitted=False) :
        image_count=0
        image_count_split = 0
        ratio = 5/100
        self._dict_breed_kpdesc = dict()
        for dirbreed, list_imagename in self._dict_breed_sample.items():
            
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
                
                self._is_splitted = is_splitted
                if is_splitted is True :
                    dict_split_pil_image \
                    = self.split_pil_image(pil_image,breedname)
                    for id_breedname, list_split_pil_image \
                    in dict_split_pil_image.items() :
                        for split_pil_image in list_split_pil_image :
                            kp, desc = get_image_kpdesc(split_pil_image)
                            self._dict_breed_kpdesc[image_count_split] \
                            = (desc,breedname)
                            image_count_split +=1
                else :            
                    kp, desc = get_image_kpdesc(pil_image)
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
            dataframe columns : descriptors occuenrencies 
        '''
        
        #-----------------------------------------------------------------------
        # Get current cluster modeler and number of clusters
        #-----------------------------------------------------------------------
        nb_cluster = self.nb_cluster
        cluster_model = self.cluster_model
        df=pd.DataFrame(np.zeros(nb_cluster, dtype=int))
        
        if 0 >= nb_cluster :
            print("\n*** ERROR : No cluster into data model!")
        else:
            #-------------------------------------------------------------------
            # Initialization
            #-------------------------------------------------------------------
            dict_feature = dict()
            for i in range(0,nb_cluster) :
                dict_feature[i]=0     
            #-------------------------------------------------------------------
            # Get cluster from image represented as Key points descriptors
            #-------------------------------------------------------------------
            y_label = cluster_model.predict(desc)

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
        for image_id, (imagedesc, breedname) in self._dict_breed_kpdesc.items():
            df_tmp = self.get_cluster_from_imagedesc(imagedesc)
            # Index is matched with image id
            df_tmp.rename(index={0:image_id}, inplace=True)
            if df is None :
                df = df_tmp.copy()
            else:
                df = pd.concat([df, df_tmp])
            # Used for Y
            dict_label[image_id] = breedname
        self._y_label = np.array(list(self._dict_breed_kpdesc.keys()))
        #-----------------------------------------------------------------------
        # L2 Normalization with :
        # * L2 per column is computed summing all terms for any column
        # * Normalization is applied on all terms or any column 
        #-----------------------------------------------------------------------
        ser_sqrt = df.apply(lambda x: np.sqrt(x.dot(x.T)), axis=0)
        ser_sqrt
        df = df/ser_sqrt
    
        self._df_bof = df.copy()
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def breed_show(self) :
        print("")
        for name,id in self._dict_breedname_id.items() :
            print("Identifier= {}  Breed name= {}".format(id,name))
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def build_dict_breedname_id(self) :
        '''Build a dictionary with keys as breed name and values as breed 
        identifier name : {breedname:breed_id}
        '''
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
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def show_image_name(self, breedname):
        ser = pd.Series(self._dict_breedname_id)
        index = np.where(ser.keys()==breedname)[0][0]
        id = list(self._dict_breedname_id.values())[index]
        dirbreed = self._dir_path+'/'+str(id)+'-'+breedname
        list_image_name = os.listdir(dirbreed)
        print("")
        print("Number of images ="+str(len(list_image_name)))
        for image_name in list_image_name :
            print("Image name= {}".format(image_name))
        
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
        indexes of raws allow to identify image.
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
            print("*** ERROR : empty Key-points descriptors! build it with build_sift_desc()")
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
            print("\n*** WARN : Nb of exceptions during process ... : "+str(error))
        self._Xdesc = X_desc[1:].copy()
    #---------------------------------------------------------------------------
#-------------------------------------------------------------------------------

