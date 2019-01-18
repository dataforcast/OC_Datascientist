import pandas as pd
import numpy as np
import pickle

# --------------------------------------------------------------------
# Useful sites 
# http://www.python-simple.com/python-pandas/dataframes-indexation.php
# --------------------------------------------------------------------

class SeriesExplorer() :
    """
    Cette classe permet de mener opérations exploiratoires sur des series.
    """
    INTERNATIONAL_FR={
        'PROTEINES' : 'PROTEINE',
        'GLUCIDESES' : 'GLUCIDES',
        'FATACID' : 'ACIDES_GRAS',
        'ACIDES_GRAS' : 'ACIDES_GRAS',
        'VITAMINES' : 'VITAMINESES',
        'SELS MINERAUX' : 'SEL MINERAUX',
        'ALCOHOL' : 'ALCOOL',
        'LIPIDES' : 'LIPIDES',
        'FATACID' : 'ACIDES_GRAS',
        'FIBERS' : 'FIBRES',

    
    }
    # Les categories de nutriments sont stockés en variables de classe sous forme de dictionaires.
    NUTRITION_QUALITY_CATEG = {
                    # ---------  Protéines ------------------------
                    'proteins_100g' : 'PROTEINES',
                    'casein_100g' : 'PROTEINES',
                    'serum-proteins_100g' : 'PROTEINES',
                    'nucleotides_100g' : 'PROTEINES',
                    'biotin_100g' : 'PROTEINES',
        
                    # -------- Acides gras insaturés --------------
                    'alpha-linolenic-acid_100g' : 'ACIDES GRAS',# AG polyinsaturé  : classe oméga-3
                    'eicosapentaenoic-acid_100g' : 'ACIDES GRAS',# AG polyinsaturé  : classe oméga-3
                    'docosahexaenoic-acid_100g' : 'ACIDES GRAS',# AG polyinsaturé  : classe oméga-3
                    'linoleic-acid_100g' : 'ACIDES GRAS',        # AG polyinsaturé  : classe oméga-6
                    'arachidonic-acid_100g' : 'ACIDES GRAS',     # AG polyinsaturé  : classe oméga-6     
                    'gamma-linolenic-acid_100g' : 'ACIDES GRAS', # AG polyinsaturé  : classe oméga-6  
                    'dihomo-gamma-linolenic-acid_100g' : 'ACIDES GRAS',        # AG insaturés :classe oméga-6
                    'oleic-acid_100g' : 'ACIDES GRAS',       # AG mono-insaturés :classe  oméga-9
                    'elaidic-acid_100g' : 'ACIDES GRAS',     # AG insaturé
                    'gondoic-acid_100g' : 'ACIDES GRAS',     # AG insaturé
                    'mead-acid_100g' : 'ACIDES GRAS',        # AG insaturé
                    'erucic-acid_100g' : 'ACIDES GRAS',      # AG insaturé
                    'nervonic-acid_100g' : 'ACIDES GRAS',    # AG insaturé                           
                    'pantothenic-acid_100g' : 'ACIDES GRAS',  #No

                    # -------- Vitamines  --------------
                    'vitamin-a_100g' : 'VITAMINES',        
                    'vitamin-d_100g' : 'VITAMINES',        
                    'vitamin-e_100g' : 'VITAMINES',        
                    'vitamin-k_100g' : 'VITAMINES',        
                    'vitamin-c_100g' : 'VITAMINES',        
                    'vitamin-b1_100g' : 'VITAMINES',        
                    'vitamin-b2_100g' : 'VITAMINES',        
                    'vitamin-pp_100g' : 'VITAMINES',        
                    'vitamin-b6_100g' : 'VITAMINES',        
                    'vitamin-b9_100g' : 'VITAMINES',        
                    'vitamin-b12_100g' : 'VITAMINES',                            
                    
                    # -------- Sels minéraux  --------------
                    'taurine_100g' : 'SELS MINERAUX',        
                    'caffeine_100g' : 'SELS MINERAUX',        
                    'iodine_100g' :   'SELS MINERAUX',        
                    'molybdenum_100g' : 'SELS MINERAUX',        
                    'chromium_100g' : 'SELS MINERAUX',        
                    'selenium_100g' : 'SELS MINERAUX',        
                    'fluoride_100g' : 'SELS MINERAUX',        
                    'manganese_100g' : 'SELS MINERAUX',        
                    'copper_100g' :    'SELS MINERAUX',        
                    'zinc_100g' :      'SELS MINERAUX',        
                    'magnesium_100g' : 'SELS MINERAUX',        
                    'iron_100g' :       'SELS MINERAUX',        
                    'phosphorus_100g' : 'SELS MINERAUX',        
                    'calcium_100g' :    'SELS MINERAUX',        
                    'sodium_100g' :     'SELS MINERAUX',        
                    'silica_100g' :     'SELS MINERAUX',        
                    'bicarbonate_100g' :'SELS MINERAUX',        
                    'potassium_100g' :  'SELS MINERAUX',        
                    'chloride_100g' :   'SELS MINERAUX',        
        
    }
    
    NUTRITION_CATEG = { 
                    'proteins_100g' : 'PROTEINES',
                    'casein_100g' : 'PROTEINES',
                    'serum-proteins_100g' : 'PROTEINES',
                    'nucleotides_100g' : 'PROTEINES',
                    'biotin_100g' : 'PROTEINES',
                    
                    'carbohydrates_100g' : 'GLUCIDES',
                    
                    'sugars_100g' : 'GLUCIDES',
        
                    'sucrose_100g' : 'GLUCIDES',
                    'glucose_100g' : 'GLUCIDES',
                    'fructose_100g' :'GLUCIDES',
                    'lactose_100g' : 'GLUCIDES',
                    'maltose_100g' : 'GLUCIDES',
                    'lactose_100g' : 'GLUCIDES',
                    'maltodextrins_100g' : 'GLUCIDES',
                    'starch_100g' : 'GLUCIDES',
                    'polyols_100g' : 'GLUCIDES',
                    
                    # --- Acides gras saturés ------
                    'caproic-acid_100g' : 'ACIDES GRAS',#AG saturé
                    'caprylic-acid_100g' : 'ACIDES GRAS',#AG saturé
                    'capric-acid_100g' : 'ACIDES GRAS',#AG saturé
                    'lauric-acid_100g' : 'ACIDES GRAS',#AG saturé
                    'myristic-acid_100g' : 'ACIDES GRAS',#AG saturé (très répandu des AGS)
                    'palmitic-acid_100g' : 'ACIDES GRAS',#AG saturé (très répandu des AGS)
                    'stearic-acid_100g' : 'ACIDES GRAS',# AG saturé (le plus répandu des AGS)
                    'arachidic-acid_100g' : 'ACIDES GRAS',# AG saturé
                    'behenic-acid_100g' : 'ACIDES GRAS',# AG saturé
                    'lignoceric-acid_100g' : 'ACIDES GRAS',# AG saturé
                    'cerotic-acid_100g' : 'ACIDES GRAS',# AG saturé
                    'montanic-acid_100g' : 'ACIDES GRAS', # AG saturé
                    'melissic-acid_100g' : 'ACIDES GRAS', # AG saturé
        
                    # --- Acides gras insaturés ------
                    'alpha-linolenic-acid_100g' : 'ACIDES GRAS',# AG polyinsaturé  : classe oméga-3
                    'eicosapentaenoic-acid_100g' : 'ACIDES GRAS',# AG polyinsaturé  : classe oméga-3
                    'docosahexaenoic-acid_100g' : 'ACIDES GRAS',# AG polyinsaturé  : classe oméga-3
                    'linoleic-acid_100g' : 'ACIDES GRAS',        # AG polyinsaturé  : classe oméga-6
                    'arachidonic-acid_100g' : 'ACIDES GRAS',     # AG polyinsaturé  : classe oméga-6     
                    'gamma-linolenic-acid_100g' : 'ACIDES GRAS', # AG polyinsaturé  : classe oméga-6  
                    'dihomo-gamma-linolenic-acid_100g' : 'ACIDES GRAS',        # AG insaturés :classe oméga-6
                    'oleic-acid_100g' : 'ACIDES GRAS',       # AG mono-insaturés :classe  oméga-9
                    'elaidic-acid_100g' : 'ACIDES GRAS',     # AG insaturé
                    'gondoic-acid_100g' : 'ACIDES GRAS',     # AG insaturé
                    'mead-acid_100g' : 'ACIDES GRAS',        # AG insaturé
                    'erucic-acid_100g' : 'ACIDES GRAS',      # AG insaturé
                    'nervonic-acid_100g' : 'ACIDES GRAS',    # AG insaturé                   
                    'pantothenic-acid_100g' : 'ACIDES GRAS',  #No
        
                    'vitamin-a_100g' : 'VITAMINES',        
                    'vitamin-d_100g' : 'VITAMINES',        
                    'vitamin-e_100g' : 'VITAMINES',        
                    'vitamin-k_100g' : 'VITAMINES',        
                    'vitamin-c_100g' : 'VITAMINES',        
                    'vitamin-b1_100g' : 'VITAMINES',        
                    'vitamin-b2_100g' : 'VITAMINES',        
                    'vitamin-pp_100g' : 'VITAMINES',        
                    'vitamin-b6_100g' : 'VITAMINES',        
                    'vitamin-b9_100g' : 'VITAMINES',        
                    'vitamin-b12_100g' : 'VITAMINES',                            
                    
                    'taurine_100g' : 'SELS MINERAUX',        
                    'caffeine_100g' : 'SELS MINERAUX',        
                    'iodine_100g' :   'SELS MINERAUX',        
                    'molybdenum_100g' : 'SELS MINERAUX',        
                    'chromium_100g' : 'SELS MINERAUX',        
                    'selenium_100g' : 'SELS MINERAUX',        
                    'fluoride_100g' : 'SELS MINERAUX',        
                    'manganese_100g' : 'SELS MINERAUX',        
                    'copper_100g' :    'SELS MINERAUX',        
                    'zinc_100g' :      'SELS MINERAUX',        
                    'magnesium_100g' : 'SELS MINERAUX',        
                    'iron_100g' :       'SELS MINERAUX',        
                    'phosphorus_100g' : 'SELS MINERAUX',        
                    'calcium_100g' :    'SELS MINERAUX',        
                    'sodium_100g' :     'SELS MINERAUX',        
                    'silica_100g' :     'SELS MINERAUX',        
                    'bicarbonate_100g' :'SELS MINERAUX',        
                    'potassium_100g' :  'SELS MINERAUX',        
                    'chloride_100g' :   'SELS MINERAUX',        

                    #'alcohol_100g' : 'ALCOHOL',        

                    'fiber_100g' : 'FIBRES',        
                    
                    'saturated-fat_100g' : 'LIPIDES',
                    'monounsaturated-fat_100g' : 'LIPIDES',
                    'polyunsaturated-fat_100g' : 'LIPIDES',
                    'omega-3-fat_100g' : 'LIPIDES',
                    'omega-6-fat_100g' : 'LIPIDES',
                    'omega-9-fat_100g' : 'LIPIDES',
                    'trans-fat_100g' : 'LIPIDES',
                    'cholesterol_100g' : 'LIPIDES',
                    #'fat_100g' : 'LIPIDES',
                   }





    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def __init__(self, series) :
        """ 
        """
        self._series = series
        self._lValue = None
        self._oldLines = len(series)
        self._newLines = 0
        
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #  Properties
    #---------------------------------------------------------------------------
    def _get_lValue(self) :
      return self._lValue

    lValue = property(_get_lValue)
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def toList(self) :
        """
        This method creates a list from series attribute.
        Duplicated values are removed for having single value.
        Nan values are removed.
        """
        
        if self._lValue is not None :
            return

        aValue = self._series.values
        lValue = []
        
        for value in aValue :
            if value is not np.nan :
                lValue.append(value)

        print("Lines before processing = {}".format(self._oldLines))
        
        # Duplicated data are removed.
        sValue = set(lValue)
        
        # List is rebuilt from Set type
        del(lValue)
        lValue = list(sValue)
        try :
            lValue.sort()
        except TypeError :
            print("*** WARNING : LIST CAN'T BE SORTED! ***")
        self._lValue = lValue
        self._newLines = len(self._lValue)
        print("Lines after processing = {}".format(self._newLines))
    #---------------------------------------------------------------------------        
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def display(self) :
        """
        This method allows to display content from series 
        """
        self.toList()
        print("Old lines count= {}".format(self._oldLines))
        print("New lines count= {}".format(self._newLines))
        for value in self._lValue :
            print(value)
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def write(self, fileName="dump") :
        """ This method allows to write into a file content from self._lValue
        """
        
        fileName += '.txt'

        if self._lValue is None :
            self.toList()
            
        print("*** Dumping into file = {} ....".format(fileName))
        text_file = open(fileName, "w")
        for value in self._lValue :
            text_file.write(str(value)+'\n')
        text_file.close()
            
        print("*** Dumped into file= {} Done!".format(fileName))
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def mergeSeries(cls, series1, series2) :
        """ Merge series2 into series1 and return series1
        Merge takes place if series1 elt is nan and series2 elt is not nan.
        """
        list1 = series1.tolist()
        list2 = series2.tolist()
        
        for key, elt in enumerate(list1) :
            if list1[key] is np.nan :
                if  list2[key] is not np.nan :
                    list1[key] = list2[key]
        series1 = pd.Series(list1)
        return series1
    #---------------------------------------------------------------------------
    # This is a class method
    mergeSeries = classmethod(mergeSeries)
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def read(cls, fileName) :
        """ Static method for readin a file.
            return : a list containing data read from file.
        """
        lValue = list()
        try :
            with open(fileName, 'r') as myFile:
                lineText = myFile.read()
                lValue = lineText.split('\n')
        except FileNotFoundError :
            print("*** ERROR : file not found = "+fileName)

        #Purger les éléments vide de la liste
        lValue = [elt for elt in lValue if elt is not '']
        
        # Purger les caracteres au dela du motif  '100g'
        tmpList = list()
        for elt in lValue :
            pos1 = elt.find('100g') 
            pos1+=len('100g')
            tmpList.append(elt[0:pos1])
        
        lValue = tmpList
        return lValue
    
    # On s'assure de l'encodage LATIN
    
    # This is a class method
    read = classmethod(read)
    #---------------------------------------------------------------------------
