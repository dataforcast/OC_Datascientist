#!/usr/bin/python3.6
import os

# Absolute directory path from where this file stands
basedir = os.path.abspath(os.path.dirname(__file__))

DUMPED_FILE='oP6_PostClassifier.dump'

CONFIG_PROD = True


if CONFIG_PROD is True:
    import P6_PostClassifier
    basedir = os.path.abspath(os.path.dirname(__file__))
    PATH_DUMPED_FILE = basedir+"/data/"+DUMPED_FILE
    print("\n*** Config : loading dumped object...")
    oP6_PostClassifier = P6_PostClassifier.load_dumped(PATH_DUMPED_FILE)
    print("*** Config : loading dumped done!\n")
    print("\n")
    oP6_PostClassifier.show()
else:
   pass

