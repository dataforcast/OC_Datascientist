#!/usr/bin/python3.6
'''
Package containing all modules related to application.
'''

from flask import Flask
from .views import app


# This decorator allows to initialize application from CLI.
# It is invoked when this package is loaded.
# For running Flask CLI :
# --> FLASK_APP=run.py flask shell

# Function load() is called from CLI when following FLASK shell command 
# is entered : 
# --> FLASK_APP=run.py flask load
# Then method views.load() will be called.

# Inside views.load(), dumped file oLinearDelayPredictor.dump is loaded 
# into RAM.
@app.cli.command()
def load():
   print("\n--> Loading component...")
   views.load()
   print("--> Loading component done!\n")

