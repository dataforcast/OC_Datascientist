##!/usr/bin/python
##-*- coding: utf-8 -*-

from flask import Flask
from flask import render_template
from flask import url_for
from flask import request
from flask import jsonify

import config
import P6_PostClassifier

#print("The value of config.DB_ENGINE_NAME is {0}".format(config.DB_ENGINE_NAME))

app = Flask(__name__)
app.config.from_object('config')

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def load() :
    '''Dumped model is loaded.
    '''
   oP6_PostClassifier = P6_PostClassifier.load_dumped()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
# Any function decored with @app.route() decorator is a view.
# View search for a route identified into a HTPP request.
# Once function matching proper route is found, mean, 'index', then server calls 
# this function.

# 'render_template' displays the given template file in parameter. 
# A template files allows to  # display HTML file into a browser with parameters 
# issues from config.py file and from python objects.


@app.route('/')
@app.route('/predictor/')
def predictor():
   #oLinearDelayPredictor = LinearDelayPredictor.load_dumped(config.PATH_DUMPED_FILE)

   if 'flight_id' in request.args :
      flight_id = int(request.args.get('flight_id'))
      print("Flight ID= "+str(flight_id))

      json_result = config.oLinearDelayPredictor.evaluate_delay(flight_id)
      return json_result
   
   elif '*' in request.args :
      nb_flights = config.NB_FLIGHTS

      #-------------------------------------------------------------------------
      # Building json structure with a list of nb_flights flights to be selected.
      #-------------------------------------------------------------------------
      print("\n Getting flight selection...")
      df_selection = config.oLinearDelayPredictor.get_random_flights(nb_flights)
      print("Getting flight selection done!\n")

      print("\n Json selection processing...")
      json_selection = config.oLinearDelayPredictor.json_selection_builder(df_selection)
      print("Json selection processing done!")
      print(json_selection)
      return json_selection
   else :
      json_result = "{\"_result\":[{\"id\":UNKNOWN / ERROR}]"
      return json_result
      

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run()
#-------------------------------------------------------------------------------


