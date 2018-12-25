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

    if 'post_id' in request.args :
        post_id = int(request.args.get('post_id'))
        print("POST ID= "+str(post_id))
        list_tag_suggested, list_tag_suggested_fw, list_assigned_tags, body, title \
        = config.oP6_PostClassifier.process_post(post_id)
        print("Processing POST= "+str(post_id)+" done!\n")

    elif '*' in request.args :

        #-----------------------------------------------------------------------
        #
        #-----------------------------------------------------------------------
        print("\n Getting random POST...")
        list_tag_suggested, list_tag_suggested_fw, list_assigned_tags, body, title \
        = config.oP6_PostClassifier.process_post(None)
        print("Processing random POST done!\n")

    else :
        json_result = "{\"_result\":[{\"id\":UNKNOWN / ERROR}]"
        return json_result
    
    print("\n Json processing...")
    json_result \
    = config.oP6_PostClassifier.json_builder(list_tag_suggested\
    , list_tag_suggested_fw, list_assigned_tags, body, title)

    print("Json processing done!")
    print(json_result)
    return json_result
      

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run()
#-------------------------------------------------------------------------------


