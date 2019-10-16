# <center><h1>Flights delays estimator</h1></center>
<hr>

<br><br><br>
<h3>Abstract</h3>
<hr>

This study aims to estimate flights delays based on US TRANSTATS database 
provided on [https://www.transtats.bts.gov](URL)

<br>

  * The slides present the overall approach of the study : <a href="URL">https://github.com/dataforcast/OC_Datascientist/blob/master/P4/soutenance/Openclassrooms_ParcoursDatascientist_P4-V1.pdf</a>
<br>
  * Jupyter notebook, Python source code :  <a href="URL">http://bit.ly/FBangui_Datascience_FlightDelayEstimator</a>
<br>

<h3>Study Overview :</h3>
<hr>
 
* A Data model is built based on the dataset cleaning and exploratory analysis.
<br>
<img src="./P4_GlobalProcess.png" alt="Drawing" style="width: 200px;"/>


* The explicative variables are selected in order to remove data-leakage and 
data-correlation. 
<br>

Matrix correlation allows to detect correlated features : 
<img src="./P4_DataLeakage_2.png" alt="Drawing" style="width: 200px;"/>


* A model is built based on routes. Study of delay variances per routes lead to 
suggest assumptions on data model variables.
<br>
Scheme below shows route building process between two airports.

<br>
<img src="./P4_RouteBuilding.png" alt="Drawing" style="width: 200px;"/>

<br>
Delay distribution shows a shifted central effect for east/west trafic. A climatic effect may be suspected.
<img src="./P4_RouteDelayDistribution.png" alt="Drawing" style="width: 200px;"/>

<br>

* Model is augmented with the US climatic model.
<br>


* Linear regressors are evaluated based on an exhautive cross-validation. 
<br>
<img src="./P4_Benchmark.png" alt="Drawing" style="width: 200px;"/>

* The best estimator among linear regression estimators is selected based on performances measures. 

* An software engineering scheme is showned for the model to be deployed.

* Deployement scheme on Hereku is presented.
<br>

Upper part of slide below shows loading process on Heroku along with embedded database.

<br>
Middle part shows result of a request returned in JSON format. A set of random flights is returned from database.

<br>
Lower part of the slide shows result returned from a request for delay estimation. Note that both results are returned from two implemented algorithm in order to estimate variance of results.


<img src="./P4_DeploymentAndTetsResult.png" alt="Drawing" style="width: 200px;"/>

* Conclusions present limits of this model and propositions to increase model preformances.
<br>

<h3>API on JSON format:</h3>
<hr>

Following displays a random list of flights


 * [https://francois-bangui-oc-p4.herokuapp.com/predictor/?*](URL)

<br> 
Flight delay evaluation  based of flight identifier : 

 * [https://francois-bangui-oc-p4.herokuapp.com/predictor/?flight_id=ID](URL) 
 
where ID is picked from previous output.



<br>
<h3>Software engineering</h3>
<hr>

<img src="./P4_SoftwareEngineering_1.png" alt="Drawing" style="width: 200px;"/>
<br>
<img src="./P4_SoftwareEngineering_2.png" alt="Drawing" style="width: 200px;"/>

