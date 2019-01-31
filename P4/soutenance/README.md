# Set of projects 
Projects carried out on behalf of the Master 2 Centrale Supélec / Openclassrooms training in datasience

## <center><h1>Flights delays estimator</h1></center>
<hr>

<br><br><br>
<h3>Abstract</h3>
<hr>

This study aims to estimate flights delays based on US TANSTATS database 
provided on https://www.transtats.bts.gov

The slides (Openclassrooms_ParcoursDatascientist_P4-V1.pdf) 
present the overall approach of the study.
<br>

**Study steps :**

* A Data model is built based on the dataset cleaning and exploratory analysis.

* The explicative variables are selected in order to remove data-leakage and 
data-correlation. 

* A model is built based on routes. Study of delay variances per routes lead to 
suggest assumptions on data model variables.

* Model is augmented with the US climatic model.

* Linear regressors are evaluated based on an exhautive cross-validation. 

* The best estimator among linear regression estimators is selected based on performances measures. 

* An software engineering scheme is showned for the model to be deployed.

* Deployement scheme on Hereku is presented.

* Conclusions present limits of this model and possibles propositions to increase 
preformances.
<br>

**API on JSON format: **

Following displays a random list of flights
 * https://francois-bangui-oc-p4.herokuapp.com/predictor/?*

<br> 
Flight delay evaluation  based of flight identifier : 

 * https://francois-bangui-oc-p4.herokuapp.com/predictor/?flight_id=ID
 
where ID is picked from previous output.
