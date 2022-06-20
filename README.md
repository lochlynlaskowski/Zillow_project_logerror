What is driving the errors in the Zestimates?

Project Goal: 
- Determine drives of logerror within Zillow's model utilizing regression models. 
- Create clusters and explore data within them.
- Create a regression model that out performs Zillow's model when predicting log error.

Note* In the prior Zillow project, I asked for additional time to explore geographical impacts, this project will explore mostly location based clusters.


Initial Hypotheses:
- Log error depends on the geographic location. 
- Age, square feet, and tax value are related to log error.

Conclusions:
- Each geographic location has it's own features that drive log error.
- Polynomial regression model was able to outperform Zillow's model on 4 out of 7 counties.
- The best performing model was tested on a subset of a cluster- Region 5C.

 Plan:
  - Hypothesize initial questions
  - Determine deliverables
  - Construct clear goals 

 Acquire:
  - Import zillow data from Codeup database
 - Preparation steps include:
    - Removing columns and rows with more than 30% data missing.
    - Removed additional rows with null values.
    - IQR outlier function utilized    
    - Converted data types to integers
    - Labeled counties by name
    - Removed erroneous or duplicated columns
    - Created age, month_of_sale, taxvalue_per_sqft
    - Remaining data comprised of 15 columns and 33,660 rows

  - Data dictionary:
    - logerror (**target**)
    - bathroomcnt
    - bedroomcnt
    - calculatedfinishedsquarefeet
    - fips - converted to Orange, LA, and Ventura counties
    - latitude
    - longitude
    - lotsizesquarefeet
    - yearbuilt
    - taxvaluedollarcnt (tax appraised value)
    - transactiondate
    - propertylandusedesc
    - age(2017-yearbuilt)
    - taxvalue_per_sqft (calculatedfinishedsquarefeet/taxvaluedollarcnt)
    - month of sale

 Explore:
  - Explore the interactions of features and target variable to determine drivers of logerror.
  - Utilize univarate and bivariate exploration, incorporporate hypothesis testing to confirm or deny initial hypotheses.
  - Create clusters for further exploration.
 
 Model:
  - Utilize regression models to predict logerror
  - Show the three best models
  - Test the best performing model (determined by results of train and validate datasets)


Conclusion:
 - Clusters were created geographically, allowing for modeling on smaller geographical areas.
 - Polynomial regression beat baseline in 4 out of 7 counties.
     - Determine the significance level in improvements of models from baseline.
 
- With additional time:
     - Continue exploration on features within regions.
     - Explore different models per region.
     


Steps to Reproduce
You will need an env.py file with credentials to the Codeup database.
Clone this repo (including wrangle.py)
Import python libraries: pandas, matplotlib, seaborn, numpy, and sklearn
Run final_report Zillow FINAL.ipynb 