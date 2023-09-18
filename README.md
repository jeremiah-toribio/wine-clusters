# Wine Clustering
---
## Project Description 
The data science team assisting the California Wine Institute is interested to see if clustering the wine data will produce better results. The data will be clustered accordingly and the working team will report findings.

## Project Goals
---
- Discover independence of features within data
- Utilizing said features to develop machine learning models that will determine the quality of wine 
- Determine the drivers of quality
- The insights discovered will be used to estimate quality and provide a more robust understanding of the attributes of a the wine and it's associated quality

## Initial Thoughts
--- 
The drivers will likely have an equal weight when determining the quality -- this will be due to the quality likely scored by a professional who is cognizant of their biasis and would like to score on an objective scale. There will still likely be clusters based on data for the wines, since different wines will share qualities. 

## Planning
--- 
1. Acquire data from data.world
2. Prepare data accordingly for exploration & machine learning needs
    a. Creating dummy columns of multi categorical data
        - N/A
    c. Cleaning numeric data
        - Clean data types
3. Explore data for assocations with quality (correlation tests)
    a. Determine our baseline prediction
    b. Determine which features would be best to cluster
    c. Create new column of clustered data
4. Develop a model to further understand churn
    a. Use various models to determine what algorithm are best for the data
    b. Select best model based on evaluation metrics
    c. Evaluate all models on respective test data
    d. Tune hyperparameters

## Data Dictionary
--- 
| Feature        | Definition                                   |
| ---            | ---                                          |
| bedroomcnt     | int; a count of how many bedrooms in the house |
| bathroomcnt    | float; a count of how many bathrooms in the house |
| sq_feet    | int; the total sq footage of the property |
| tax_value        | int; **target** the predetermined tax value assessed off property value |
| yearbuilt    | int; the year the respective property was built |
| county         | string; the county the property is located in |


## Reproducability Requirements
---
1. Clone repo
2. Run notebook

## Conclusions (IN PROGRESS)
---


## Recommendation (IN PROGRESS)
---