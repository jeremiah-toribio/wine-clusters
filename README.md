# Wine Clustering
---
## Project Description 
The data sceince team assisting the California Wine Institute is interested to see if clustering the wine data will produce better results

## Project Goals
---
- Discover independence of features within home data
- Utilizing said features to develop machine learning models that will determine the price of a single-family home 
- Tax Value Amount is determined by the Property Value and annual Tax %, this can vary even between homes that are within close proximity
- The insights discovered will be used to estimate property value and provide a more robust understanding of the attributes of a home and it's associated value

## Initial Thoughts
--- 
When finding the price of a home, attributes like sq footage and bedrooms are certaintly what I think would drive the price. We can assess this through our tests and see if this is true.

## Planning
--- 
1. Acquire data from MySQL Server
2. Prepare data accordingly for exploration & machine learning needs
    a. Creating dummy columns of multi categorical data
        - county
    c. Cleaning numeric data
        - total_charges
3. Explore data for assocations with tax_value
    a. Determine our baseline prediction
    b. Does the number of bedroom directly effect price
    c. Is sq footage independent of tax_value
    d. Does pricing matter based on county
4. Develop a model to further understand churn
    a. Use various models to determine what algorithm are best to use
    b. Select best model based on evaluation metrics
    c. Evaluate all models on respective test data

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
2. Establish credentials in *env.py* file in order to access codeup MySQL server
    2a. OR have the csv already downloaded
3. Run notebook

## Conclusions
---


## Recommendation
---