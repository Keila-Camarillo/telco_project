# Telco Churn Project

# Project Description
Telco is a communication service provider (CSP) company that transports information electronically through telephony and data communication services in the networking industry. The Telco Churn dataset is utilized in this project to find drivers for customer churn at Telco. 

# Project Goal

* Discover drivers of churn in the telco dataset 
* Use drivers to develop a machine learning model to classify customers with the probablity of churning or not churning
* Churn is defined as a customer who left within the last month.
* This information could be used to further our understanding of which elements contribute to a customer churning.

# Initial Thoughts

My initial hypothesis is that drivers of churn will be elements such as cost, dependents/partners, and payment type (manual/ automatic).

# The Plan

* Aquire data from Codeup database

* Prepare data

- Create Engineered columns from existing data
    * churn
    * monthly_charge
    * dependents
    * partners
    * payment_type
* Explore data in search of drivers of upsets

- Answer the following initial questions
    * Do customers that have an automatic payment type more or less likely to churn?
    * Do customers who churn have a higher average monthly spend than those who don't?
    * Do customers that have partner more or less likely to churn?
    * Do customers that have dependents more or less likely to churn?

* Develop a Model to predict if a customer will churn

    - Use drivers identified in explore to build predictive models of different types
    - Evaluate models on train and validate data
    - Select the best model based on highest accuracy
    - Evaluate the best model on test data

* Draw conclusions


# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|Churn (Target)| True or False, The customer churned|

# Steps to Reproduce
    1. Clone this repo.
    2. Acquire the data from Codeup Database - create username and password
    3. Put the data in the file containing the cloned repo.
    4. Run notebook.

# Takeaways and Conclusions

# Recommendations