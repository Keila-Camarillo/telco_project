# Telco Churn Project

# Project Description
Telco is a communication service provider (CSP) company that transports information electronically through telephony and data communication services in the networking industry. The Telco Churn dataset is utilized in this project to find drivers for customer churn at Telco. 

# Project Goal

* Discover drivers of churn in the telco dataset 
* Use drivers to develop a machine learning model to classify customers with the probablity of churning or not churning
* Churn is defined as a customer who left within the last month.
* This information could be used to further our understanding of which elements contribute to a customer churning.

# Initial Thoughts

My initial hypothesis is that drivers of churn will be elements such as dependents/partners, payment type (manual/ automatic), gender, paperless billing, and tenure.

# The Plan

* Aquire data from Codeup database

* Prepare data

- Create Engineered columns from existing data
    * churn
    * monthly_charge
    * dependents
    * partners
    * payment_type
    * gender
    * tenure
* Explore data in search of drivers of upsets

- Answer the following initial questions
    * How often are customers' churning?
    * What's the relationship between churn and automatic payment?
    * What's the relationship between churn and paperless billing?
    * What's the Relationship Between churn and customers' with a Partner?
    * What's the relationship between churn and customers' with dependents?
    * What's the relationship between churn and customers' gender?
    * Does tenure affect whether a customer will churn or not?

* Develop a Model to predict if a customer will churn

    - Use drivers identified in explore to build predictive models of different types
    - Evaluate models on train and validate data
    - Select the best model based on highest accuracy
    - Evaluate the best model on test data

* Draw conclusions



# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|Churn (Target)| 0 (Did Not Churn) or 1 (Did Churn), The customer churned|
|Automatic Payment| 0 (No) or 1 (Yes), The customer is setup for automatic payments|
|Paperless Billing| 0 (No) or 1 (Yes), The customer is setup for paperless billing|
|Partner| 0 (No) or 1 (Yes), The customer has a partner|
|Dependents| 0 (No) or 1 (Yes), The customer has dependents|
|Gender| 0 (Female) or 1 (Male), The customers' gender|
|Tenure| The customers' number of months at Telco|

# Steps to Reproduce
    1. Clone this repo.
    2. Acquire the data from Codeup Database - create username and password
    3. Place the data in the file containing the cloned repo.
    4. Run notebook.

# Takeaways and Conclusions
* "Automatic Payment" was found to be a driver of "Churn", customers' with auto pay set were not churning as often
* "Paperless Billing" was found to be a driver of "Churn",  customers' with paperless billing were churning more
* "Partner" was found to be a driver of "Churn", are more likely to churn if the customer does not have a partner
* "Dependents" was found to be a driver of "Churn", are more likely to churn if the customer does not have dependents
* "Gender" was not found to be a driver of "Churn"
* "Tenure" was found to be a driver of "Churn", Customers' with low 'Tenure' churned more than those who had high 'Tenure

# Recommendations
* To decrease the likelihood of customers churning offer a small incentive for customers to initiate automatic payments.
* To decrease the likelihood of customers' with low tenure, dependents, or partners from churning, offer these selected customers discounted rates for longer contracts.