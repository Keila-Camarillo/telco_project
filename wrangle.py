import pandas as pd 
import env 
import os 
from sklearn.model_selection import train_test_split

def get_telco_data(directory=os.getcwd(), filename="telco_churn.csv"):
    """
    This function searches the local directory for the csv file and returns if exists.
    However, if csv doesn't exists it will create a df of the predefined SQL_query and write the df to csv.
    This function is currently set to output the telco_churn df from the current working directory.
"""
    SQL_query = ''' select * from customers
                    left join contract_types
                    using (contract_type_id)
                    left join customer_churn
                    using (customer_id)
                    left join customer_signups
                    using (customer_id)
                    join internet_service_types
                    using (internet_service_type_id)
                    join payment_types
                    using (payment_type_id);
'''

    if os.path.exists(directory + filename):
        df = pd.read_csv(filename) 
        return df
    
    else:
        df = pd.read_sql(SQL_query, env.get_db_url('telco_churn'))
#         df = new_telco_data(SQL_query)
        
        #want to save to csv
        df.to_csv(filename)
        return df

# def prep_telco(df=get_telco_data(directory=os.getcwd())):
#     '''
#     The function will clean the telco dataset
#     '''
#     # drop unecessary columns
#     df = df.drop(columns=["internet_service_type_id", "contract_type_id", "payment_type_id"])
    
#     # create dummies
#     dummy_df = pd.get_dummies(df[[
#                                  "multiple_lines",
#                                  "online_security",
#                                  "online_backup",
#                                  "device_protection", 
#                                  "tech_support",
#                                  "streaming_tv",
#                                  "streaming_movies", 
#                                  "contract_type", 
#                                  "internet_service_type",
#                                  "payment_type"]],
#                               drop_first=True)
#     df = df.drop(columns=["gender", "partner", "dependents", , "churn_month",  "internet_service_type", "payment_type", "tech_support", "signup_date" ])
#     df = pd.concat([df, dummy_df], axis=1)
#     df.total_charges = df.total_charges.str.replace(' ', '0').astype(float)
    return df

def prep_telco_2(df=get_telco_data(directory=os.getcwd())):
    '''
    The function will clean the telco dataset
    '''
    # drop unecessary columns
    df = df.drop(columns=["internet_service_type_id", "contract_type_id", "payment_type_id"])
    # encoding payment type automatic payment equals 1 and non_automatic equals 0
    df["payment_encoded"] = df.payment_type.map({"Bank transfer (automatic)": 1, "Credit card (automatic)": 1, "Mailed check": 0, "Electronic check": 0})
    
    # create dummies
    dummy_df = pd.get_dummies(df[["partner",
                                 "dependents", 
                                 "paperless_billing", 
                                 "churn"]],
                              drop_first=True)
    df = df.drop(columns=["gender", 
                            "online_security",
                            "multiple_lines",
                            "online_backup",
                            "device_protection", 
                            "tech_support",
                            "streaming_tv",
                            "streaming_movies", 
                            "contract_type", 
                            "internet_service_type",
                            "churn_month",  
                            "internet_service_type",
                            "tech_support",
                            "signup_date" ])
    df = pd.concat([df, dummy_df], axis=1)
    df.total_charges = df.total_charges.str.replace(' ', '0').astype(float)
    return df

def prep_telco(df=get_telco_data(directory=os.getcwd())):
    '''
    The function will clean the telco dataset.
    The function will also return to dataframes:
    df = for modeling
    exp_df = for exploring 
    '''
    # encoding payment type automatic payment equals 1 and non_automatic equals 0
    df["automatic_payment"] = df["payment_type"].map({"Bank transfer (automatic)": 1, "Credit card (automatic)": 1, "Mailed check": 0, "Electronic check": 0})
    exp_df = df[["partner", "dependents", "paperless_billing", "automatic_payment", "churn"]]
    # create dummies
    dummy_df = pd.get_dummies(df[["partner",
                                 "dependents", 
                                 "paperless_billing", 
                                 "churn"]],
                              drop_first=True)
    # df for explore
    exp_df = pd.concat([exp_df, dummy_df], axis=1)
    # rename columns
    df = exp_df.drop(columns=["partner", "dependents", "paperless_billing", "churn"])
    df = df.rename(columns={"partner_Yes": "partner", "dependents_Yes": "dependents", "paperless_billing_Yes": "paperless_billing", "churn_Yes": "churn" })
    # df for modeling
    df = df[["partner", "dependents", "paperless_billing", "automatic_payment", "churn"]]
    return df, exp_df

def split_data(df, target_variable):
    '''
    Takes in two arguments the dataframe name and the ("target_variable" - must be in string format) to stratify  and 
    return train, validate, test subset dataframes will output train, validate, and test in that order.
    '''
    train, test = train_test_split(df, #first split
                                   test_size=.2, 
                                   random_state=123, 
                                   stratify= df[target_variable])
    train, validate = train_test_split(train, #second split
                                    test_size=.25, 
                                    random_state=123, 
                                    stratify=train[target_variable])
    return train, validate, test