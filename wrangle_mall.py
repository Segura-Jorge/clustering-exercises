## IMPORTS ##

import pandas as pd
import numpy as np
from env import user, password, host
import os
directory = os.getcwd()
from sklearn.model_selection import train_test_split

def acquire_mall(
    user=user, 
    password=password, 
    host=host,
    db='mall_customers'
                ) -> pd.DataFrame:
    '''
    acquire_mall will make a request to our msql server associated
    with the credentials taken from an imported env.py
    please check credentials and access to the mall_customers
    schema before interacting with acquisition script
    
    return: a single pandas dataframe with all mall customer data
    '''
    if os.path.exists('./mall_customers.csv'):
        return pd.read_csv('mall_customers.csv', index_col=0)
    else:
        connection = f'mysql+pymysql://{user}:{password}@{host}/{db}'
        query = 'SELECT * FROM customers'
        df = pd.read_sql(query, connection)
        df.to_csv('mall_customers.csv')
        return df
    
def missing_by_row(df):
    return pd.concat(
        [
            df.isna().sum(axis=1),
            (df.isna().sum(axis=1) / df.shape[1])
        ], axis=1).rename(
        columns={0:'missing_cells', 1:'percent_missing'}
    ).groupby(
        ['missing_cells',
         'percent_missing']
    ).count().reset_index().rename(columns = {'index': 'num_mising'})

# acquisition, huzzah!
# lets move on to preparation and summarization:
def summarize(df) -> None:
    '''
    Summarize will take in a dataframe and report out statistics
    regarding the dataframe to the console.
    
    this will include:
     - the shape of the dataframe
     - the info reporting on the dataframe
     - the descriptive stats on the dataframe
     - missing values by column
     - missing values by row
     
    '''
    print('--------------------------------')
    print('--------------------------------')
    print('Information on DataFrame: ')
    print(f'Shape of Dataframe: {df.shape}')
    print('--------------------------------')
    print(f'Basic DataFrame info:')
    print(df.info())
    print('--------------------------------')
    # print out continuous descriptive stats
    print(f'Continuous Column Stats:')
    print(df.describe().to_markdown())
    print('--------------------------------')
    # print out objects/categorical stats:
    print(f'Categorical Column Stats:')
    print(df.select_dtypes('O').describe().to_markdown())
    print('--------------------------------')
    print('Missing Values by Column: ')
    print(df.isna().sum())
    print('Missing Values by Row: ')
    print(missing_by_row(df).to_markdown())
    print('--------------------------------')
    print('--------------------------------')

def prep_mall(df):
    '''
    prep_mall will take in values in form of a single pandas dataframe
    and take the necessary preparation steps in order to turn it into a clean
    df!
    
    '''
    #capture any missing values and handle them (impute, drop, etc)
    # conveniently no missing values on this specific one
    # rename some columns:
    # we have the arbitrary customer id field that appears to be an index
    # so lets mark it as such:
    df = df.set_index('customer_id')
    return df


def split_data(df):
    '''
    based on an input dataframe,
    we will split the information into train, val, test
    an return three dataframes
    '''
    train_val, test = train_test_split(
        df,
        train_size=0.8,
        random_state=1349)
    train, validate = train_test_split(
        train_val,
        train_size=0.7,
        random_state=1349)
    return train, validate, test

def preprocess_mall(df):
    '''
    preprocess_mall will take in values in form of a single pandas dataframe
    and make the data ready for spatial modeling,
    including:
     - splitting the data
     - encoding categorical features
     - scaling information (continuous columns)

    return: three pandas dataframes, ready for modeling structures.
    '''
    #capture any missing values and handle them (impute, drop, etc)
    # conveniently no missing values on this specific one
    # rename some columns:
    # we have the arbitrary customer id field that appears to be an index
    # so lets mark it as such:
    # encode categoricals:
    df = df.assign(
        is_male= pd.get_dummies(
            df['gender'], drop_first=True
        ).astype(int).values)
    # drop original gender col:
    df = df.drop(columns='gender')
    # split data:
    train, validate, test = split_data(df)
    # scale continuous features:
    scaler = MinMaxScaler()
    train = pd.DataFrame(
        scaler.fit_transform(train),
        index=train.index,
        columns=train.columns)
    validate = pd.DataFrame(
        scaler.transform(validate),
        index=validate.index,
        columns=validate.columns
    )
    test = pd.DataFrame(
        scaler.transform(test),
        index=test.index,
        columns=test.columns)
    return train, validate, test