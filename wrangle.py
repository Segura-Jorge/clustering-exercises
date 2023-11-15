## IMPORTS ##

import pandas as pd
import numpy as np
from env import user, password, host
import os
directory = os.getcwd()
from sklearn.model_selection import train_test_split


## FUNCTIONS ##
##-------------------------------------------------------------------##
def get_db_url(database_name):
    """
    this function will:
    - take in a string database_name 
    - return a string connection url to be used with sqlalchemy later.
    """
    return f'mysql+pymysql://{user}:{password}@{host}/{database_name}'

def acquire_zillow():
    """
    This function will:
    - take in a SQL_query
    - create a connection_url to mySQL
    - return a df of the given query from the zillow
    """
    sql_query = """
        SELECT 
    p.*, 
    ac.*, 
    pr.*, 
    plt.*, 
    arc.*, 
    bc.*, 
    ht.*, 
    pl.*, 
    st.*, 
    tct.*
FROM 
    zillow.properties_2017 AS p
LEFT JOIN 
    airconditioningtype AS ac ON p.airconditioningtypeid = ac.airconditioningtypeid
LEFT JOIN 
    architecturalstyletype AS arc ON p.architecturalstyletypeid = arc.architecturalstyletypeid
LEFT JOIN 
    buildingclasstype AS bc ON p.buildingclasstypeid = bc.buildingclasstypeid
LEFT JOIN 
    heatingorsystemtype AS ht ON p.heatingorsystemtypeid = ht.heatingorsystemtypeid
LEFT JOIN 
    propertylandusetype AS pl ON p.propertylandusetypeid = pl.propertylandusetypeid
LEFT JOIN 
    storytype AS st ON p.storytypeid = st.storytypeid
LEFT JOIN 
    typeconstructiontype AS tct ON p.typeconstructiontypeid = tct.typeconstructiontypeid
INNER JOIN 
    predictions_2017 AS pr ON p.id = pr.id 
INNER JOIN 
    (SELECT 
         id, 
         MAX(transactiondate) AS MaxDate 
     FROM 
         predictions_2017 
     GROUP BY 
         id) AS latest_pr ON pr.id = latest_pr.id AND pr.transactiondate = latest_pr.MaxDate
LEFT JOIN 
    propertylandusetype AS plt ON p.propertylandusetypeid = plt.propertylandusetypeid
WHERE 
    YEAR(pr.transactiondate) = 2017;

        """
    
    url = get_db_url('zillow')
    
    df = pd.read_sql(sql_query, url)
    
    return df


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
    print(df.describe())
    print('--------------------------------')
    # print out objects/categorical stats:
    print(f'Categorical Column Stats:')
    print(df.select_dtypes('O').describe())
    print('--------------------------------')
    print('Missing Values by Column: ')
    print(df.isna().sum())
    print('Missing Values by Row: ')
    print(missing_by_row(df))
    print('--------------------------------')
    print('--------------------------------')

    
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





def prep_zillow(df):
    '''
    This function takes in a dataframe
    renames the columns and drops nulls values
    Additionally it changes datatypes for appropriate columns
    and renames fips to actual county names.
    Then returns a cleaned dataframe
    '''
    df = df.set_index('id')
    df = df.drop(columns=['airconditioningtypeid','architecturalstyletypeid','basementsqft', 'buildingclasstypeid', 'decktypeid',\
                          'finishedfloor1squarefeet', 'finishedsquarefeet13', 'finishedsquarefeet15', 'finishedsquarefeet50',\
                          'finishedsquarefeet6', 'poolsizesum', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'propertyzoningdesc',\
                          'rawcensustractandblock', 'regionidcounty', 'regionidneighborhood', 'storytypeid', 'threequarterbathnbr',\
                          'typeconstructiontypeid', 'unitcnt', 'yardbuildingsqft17', 'yardbuildingsqft26', 'numberofstories',\
                          'fireplaceflag', 'taxdelinquencyflag', 'taxdelinquencyyear', 'airconditioningtypeid', 'airconditioningdesc',\
                          'architecturalstyletypeid', 'architecturalstyledesc', 'buildingclasstypeid', 'buildingclassdesc',\
                          'storytypeid', 'storydesc', 'typeconstructiontypeid', 'typeconstructiondesc'])
    df = df.rename(columns = 
                   {'bedroomcnt':'bedrooms',
                    'bathroomcnt':'bathrooms',
                    'landtaxvaluedollarcnt':'tax_land',
                    'structuretaxvaluedollarcnt':'tax_structure',
                    'calculatedfinishedsquarefeet':'sqft',
                    'taxvaluedollarcnt':'taxvalue',
                    'fips':'county'})
    
    make_ints = ['bedrooms','sqft','taxvalue','yearbuilt']

    for col in make_ints:
        df[col] = df[col].astype(int)
        
    df.county = df.county.map({6037:'LA',6059:'Orange',6111:'Ventura'})
    
    return df


def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    
    # Create train_validate and test datasets
    train, validate_test = train_test_split(df, train_size=0.60, random_state=123)
    
    # Create train and validate datsets
    validate, test = train_test_split(validate_test, test_size=0.5, random_state=123)

    # Take a look at your split datasets

    print(f"""
    train -> {train.shape}
    validate -> {validate.shape}
    test -> {test.shape}""")
    
    return train, validate, test


def preprocess_zillow(df):
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


