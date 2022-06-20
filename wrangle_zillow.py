import pandas as pd
import numpy as np
import os
from env import get_db_url
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

def get_zillow_data():
    '''Returns a dataframe of all single family residential properties from 2017. Initial 
    query is from the Codeup database. File saved as CSV and called upon after initial query.'''
    filename = "zillow.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        sql = '''SELECT *
        FROM properties_2017
        JOIN predictions_2017 USING (parcelid)
        LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
        LEFT JOIN propertylandusetype USING (propertylandusetypeid)
        LEFT JOIN airconditioningtype USING (airconditioningtypeid)
        LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
        LEFT JOIN storytype USING (storytypeid)
        LEFT JOIN unique_properties USING (parcelid)
        LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
        WHERE propertylandusetype.propertylandusedesc = 'Single Family Residential'
        AND predictions_2017.transactiondate LIKE '2017%%'
        AND properties_2017.latitude IS NOT NULL
        AND properties_2017.longitude IS NOT NULL;'''
        df = pd.read_sql(sql, get_db_url('zillow'))
        return df


def find_missing_values(df):
    '''Creates dataframe for missing values.'''
    column_name = []
    num_rows_missing = []
    pct_rows_missing = []

    for column in df.columns:       
        num_rows_missing.append(df[column].isna().sum())
        pct_rows_missing.append(df[column].isna().sum()/ len(df))
        column_name.append(column)
    data = {'column_name':column_name, 'num_rows_missing': num_rows_missing, 'pct_rows_missing': pct_rows_missing}
    return pd.DataFrame(data, index=None)

def create_features(df):
    ''' This function creates three new columns age, taxvalue_per_sqft, and month_of_sale and 
    adds them to the existing dataframe.'''
    df['age'] = 2017 - df.yearbuilt
    df['taxvalue_per_sqft'] = df.taxvaluedollarcnt / df.calculatedfinishedsquarefeet
    df['month_of_sale'] = pd.DatetimeIndex(df['transactiondate']).month
    return df


def handle_missing_values(df, prop_required_column, prop_required_row):
    '''This function removes null values from the entire dataframe if that percentage of nulls
    is above the given threshold.'''
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    # remove additional rows with remaining nulls
    df = df.dropna()
    return df


cols_to_remove = ['parcelid', 'propertylandusetypeid', 'id','calculatedbathnbr', 
    'finishedsquarefeet12', 'fullbathcnt', 'propertycountylandusecode',
    'rawcensustractandblock','regionidcounty', 'roomcnt', 'structuretaxvaluedollarcnt',
    'assessmentyear', 'landtaxvaluedollarcnt', 'censustractandblock', 'id', 'regionidzip', 'regionidcity','taxamount']

def remove_columns(df, cols_to_remove):
    '''This function removes columns(cols_to_remove) from the dataframe due to duplicates or
    erroneous data.'''
    df = df.drop(columns=cols_to_remove)
    return df

def map_counties(df):
    '''This function takes in the fips code and maps the county names.'''
    # identified counties for fips codes 
    counties = {6037: 'Los_Angeles',
                6059: 'Orange',
                6111: 'Ventura'}
    # map counties to fips codes
    df.fips = df.fips.map(counties)
    return df

def remove_outliers(df, k):
    ''' Take in a dataframe, k value, and specified columns within a dataframe 
    and then return the dataframe with outliers removed.
    '''
    columns=['bathroomcnt',
	'bedroomcnt',
	'calculatedfinishedsquarefeet',
	'lotsizesquarefeet',
	'taxvaluedollarcnt',
    'yearbuilt',
	'taxamount',
    'logerror',
    'age',
    'taxvalue_per_sqft']

    for col in columns:
        # Get quartiles
        q1, q3 = df[col].quantile([.25, .75]) 
        # Calculate interquartile range
        iqr = q3 - q1 
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df
def integers(df):
    """This function converts datatypes to integers for the given columns."""
    df['calculatedfinishedsquarefeet'] = df['calculatedfinishedsquarefeet'].astype(int)
    df['bedroomcnt'] = df['bedroomcnt'].astype(int)
    df['latitude'] = df['latitude'].astype(int)
    df['longitude'] = df['longitude'].astype(int)
    df['lotsizesquarefeet'] = df['lotsizesquarefeet'].astype(int)
    df['yearbuilt'] = df['yearbuilt'].astype(int)
    df['taxvaluedollarcnt'] = df['taxvaluedollarcnt'].astype(int)
    df['age'] = df['age'].astype(int)
    df['taxvalue_per_sqft'] = df['taxvalue_per_sqft'].astype(int)
    return df
    




def prepare_zillow_data(df):
    '''Puts together all of the previsouly created functions to prepare Zillow data.'''
    df = create_features(df)
    df = handle_missing_values(df,.7,.7)
    df = remove_outliers(df,k=1.5)
    df = map_counties(df)
    df = remove_columns(df,cols_to_remove)
    df = integers(df)
    return df





def split_zillow_data(df):
    ''' This function splits the cleaned dataframe into train, validate, and test 
    datasets.'''

    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123) 
                                   
    return train, validate, test

def scale_data(train,
              validate,
              test,
              columns_to_scale=['longitude', 'latitude','yearbuilt','calculatedfinishedsquarefeet','bathroomcnt',
              'bedroomcnt','lotsizesquarefeet','taxvaluedollarcnt', 'age', 'taxvalue_per_sqft', 'month_of_sale']):
    '''
    Scales the split data.
    Takes in train, validate and test data and returns the scaled data.
    '''
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    #using MinMaxScaler (best showing distribution once scaled)
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(train[columns_to_scale])
    
    #creating a df that puts MinMaxScaler to work on the wanted columns and returns the split datasets and counterparts
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                 columns=train[columns_to_scale].columns.values).set_index([train.index.values])
    
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                 columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    
    return train_scaled, validate_scaled, test_scaled
