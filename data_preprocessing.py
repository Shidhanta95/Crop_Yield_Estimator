import pandas as pd
import numpy as np
from data_analysis import dataAnalysis


def outlierTreatment(df):
    # Outlier Treatment 
    # IQR
    df.reset_index(drop=True)

    # Create arrays of Boolean values indicating the outlier rows
    # upper_array = np.where(df[col]>=upper)[0]
    # lower_array = np.where(df[col]<=lower)[0]
    
    # # Removing the outliers
    # df.drop(index=upper_array, inplace=True)
    # df.drop(index=lower_array, inplace=True)
    num_types = df.select_dtypes("number").columns
    # Calculate the upper and lower limits
    for col in num_types:    
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR
    
  
    
    df[col] = np.where(df[col] > upper, upper,df[col])
    df[col] = np.where(df[col]<lower, lower, df[col])
    return df


def dataPrepocessing():
    df = dataAnalysis()

    x = df.columns[0]
    #drop column
    df.drop(x,inplace=True, axis=1)
    
    # remove countries with less than 100 record
    country_counts =df['Area'].value_counts()
    countries_to_drop = country_counts[country_counts < 100].index.tolist()
    df_filtered = df[~df['Area'].isin(countries_to_drop)]
    df = df_filtered.reset_index(drop=True)

    # filling null values 
    names = df[df.columns[df.isna().any()]].columns
    for name in names:
        df[name].fillna(df[name].mode(), inplace = True)

    return df


dataPrepocessing()

