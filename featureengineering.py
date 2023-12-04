import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from datavisualization import data_visualization
import numpy as np 

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



def featureEngineering():
    df = data_visualization()
    
        #outlier treatment
    df = outlierTreatment(df)

    #delete this before uploading
    col = [df.columns[0],'Year','hg/ha_yield']

    #labels='hg/ha_yield'
    Y = df['hg/ha_yield']
    X_ = df.drop(col, axis=1)

    # extracting categorical columns
    cat_df = X_.select_dtypes(include = ['object'])
    num_df = X_.select_dtypes(include=["number"])

    lb = LabelEncoder()
    cat_df = cat_df.apply(LabelEncoder().fit_transform)

    scaler = MinMaxScaler()

    scaler.fit(num_df)
    scaled = scaler.fit_transform(num_df)
    scaled_df = pd.DataFrame(scaled, columns=num_df.columns)

    X = pd.concat([cat_df,scaled_df,Y], axis = 1)
    X.to_csv("yield_final_df.csv", index = False)
    return X

featureEngineering()
