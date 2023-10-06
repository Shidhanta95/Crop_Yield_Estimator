import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from data_preprocessing import dataPrepocessing

def featureEngineering():
    df = dataPrepocessing()

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
    X.to_csv("yield_df.csv", index = False)
    return X

featureEngineering()