import pandas as pd

def loadData():
    df=pd.read_csv("dataset/yield_df.csv")
    return df

loadData()