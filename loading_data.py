import pandas as pd

def loadData():
    df=pd.read_csv("yield_df.csv")
    return df.iloc[:1000]

loadData()
