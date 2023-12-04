import pandas as pd

def loadData():
    df=pd.read_csv(r"dataset\yield_df.csv")
    return df

loadData()
