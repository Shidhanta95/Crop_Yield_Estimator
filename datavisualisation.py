import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from featureengineering import featureEngineering
from sklearn.preprocessing import LabelEncoder

def visualisation():
    df = featureEngineering()

    # correlated features
    datacorr=df.copy()
    categorical_columns = datacorr.select_dtypes(include=['object']).columns.tolist()
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        datacorr[column] = label_encoder.fit_transform(datacorr[column])
    sns.heatmap(datacorr.corr() , annot= True , cmap='PuOr')
    plt.show()

    #pairplot 
    sns.pairplot(data=df,hue='Item',kind='scatter')
    plt.show()

    #area based plot
    # palette = sns.color_palette('tab20', 21,as_cmap=True)
    # num_plots = 7
    # areas_per_plot = 10

    # # Get unique areas 
    # unique_areas = sorted(df['Area'].unique())

    # # Split into chunks
    # area_chunks = [unique_areas[i:i+areas_per_plot] for i in range(0, len(unique_areas), areas_per_plot)]
    # area_chunks[-2] = unique_areas[-11:] 
    # # fig, axs = plt.subplots(ncols=num_plots, figsize=(30, 10))
    # j=0
    # for i, ax in enumerate(axs):

    #     plot_df = df[df['Area'].isin(area_chunks[i])]
    #     for i, area in enumerate(plot_df['Area'].unique()):
    #         data = plot_df[plot_df['Area'] == area]
    #         ax.hist(data['hg/ha_yield'], facecolor=palette(i), label=area)
    #     ax.legend()
    #     j+=1
    #     plt.show()

    # for i in range(0,7):
    #     plot_df = df[df['Area'].isin(area_chunks[i])]
    #     plot_df.groupby(['Area'])['average_rain_fall_mm_per_year'].mean().plot(kind='bar',rot=0)
    #     plt.xticks(rotation=90)
    #     plt.show()

    a_dims = (16.7, 8.27)
    fig, ax = plt.subplots(figsize=a_dims)
    sns.boxplot(x="Item",y="hg/ha_yield",palette="BrBG",data=df,ax=ax)
    plt.show()



visualisation()







