import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_preprocessing import dataPrepocessing
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import plotly.express as px
from IPython.display import Image
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.io as pio
import io
from PIL import Image

# def visualisation():
#     df = featureEngineering()

#     # correlated features
#     datacorr=df.copy()
#     categorical_columns = datacorr.select_dtypes(include=['object']).columns.tolist()
#     label_encoder = LabelEncoder()
#     for column in categorical_columns:
#         datacorr[column] = label_encoder.fit_transform(datacorr[column])
#     sns.heatmap(datacorr.corr() , annot= True , cmap='PuOr')
#     plt.show()

#     #pairplot 
#     sns.pairplot(data=df,hue='Item',kind='scatter')
#     plt.show()

#     #area based plot
#     # palette = sns.color_palette('tab20', 21,as_cmap=True)
#     # num_plots = 7
#     # areas_per_plot = 10

#     # # Get unique areas 
#     # unique_areas = sorted(df['Area'].unique())

#     # # Split into chunks
#     # area_chunks = [unique_areas[i:i+areas_per_plot] for i in range(0, len(unique_areas), areas_per_plot)]
#     # area_chunks[-2] = unique_areas[-11:] 
#     # # fig, axs = plt.subplots(ncols=num_plots, figsize=(30, 10))
#     # j=0
#     # for i, ax in enumerate(axs):

#     #     plot_df = df[df['Area'].isin(area_chunks[i])]
#     #     for i, area in enumerate(plot_df['Area'].unique()):
#     #         data = plot_df[plot_df['Area'] == area]
#     #         ax.hist(data['hg/ha_yield'], facecolor=palette(i), label=area)
#     #     ax.legend()
#     #     j+=1
#     #     plt.show()

#     # for i in range(0,7):
#     #     plot_df = df[df['Area'].isin(area_chunks[i])]
#     #     plot_df.groupby(['Area'])['average_rain_fall_mm_per_year'].mean().plot(kind='bar',rot=0)
#     #     plt.xticks(rotation=90)
#     #     plt.show()

#     a_dims = (16.7, 8.27)
#     fig, ax = plt.subplots(figsize=a_dims)
#     sns.boxplot(x="Item",y="hg/ha_yield",palette="BrBG",data=df,ax=ax)
#     plt.show()



# visualisation()



# a =[]
def data_visualization():
    count = 0
    data = dataPrepocessing()
    col=list(data.columns)
    for i in col:
        count += 1
        fig = px.box(data, y=i)
        fig.update_layout(template='plotly_dark')
        #fig.update_layout(plot_bgcolor = "plotly_dark")
        fig.update_xaxes(showgrid=False,zeroline=False)
        fig.update_yaxes(showgrid=False,zeroline=False)
        fig.write_image(f"{count}_box_{i}.jpg")
        # a.append(fig)
    for i in col:
        count += 1
        fig = ff.create_distplot(int([data[i].values]),group_labels=[i])
        fig.update_layout(template='plotly_dark')
        #fig.update_layout(plot_bgcolor = "plotly_dark")
        fig.update_xaxes(showgrid=False,zeroline=False)
        fig.update_yaxes(showgrid=False,zeroline=False)
        fig.write_image(f"{count}_dist_{i}.jpg")
        # a.append(fig)
    y=data.corr().columns.tolist()
    z=data.corr().values.tolist()
    z_text = np.around(z, decimals=4) # Only show rounded value (full value on hover)
    fig = ff.create_annotated_heatmap(z,x=y,y=y,annotation_text=z_text,colorscale=px.colors.sequential.Cividis_r,showscale=True)
    fig.update_layout(template='plotly_dark')
    count += 1
    fig.write_image(f"{count}_heatmap.jpg")
    # a.append(fig)




    # figures = a
    # image_list = [pio.to_image(fig, format='png', width=1440, height=900, scale=1.5) for fig in figures]
    # for index, image in enumerate(image_list):
    #     with io.BytesIO() as tmp:
    #         tmp.write(image)  # write the image bytes to the io.BytesIO() temporary object
    #         image = Image.open(tmp).convert('RGB')  # convert and overwrite 'image' to prevent creating a new variable
    #         image_list[index] = image  # overwrite byte image data in list, replace with PIL converted image data

    # # pop first item from image_list, use that to access .save(). Then refer back to image_list to append the rest
    # image_list.pop(0).save(r'./Student Performance Prediction#587.pdf', 'PDF',
    #                     save_all=True, append_images=image_list, resolution=100.0)  # TODO improve resolution
    
    return data

data_visualization()




