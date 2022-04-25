import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

# Importing the data
df_dir = './Final_Dataset_new.xlsx'
df = pd.read_excel(df_dir)

list_of_series = df.ID_Series.unique()

# Plot out just some of the time series
#list_of_series = list_of_series[1:5]

cont = 101
for time_series_id in list_of_series:
    # Select a single time series from the input dataset
    temp_df = df[df["ID_Series"] == time_series_id]
    
    # Assignment of plot values
    x = temp_df["Month"] 
    y = temp_df["Value"]
    year = str(int(temp_df["Year"].iloc[0]))
    geo = temp_df["Geo"].iloc[0]
    title = temp_df["About"].iloc[0]
    unit_of_measure = temp_df["UOM"].iloc[0]
    scale = temp_df["Scalar_Factor"].iloc[0]
    
    # Plot configurations settings
    my_dpi = 100
    fig = plt.figure(figsize=(1000/my_dpi, 600/my_dpi), dpi=my_dpi)
    ax = fig.add_subplot(1, 1, 1)
    plt.xticks(np.arange(0, 12, 1))
    plt.xlabel("Month")
    if scale.strip() == "units":
        plt.ylabel(unit_of_measure)
    else:
        plt.ylabel(scale.strip().capitalize()  + " " + unit_of_measure.lower())
    plt.suptitle(title, y = 0.97, size=13)
    plt.title("- " + geo + " " + year + " -", y = 1.01, size=11)
    plt.grid(linestyle="-.", linewidth=0.5)

    # Plot displaying
    plt.plot(x, y, '-o')
    #plt.savefig("Final dataset plots single/" + title + "_" + year + ".png")
    plt.savefig("new dataset plot images/" + str(cont) + ".png")
    #plt.show()
    plt.close('all')
    cont = cont + 1