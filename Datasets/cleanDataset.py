import pandas as pd
import numpy as np
import csv

# Importing the dataset about the collected captions and about the generated line charts
v1_captions_collection = pd.read_excel("preprocessed_new_captions_collection.xlsx")

IDs = np.unique(v1_captions_collection.ID_Series)

# Combining information about captions and about line charts in the same dataset.
for ID in IDs:
    temp_ID_captions_datasets = v1_captions_collection[v1_captions_collection.ID_Series == ID]
    for idx, temp_dataset in temp_ID_captions_datasets.iterrows():
        # write captions to folder
        folder_caption = "D:/study/CPSC449/Transformer_timeseries/Datasets/caption_old/"
        filename_caption_txt = folder_caption + str(temp_dataset.ID_Caption) + ".txt"
        file_caption = open(filename_caption_txt, "w", encoding='utf-8')
        try:
            file_caption.write(temp_dataset.Caption + "\n")
        except:
            print("An exception occurred on the following sentence in caption: ")
            print(temp_dataset.ID_Caption)
        file_caption.close()
        # write titles to folder
        folder_title = "D:/study/CPSC449/Transformer_timeseries/Datasets/title_old/"
        filename_title_txt = folder_title + str(temp_dataset.ID_Caption) + ".txt"
        file_title = open(filename_title_txt, "w", encoding='utf-8')
        try:
            file_title.write(temp_dataset.About + " in " + temp_dataset.Geo + " " + str(temp_dataset.Year) + "\n")
        except:
            print("An exception occurred on the following sentence in title: ")
            print(temp_dataset.ID_Caption)
        file_title.close()
        # write data to folder
        folder_data = "D:/study/CPSC449/Transformer_timeseries/Datasets/data/"
        filename_data_csv = folder_data + str(temp_dataset.ID_Caption) + ".csv"
        # with open(filename_data_csv, 'w') as csvfile:
        #     writer = csv.writer(csvfile, delimiter='\t', lineterminator='\n', )
            # Add the data row
        data = []
        data.append(['January', temp_dataset.M1_Jan])
        data.append(['February', temp_dataset.M2_Feb])
        data.append(['March', temp_dataset.M3_Mar])
        data.append(['April', temp_dataset.M4_Apr])
        data.append(['May', temp_dataset.M5_May])
        data.append(['June', temp_dataset.M6_Jun])
        data.append(['July', temp_dataset.M7_Jul])
        data.append(['August', temp_dataset.M8_Aug])
        data.append(['September', temp_dataset.M9_Sep])
        data.append(['October', temp_dataset.M10_Oct])
        data.append(['November', temp_dataset.M11_Nov])
        data.append(['December', temp_dataset.M12_Dec])
        # Add the header row
        if temp_dataset.Scale.lower().strip() == "units":
            data_csv = pd.DataFrame(data, columns=['Month', temp_dataset.UOM])
        else:
            data_csv = pd.DataFrame(data, columns=['Month', temp_dataset.Scale[:-1].strip().capitalize() + " " + temp_dataset.UOM.lower()])
        data_csv.to_csv(filename_data_csv, line_terminator= '\n', encoding= 'utf-8', index=False)
