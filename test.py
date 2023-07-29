import os
import pandas as pd
full_dataframe = pd.read_csv(os.path.join('data/airbus', 'train_ship_segmentations_v2.csv'))
print(len(full_dataframe))
good_dataframe = pd.read_csv(os.path.join('good_images.csv'))
bad_dataframe = pd.read_csv(os.path.join('bad_images.csv'))

common_image_ids_1 = set(full_dataframe['ImageId']).intersection(good_dataframe['id'])
common_image_ids_2 = set(full_dataframe['ImageId']).intersection(bad_dataframe['id'])

new_df = full_dataframe[~full_dataframe['ImageId'].isin(common_image_ids_1)]
new_df = new_df[~new_df['ImageId'].isin(common_image_ids_2)]

new_csv_file = 'new_train_ship_segmentations.csv'
new_df.to_csv(new_csv_file, index=False)

print(len(new_df))
