import json

f = open('kaggle.json')
data = json.load(f)

import os
os.environ['KAGGLE_USERNAME'] = data['username']
os.environ['KAGGLE_KEY'] = data['key']

from kaggle.api.kaggle_api_extended import KaggleApi
from torch.utils.data import Dataset
import zipfile
import shutil
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from src.utils.airbus_utils import masks_as_image, imshow

class AirbusDataset(Dataset):
    def __init__(self, data_dir:str = 'data/airbus', undersample:int = 1400, subset:int = 500) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.prepare_data()

        masks = pd.read_csv(os.path.join(self.data_dir, 'train_ship_segmentations_v2.csv'))

        # undersample non-ship images
        if undersample != 0:
            self.dataframe = masks.drop(masks[masks.EncodedPixels.isnull()].sample(undersample, random_state=42).index)
        else: 
            self.dataframe = masks

        image_ids = self.dataframe['ImageId'].unique()
        self.filenames = [os.path.join(self.data_dir, "train_v2", image_id) for image_id in image_ids]

        # use subset of data 
        if subset != 0:
            image_ids_subset = image_ids[:subset].tolist()
            self.filenames = [os.path.join(self.data_dir, "train_v2", image_id) for image_id in image_ids_subset]
            self.dataframe = self.dataframe[self.dataframe['ImageId'].isin(image_ids_subset)].reset_index(drop=True)
            assert len(self.filenames) == self.dataframe['ImageId'].nunique(), \
                "The number of filenames does not match the number of unique ImageIds"

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        image = self.filenames[index]
        file_id = image.split("/")[-1]

        mask = self.dataframe[self.dataframe['ImageId'] == file_id]['EncodedPixels']

        image = Image.open(image).convert('RGB')
        mask = masks_as_image(mask)

        return np.array(image, dtype=np.uint8), mask

    def prepare_data(self):
        data_path = os.path.join(self.data_dir, 'train_v2')
        if (os.path.exists(data_path)):
            print("Data is downloaded")
            return

        api = KaggleApi()
        api.authenticate()

        # Specify the competition and folder names
        competition = 'airbus-ship-detection'

        # Create the folder to save the files
        os.makedirs(self.data_dir, exist_ok=True)

        print("Downloading data")
        api.competition_download_files(competition, path = self.data_dir, quiet=False)

        downloaded_file = os.path.join(self.data_dir, 'airbus-ship-detection.zip')

        print("Extracting ...")
        with zipfile.ZipFile(downloaded_file,"r") as zip_ref:
            zip_ref.extractall(self.data_dir)

        print("Removing unneccessary files and folders")
        os.remove(downloaded_file)      # delete zip file
        os.remove(os.path.join(self.data_dir, 'sample_submission_v2.csv'))    # delete sample submissiong
        shutil.rmtree(os.path.join(self.data_dir, 'test_v2'))       # delete test data (cuz labels are not provided)
        
        print("Done!")

    # def rle_decode(self, mask_rle, shape=(768, 768)):
    #     '''
    #     mask_rle: run-length as string formated (start length)
    #     shape: (height,width) of array to return 
    #     Returns numpy array, 1 - mask, 0 - background
    #     '''
    #     s = mask_rle.split()
    #     starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    #     starts -= 1
    #     ends = starts + lengths
    #     img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    #     for lo, hi in zip(starts, ends):
    #         img[lo:hi] = 1
    #     return img.reshape(shape).T  # Needed to align to RLE direction

    # def masks_as_image(self, in_mask_list):
    #     # Take the individual ship masks and create a single mask array for all ships
    #     all_masks = np.zeros((768, 768), dtype = np.uint8)
    #     for mask in in_mask_list:
    #         if isinstance(mask, str):
    #             all_masks |= self.rle_decode(mask)
    #     return all_masks

    # @staticmethod
    # def mask_overlay(image, mask, color=(0, 1, 0)):
    #     """
    #     Helper function to visualize mask on the top of the image
    #     """
    #     mask = mask.squeeze() # mask could be (1, 768, 768) or (768, 768)
    #     mask = np.dstack((mask, mask, mask)) * np.array(color, dtype=np.uint8) * 255
    #     weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    #     img = image.copy()
    #     ind = mask[:, :, 1] > 0
    #     img[ind] = weighted_sum[ind]    
    #     return img

    # @staticmethod
    # def imshow(img, mask, title=None):
    #     fig = plt.figure(figsize = (6,6))
    #     plt.imshow(AirbusDataset.mask_overlay(img, mask))
    #     if title is not None:
    #         plt.title(title)
    #     plt.show()
            

if __name__ == "__main__":
    airbus = AirbusDataset()
    img, mask = airbus[2]
    print(img.shape)
    print(mask.shape)

    imshow(img, mask)
    