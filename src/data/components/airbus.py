from torch.utils.data import Dataset

import json

f = open('kaggle.json')
data = json.load(f)

import os
os.environ['KAGGLE_USERNAME'] = data['username']
os.environ['KAGGLE_KEY'] = data['key']

from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import shutil
import glob
import pandas as pd

class AirbusDataset(Dataset):
    def __init__(self, data_dir:str = 'data/airbus') -> None:
        super().__init__()

        self.data_dir = data_dir
        self.prepare_data()

        self.filenames = glob.glob(os.path.join(self.data_dir, 'train_v2', "*.jpg"))

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        input = self.filenames[index]
        file_id = input.split(" ")[-1]

        df = pd.read_csv(os.path.join(self.data_dir, 'train_ship_segmentations_v2.csv'))
        output = df[df['ImageId'] == file_id]['EncodedPixels'].values.item()

        return input, output


    def prepare_data(self):
        data_path = os.path.join(self.data_dir, 'train_v2')
        if (os.path.exists(data_path)):
            if (len(os.listdir(data_path)) == 192556):
                print("Data is downloaded")
                return
            else:
                shutil.rmtree(self.data_dir)

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
            

if __name__ == "__main__":
    airbus = AirbusDataset()
    airbus.prepare_data()