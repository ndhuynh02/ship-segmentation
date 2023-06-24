from torch.utils.data import Dataset
import os
import kaggle
from pathlib import Path
import zipfile

class AirbusDataset(Dataset):
    def prepare_data(self):
        # Specify the competition and folder names
        competition = 'airbus-ship-detection'
        folder = 'train_v2'

        # Create the folder to save the files
        save_folder = 'data/airbus'
        os.makedirs(save_folder, exist_ok=True)

        # Download the list of files in the folder
        files = kaggle.api.competition_list_files(competition)

        # Filter the files in the specified folder
        folder_files = [str(file) for file in files if str(file).startswith(folder)]

        # Download each file in the folder
        for file in folder_files:
            file_path = os.path.join(save_folder, folder)   
            kaggle.api.competition_download_file(competition, file, path=file_path, quiet=True)

        label_file = 'train_ship_segmentations_v2.csv'
        kaggle.api.competition_download_file(competition, label_file, 
                                             path=save_folder, quiet=True)
        label_zip = os.path.join(save_folder, label_file + ".zip")
        with zipfile.ZipFile(label_zip,"r") as zip_ref:
            zip_ref.extractall(save_folder)
        os.remove(label_zip)
            

if __name__ == "__main__":
    airbus = AirbusDataset()
    airbus.prepare_data()