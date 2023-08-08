import json

f = open("kaggle.json")
data = json.load(f)

import os

os.environ["KAGGLE_USERNAME"] = data["username"]
os.environ["KAGGLE_KEY"] = data["key"]

import shutil
import zipfile

import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
from src.utils.airbus_utils import imshow, masks_as_image


class AirbusDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/airbus",
        undersample: int = 140000,
        subset: int = 10000,
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.prepare_data()

        masks = pd.read_csv(os.path.join(self.data_dir, "train_ship_segmentations_v2.csv"))

        # undersample non-ship images
        if undersample == -1:
            self.dataframe = masks.dropna(subset=["EncodedPixels"])  # Drop all non-ship images
        elif undersample > 0:
            self.dataframe = masks.drop(
                masks[masks.EncodedPixels.isnull()].sample(undersample, random_state=42).index
            )
        else:
            self.dataframe = masks

        image_ids = self.dataframe["ImageId"].unique()

        # use subset of data
        if subset != 0:
            dataframe_subset = (
                self.dataframe.groupby("ImageId").size().reset_index(name="counts")
            )  # cols: ImageId & counts
            image_ids_subset, _ = train_test_split(
                dataframe_subset,
                train_size=subset / len(image_ids),
                stratify=dataframe_subset["counts"],
                shuffle=True,
                random_state=42,
            )
            image_ids = image_ids_subset["ImageId"]
            self.dataframe = self.dataframe[self.dataframe["ImageId"].isin(image_ids)].reset_index(
                drop=True
            )

        self.filenames = [
            os.path.join(self.data_dir, "train_v2", image_id) for image_id in image_ids
        ]
        assert (
            len(self.filenames) == self.dataframe["ImageId"].nunique()
        ), "The number of filenames does not match the number of unique ImageIds"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image = self.filenames[index]
        file_id = image.split("/")[-1]

        mask = self.dataframe[self.dataframe["ImageId"] == file_id]["EncodedPixels"]

        image = Image.open(image).convert("RGB")
        mask = masks_as_image(mask)
        label = 0 if mask.sum() == 0 else 1

        return np.array(image, dtype=np.uint8), mask, label, file_id

    def prepare_data(self):
        data_path = os.path.join(self.data_dir, "train_v2")
        if os.path.exists(data_path):
            print("Data is downloaded")
            return

        api = KaggleApi()
        api.authenticate()

        # Specify the competition and folder names
        competition = "airbus-ship-detection"

        # Create the folder to save the files
        os.makedirs(self.data_dir, exist_ok=True)

        print("Downloading data")
        api.competition_download_files(competition, path=self.data_dir, quiet=False)

        downloaded_file = os.path.join(self.data_dir, "airbus-ship-detection.zip")

        print("Extracting ...")
        with zipfile.ZipFile(downloaded_file, "r") as zip_ref:
            zip_ref.extractall(self.data_dir)

        print("Removing unnecessary files and folders")
        os.remove(downloaded_file)  # delete zip file
        os.remove(
            os.path.join(self.data_dir, "sample_submission_v2.csv")
        )  # delete sample submissiong
        shutil.rmtree(
            os.path.join(self.data_dir, "test_v2")
        )  # delete test data (cuz labels are not provided)

        print("Done!")


if __name__ == "__main__":
    airbus = AirbusDataset()
    img, mask, label, img_id = airbus[2]
    print(img.shape)
    print(mask.shape)
    print(label)
    print(img_id)
    imshow(img, mask)
