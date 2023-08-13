import os
import shutil
from pathlib import Path

import opendatasets as od


class DataRetriever():
    """
    A class to retrieve data from Kaggle specifically.

    Parameters:
        paths_list (list of str): Requires a list of strings with paths to download and store files addecuately.
            [0] = DATASETS_DIR
            [1] = KAGGLE_URL
            [2] = KAGGLE_LOCAL_DIR
            [3] = DATA_RETRIEVED

    Attributes:
        DATASETS_DIR (str): 1st element of 'paths_list'. Directory where data retrieved is going to be stored.
        KAGGLE_URL (str): 2nd element of 'paths_list'. URL from which dataset is going to be downloaded.
        KAGGLE_LOCAL_DIR (str): 3rd element of 'paths_list'. Folder extracted from KAGGLE_URL, where dataset is dowonloaded by default.
        DATA_RETRIEVED: 4th element of 'paths_list'. Name for the final retrieved dataset.

    """

    def __init__(self, paths_list: list) -> None:
        self.DATASETS_DIR = paths_list[0]
        self.KAGGLE_URL = paths_list[1]
        self.DATASETS_DIR = paths_list[2]
        self.KAGGLE_LOCAL_DIR = paths_list[3]

    def retrieve_data(self):
        # Downloads dataset from kaggle with pre-defined structure (folder)
        od.download(self.KAGGLE_URL, force=True)

        # Finds the recently downloaded file
        paths = sorted(Path(self.KAGGLE_LOCAL_DIR).iterdir(), key=os.path.getmtime)
        path_new_file = str(paths[-1])
        name_new_file = str(path_new_file).split('\\')[-1]

        # If recently downloaded file already exists in root, delete it
        if os.path.isfile(path_new_file):
            print("Dataset downloaded: " + path_new_file)
        else:
            print("Something went wrong, dataset not downloades!")

        # Moves the file to root instead of downloaded folder
        if os.path.isfile(self.DATASETS_DIR + name_new_file):            # Searches for the new file downloaded
            os.remove(self.DATASETS_DIR + name_new_file)                 # ,and deletes it
        if os.path.isfile(self.DATASETS_DIR + self.DATA_RETRIEVED):           # Searches for any old file with FILE_NAME specified
            os.remove(self.DATASETS_DIR + self.DATA_RETRIEVED)                # ,and deletes it too
        os.rename(path_new_file, self.DATASETS_DIR + self.DATA_RETRIEVED)     # Finally, moves downloaded file to default datasets folder
        print("And stored in: " + self.DATASETS_DIR + self.DATA_RETRIEVED)
        shutil.rmtree(self.KAGGLE_LOCAL_DIR)                         # Deletes the folder where kaggle library downloaded dataset

# Usage example:
