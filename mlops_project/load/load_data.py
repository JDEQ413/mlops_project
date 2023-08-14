import os
import shutil
from pathlib import Path

import opendatasets as od


class DataRetriever():
    """
    A class to retrieve data from Kaggle specifically.

    Parameters:
        paths_list (list of str): Requires a list of strings with paths to download and store files addecuately.
            [0] = MAIN_DIR
            [1] = DATASETS_DIR
            [2] = KAGGLE_URL
            [3] = KAGGLE_LOCAL_DIR
            [4] = DATA_RETRIEVED

    Attributes:
        MAIN_DIR (str): 1st element of 'paths_list'. Main (root) directory of the project.
        DATASETS_DIR (str): 1st element of 'paths_list'. Directory where data retrieved is going to be stored.
        KAGGLE_URL (str): 2nd element of 'paths_list'. URL from which dataset is going to be downloaded.
        KAGGLE_LOCAL_DIR (str): 3rd element of 'paths_list'. Folder extracted from KAGGLE_URL, where dataset is dowonloaded by default.
        DATA_RETRIEVED: 4th element of 'paths_list'. Name for the final retrieved dataset.

    """

    def __init__(self, paths_list: list) -> None:
        self.MAIN_DIR = paths_list[0]
        self.DATASETS_DIR = paths_list[1]
        self.KAGGLE_URL = paths_list[2]
        self.KAGGLE_LOCAL_DIR = paths_list[3]
        self.DATA_RETRIEVED = paths_list[4]
        print("MAIN_DIR:" + self.MAIN_DIR)
        print("DATASETS_DIR:" + self.DATASETS_DIR)
        print("KAGGLE_URL:" + self.KAGGLE_URL)
        print("KAGGLE_LOCAL_DIR:" + self.KAGGLE_LOCAL_DIR)
        print("DATA_RETRIEVED:" + self.DATA_RETRIEVED)
        print()

    def retrieve_data(self):
        # Downloads dataset from kaggle with pre-defined structure (folder)
        od.download(self.KAGGLE_URL, force=True)

        # Finds the recently downloaded file
        if not os.path.isdir("/" + self.KAGGLE_LOCAL_DIR + "/"):
            paths = sorted(Path(self.KAGGLE_LOCAL_DIR + "/").iterdir(), key=os.path.getmtime)
        else:
            print("Directory could not be found: " + self.KAGGLE_LOCAL_DIR)
            return
        path_new_file = str(paths[-1])
        name_new_file = str(path_new_file).split('\\')[-1]
        path_new_file = "./" + str(path_new_file).split('\\')[0] + "/" + str(path_new_file).split('\\')[-1]
        print("Dataset downloaded: " + path_new_file)

        # Moves the file to default data directory instead of downloaded folder
        if os.path.isfile(self.MAIN_DIR + self.DATASETS_DIR + name_new_file):            # Searches for the new file downloaded inside default data directory
            os.remove(self.MAIN_DIR + self.DATASETS_DIR + name_new_file)                 # ,and deletes it
        if os.path.isfile(self.MAIN_DIR + self.DATASETS_DIR + self.DATA_RETRIEVED):           # Searches for any old file with FILE_NAME specified
            os.remove(self.MAIN_DIR + self.DATASETS_DIR + self.DATA_RETRIEVED)                # ,and deletes it too
        os.rename(path_new_file, self.MAIN_DIR + self.DATASETS_DIR + self.DATA_RETRIEVED)     # Finally, moves downloaded file to default datasets folder
        print("And stored in: " + self.MAIN_DIR + self.DATASETS_DIR + self.DATA_RETRIEVED)
        shutil.rmtree(self.KAGGLE_LOCAL_DIR)

        print()

# Usage example:
