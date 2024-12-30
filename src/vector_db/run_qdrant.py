# set working directory to LyricChat repo root
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "../.."))
os.chdir(repo_root)
sys.path.append(repo_root)

# load self-written functions:
import src.vector_db.qdrant_db as DB
import src.utils as utils

#
import json
import pandas as pd
from importlib import reload

reload(DB)


def main():
    # load data:
    filepath = "data/NEFFEX_2024_09_19_23_07_06/data_for_DB.parquet"
    df = pd.read_parquet(filepath)

    # upload to Qdrant:
    client = DB.QdrantVecDB(device="cuda")  # load cutomized OOP
    client.create("NEFFEX", recreate=True)  # create collection
    client.update("NEFFEX", df)  # upload data


if __name__ == "__main__":
    main()
