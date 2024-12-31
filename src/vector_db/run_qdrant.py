# set working directory to LyricChat repo root
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "../.."))
os.chdir(repo_root)
sys.path.append(repo_root)

# load pkg & self-written modules:
import src.vector_db.qdrant_db as qd
import src.utils as utils
import logging
import pandas as pd
from dotenv import dotenv_values

ENV_VAR = dotenv_values(".env")


def main():
    # setup
    filepath = "data/NEFFEX_2024_09_19_23_07_06/data_for_DB.parquet"
    collection_name = "NEFFEX"
    qdrant_url = "https://94ddfbca-50be-4fb8-8791-ed716c146a08.europe-west3-0.gcp.cloud.qdrant.io:6333"
    df = pd.read_parquet(filepath)
    utils.set_loggings("info")  # set logging level

    # upload to Qdrant:
    db = qd.QdrantVecDB(device="cuda", url_db=qdrant_url)  # load cutomized OOP
    db.create(collection_name, recreate=True)  # create collection
    db.update(collection_name, df)  # upload data
    db.index(  # set payload schema for efficient filtering
        collection_name,
        {
            "metadata.primary_emotion": "keyword",
            "metadata.supporting_emotion": "keyword",
        },
    )
    logging.info(
        f"""Upload to Qdrant completed! Collection name: {collection_name}, Number of samples: {df.shape[0]}\n
Link to Database: {qdrant_url}/dashboard#/collections/{collection_name}
"""
    )


if __name__ == "__main__":
    main()
