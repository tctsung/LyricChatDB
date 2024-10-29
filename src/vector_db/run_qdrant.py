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
    # combine lyrics & summary, emotions into one df:
    folder_path = "data/NEFFEX_2024_09_19_23_07_06"
    with open(folder_path + "/lyrics_processed.json", "r") as f:
        lyrics = json.load(f)
    df_lyrics = pd.DataFrame.from_dict(
        lyrics, orient="index"
    )  # orient='index' means each key is a row
    df_lyrics.columns = ["lyrics"]
    df_features = utils.read_jsonl(folder_path + "/summary_and_emotions.json")
    df_features.set_index("key", inplace=True)
    # join two dataframe:
    df_combined = df_features.join(df_lyrics, how="inner")

    vec_embeddings = DB.VecEmbeddings(
        device="cuda", model="models/BAAI_bge-small-en-v1.5"
    )

    qdrant_db = DB.QdrantVecDB(
        collection_name="NEFFEX",
        vec_embeddings=vec_embeddings,
        url_db="http://localhost:6333",
    )


if __name__ == "__main__":
    main()
