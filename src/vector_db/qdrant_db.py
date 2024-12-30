# env
# pip install qdrant-client

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import Literal
from sentence_transformers import SentenceTransformer
import uuid


class QdrantVecDB:
    def __init__(
        self,
        model="models/BAAI_bge-small-en-v1.5",
        device="cpu",
        url_db="http://localhost:6333",
    ):
        """
        TODO: customized Qdrant interface for LyricChat, includes CRUD operations
        """
        # setup args:
        self.model = model
        self.device = device
        self.url_db = url_db

        # setup model & DB client:
        self.load_model()  # load embedding model
        self.client = QdrantClient(url=self.url_db)

    def load_model(self, model=None):
        """TODO: load the sentence transformer model & get the embedding dimension"""
        # change to new model if provided:
        if model is not None:
            self.model = model
        # load model:
        self.embedding_model = SentenceTransformer(self.model)
        self.embedding_model.to(self.device)  # move to GPU if available

        # get embedding dimension:
        temp_output = self.embedding_model.encode("", batch_size=1)
        self.embedding_dimension = temp_output.shape[0]

    def create(self, collection_name, recreate=False):
        if recreate:  # delete if recreate = True
            self.client.delete_collection(collection_name=collection_name)
        if not self.client.collection_exists(collection_name):  # create if not exists
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension, distance=Distance.COSINE
                ),
            )

    def update(self, collection_name, data, batch_size=8):
        """
        TODO: update the collection with new data
        Args:
            collection_name: name of the collection
            data: a dataframe with `input` column as embedding input & row index as id (uuid)
            batch_size: batch size for encoding
        """
        # sanity check:
        assert isinstance(data, pd.DataFrame), "data must be a dataframe"
        assert "input" in data.columns, "dataframe must contain `input` column"
        assert (
            data.index.to_series()
            .apply(lambda x: isinstance(x, str) and len(x) == 32)
            .all()
        ), "row index must be uuid"
        # prepare input:
        self.create(collection_name, recreate=False)  # create collection if not exist
        data = self.encode(data, batch_size=batch_size)  # get vector embeddings

        # update collection:
        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=id,  # uuid
                    vector=row["embedding"],  # embedding from self.encode()
                    payload={
                        "text": row["input"],  # text for RAG to return
                        "metadata": row.drop(  # remaining features
                            ["embedding", "input"]
                        ).to_dict(),
                    },
                )
                for id, row in data.iterrows()
            ],
        )

    def encode(self, data, target="input", batch_size=8):
        """
        TODO: Helper for update(); encode data[target] and update data with new column 'embedding'
        Args:
            data: dataframe with `target` column to encode
            target: column name to encode
            batch_size: batch size for encoding
        """
        embeddings = self.embedding_model.encode(
            data[target].tolist(), batch_size=batch_size
        )
        data["embedding"] = embeddings.tolist()
        return data
