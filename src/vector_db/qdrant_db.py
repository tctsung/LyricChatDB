# env
# pip install qdrant-client pyarrow sentence-transformers[onnx-gpu]

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny  # filter
import types  # for generator type
from typing import Literal
from fastembed import TextEmbedding
import uuid
from dotenv import dotenv_values
import logging
import src.utils as utils
from tqdm import tqdm

ENV_VAR = dotenv_values(".env")


class QdrantVecDB:
    def __init__(
        self,
        model="BAAI/bge-small-en-v1.5",
        url_db="http://localhost:6333",
        api_key=None,
        logging_level="info",
    ):
        """
        TODO: customized Qdrant interface for LyricChat, includes CRUD operations
        """
        # setup args:
        self.model = model
        self.url_db = url_db
        if api_key is None:  # get from .env if not provided
            api_key = ENV_VAR.get("Qdrant_API_KEY", None)
        utils.set_loggings(logging_level, func_name="QdrantVecDB")
        # setup model & DB client:
        self.load_model()  # load embedding model
        self.client = QdrantClient(url=self.url_db, api_key=api_key)

    def load_model(self, model=None):
        """TODO: load the sentence transformer model & get the embedding dimension"""
        # change to new model if provided:
        if model is not None:
            self.model = model
        # load model (cache at models so no need to reload after first time)
        self.embedding_model = TextEmbedding(model_name=self.model, cache_dir="models/")

        # get embedding dimension:
        temp_output = self.embedding_model.embed("")
        self.embedding_dimension = len(list(temp_output)[0])
        logging.info(
            f"Embedding Model {self.model} load successfully, output dimension is {self.embedding_dimension}"
        )

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

    def update(self, collection_name, data, target="input", batch_size=256):
        """
        TODO: update the collection with new data
        Args:
            collection_name: name of the collection
            data: a dataframe with `target` column as embedding input & row index as id (uuid)
            batch_size: batch size for encoding
        """
        # sanity check:
        assert isinstance(data, pd.DataFrame), "data must be a dataframe"
        assert (
            target in data.columns
        ), f"dataframe must contain `{target}` column, or modify arg `target`"
        assert (
            data.index.to_series()
            .apply(lambda x: isinstance(x, str) and len(x) == 32)
            .all()
        ), "row index must be uuid"

        self.create(collection_name, recreate=False)  # create collection if not exist

        # batch upload to Qdrant
        for start_idx in tqdm(range(0, len(data), batch_size)):
            batch_df = data.iloc[start_idx : start_idx + batch_size].copy()
            batch_df_w_embeddings = self._encode(batch_df, target=target)
            batch_points = [
                PointStruct(
                    id=id,
                    vector=row["embedding"].tolist(),
                    payload={
                        "text": row[target],
                        "metadata": row.drop(["embedding", target]).to_dict(),
                    },
                )
                for id, row in batch_df_w_embeddings.iterrows()
            ]
            self.client.upsert(collection_name=collection_name, points=batch_points)

    def _encode(self, data, target="input", batch_size=256):
        """
        TODO: Helper for update(); encode data[target] and update data with new column 'embedding'
        Args:
            data: dataframe with `target` column to encode
            target: column name to encode
            batch_size: batch size for encoding
        """
        embeddings = list(
            self.embedding_model.embed(data[target].tolist(), batch_size=batch_size)
        )
        data["embedding"] = embeddings  # numpy array
        return data

    def index(self, collection_name, feature_dict: dict[str, PayloadSchemaType]):
        """
        TODO: index specific payload features to increase filtering speed
        Args:
            collection_name: name of the collection
            feature_dict: dictionary with field names as keys and schemas as values
        Eg: index("NEFFEX", {"metadata.primary_emotion": "keyword", "metadata.supporting_emotion": "keyword"})
        """
        for field_name, schema in feature_dict.items():
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema,
            )

    def read(
        self,
        collection_name,
        query: str,
        limit: int = 5,
        should_conditions: dict = None,
        must_conditions: dict = None,
    ):
        """
        TODO: similarity search with query
        """
        # create filter:
        filter = self._create_qdrant_filter(should_conditions, must_conditions)
        generator_output = self.embedding_model.embed(query)
        query_vector = list(generator_output)[0].tolist()
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            with_payload=True,
            limit=limit,
            query_filter=filter,
        )
        return [result.payload for result in search_result]

    def _create_qdrant_filter(self, should_conditions=None, must_conditions=None):
        """
        TODO: Helper for read(); create Qdrant filter from should & must conditions
        Eg. should_conditions = {"metadata.primary_emotion": "happy", "metadata.supporting_emotion": ["sad", "angry"]}
        """

        def create_field_condition(key, value):
            # use MatchAny for list of values, MatchValue for single value
            if isinstance(value, (list, tuple, types.GeneratorType)):
                return FieldCondition(key=key, match=MatchAny(any=value))
            else:
                return FieldCondition(key=key, match=MatchValue(value=value))

        should_filters, must_filters = [], []
        if should_conditions:
            should_filters = [
                create_field_condition(key, value)
                for key, value in should_conditions.items()
            ]

        if must_conditions:
            must_filters = [
                create_field_condition(key, value)
                for key, value in must_conditions.items()
            ]

        return Filter(should=should_filters, must=must_filters)

    # def read_rerank(self, collection_name, query: str, limit: int = 5):
    #     """TODO: Rerank search results"""
