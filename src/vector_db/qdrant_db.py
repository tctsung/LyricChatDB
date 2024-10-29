# env
# pip install qdrant-client


from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import Literal
from sentence_transformers import SentenceTransformer
import uuid


class VecEmbeddings:
    def __init__(
        self, device: Literal["cuda", "cpu"], model="models/BAAI_bge-small-en-v1.5"
    ):
        """
        Args:
            device (Literal["cuda", "cpu"]): The device to run the sentence transformer model on.
            model (str, optional): The name of the sentence transformer model to use. Defaults to "models/BAAI_bge-small-en-v1.5".

        Attributes:
            model_name (str): The name of the sentence transformer model being used.
            embedding_model (SentenceTransformer): The sentence transformer model instance.
        """
        self.model_name = model
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # > disable because doesn't worth it to instasll & load pytorch for this
        self.embedding_model = SentenceTransformer(model)
        self.embedding_model.to(device)  # move to GPU if available

        # get embedding dimension
        self.get_dim()

    def encode(self, texts: list[str], batch_size=8):
        return self.embedding_model.encode(texts, batch_size=batch_size)

    def get_dim(self):
        temp_output = self.encode("", batch_size=1)
        self.word_embedding_dimension = temp_output.shape[0]


class QdrantVecDB:
    def __init__(
        self,
        collection_name,
        vec_embeddings,
        url_db="http://localhost:6333",
        recreate=False,
    ):
        self.collection_name = collection_name
        self.embedding_model = vec_embeddings  # from class VecEmbeddings
        self.word_embedding_dimension = vec_embeddings.word_embedding_dimension
        self.url_db = url_db
        self.client = QdrantClient(url=self.url_db)

        # create collection
        self.create_collection(recreate=recreate)

    def create_collection(self, recreate=False):
        if recreate:  # delete if recreate
            self.client.delete_collection(collection_name=self.collection_name)
        if not self.client.collection_exists(
            self.collection_name
        ):  # create if not exists
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.word_embedding_dimension, distance=Distance.COSINE
                ),
            )

    def add_data(self, lyrics: dict, features: list[dict], batch_size=8):
        vec_cnt = len(lyrics)
        map(lambda x: x["lyrics"], lyrics)
        vecs = self.embedding_model.encode(texts, batch_size=batch_size)
        uuids = [uuid.uuid4().hex for _ in range(vec_cnt)]
