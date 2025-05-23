import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from const import TEST_DATA_PATH
from utils import load_data


class VectorDB:
    def __init__(
        self,
        dataset_path,
        index_path="./data/default.index",
        model_name="all-MiniLM-L6-v2",
    ):
        self.dataset_path = dataset_path
        self.index_path = Path(index_path)
        self.model_name = model_name
        if model_name == "emilyalsentzer/Bio_ClinicalBERT":
            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.model = SentenceTransformer(model_name)
        self.df = pd.read_csv(dataset_path)
        self.data_dict = self.df.to_dict(orient="records")
        self.index = None

    def initialize(self):
        if not self.index_path.exists():
            self.build_index()
        else:
            self.load_index()

    def embed_text(self, query):
        if self.model_name == "all-MiniLM-L6-v2":
            return self.model.encode(query, convert_to_tensor=True).cpu()
        inputs = self.tokenizer(query, return_tensors="pt")
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embedding.detach().numpy()

    def save_index(self):
        faiss.write_index(self.index, str(self.index_path))

    def load_index(self):
        print(f"Loading index from {self.index_path}...")
        self.index = faiss.read_index(str(self.index_path))

    def build_index(self):
        embeddings = []

        print(f"Buidling index and save to {self.index_path} ...")
        # index all questions in the dataset and save the index
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            question = row["question"]
            embedding = self.embed_text(question)
            embeddings.append(embedding)

        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)  # type: ignore
        self.save_index()


class Retriever:
    def __init__(self, vector_db: VectorDB, top_n=3):
        self.vector_db = vector_db
        self.top_n = top_n

    def retrieve(self, question):
        """
        Given a user input, relevant splits are retrieved from storage using a Retriever.
        """

        query_embedding = np.array(
            [self.vector_db.embed_text(question)], dtype=np.float32
        )
        faiss.normalize_L2(query_embedding)
        if self.vector_db.index:
            distances, indices = self.vector_db.index.search(
                query_embedding, k=self.top_n
            )  # type: ignore
            results = [
                (self.vector_db.data_dict[idx], distances[0][i])
                for i, idx in enumerate(indices[0])
            ]
            return results
        return None


if __name__ == "__main__":
    # Load the dataset
    dataset_path = "./data/text_sql.csv"
    # dataset_path = "./data/ehr_null_data.csv"

    if not os.path.exists("./data/index"):
        os.makedirs("./data/index")

    test_data, _ = load_data(TEST_DATA_PATH, None, is_test=True)
    vector_db = VectorDB(
        dataset_path=dataset_path,
        index_path="./data/index/text_sql.index",
        # index_path="./data/index/ehr_null.index",
        # model_name="emilyalsentzer/Bio_ClinicalBERT",
    )

    if not vector_db.index_path.exists():
        vector_db.build_index()
    else:
        vector_db.load_index()

    retriever = Retriever(vector_db)

    thres = 0.4
    cnt = 0
    tmp = 0
    for item in test_data["data"]:
        str_id, question = item["id"], item["question"]
        results = retriever.retrieve(question)
        print("===============")
        print(f"Question : {question}")
        if results:
            old_cnt = cnt
            for i in range(len(results)):
                item, distance = results[i]
                if distance <= thres:
                    cnt += 1
                    print(f"Similarity score ({distance:.2f}): {item['question']}")
            if old_cnt != cnt:
                tmp += 1

    print(f"Total: {len(test_data['data'])}, Matched: {tmp}")
    print(f"Number of examples: {cnt}")
