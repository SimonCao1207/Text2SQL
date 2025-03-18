import pandas as pd
from sentence_transformers import SentenceTransformer, util

from const import TEST_DATA_PATH
from utils import load_data


class Retriever:
    def __init__(
        self, dataset_path="./data/ehr_null_data.csv", model_name="all-MiniLM-L6-v2"
    ):
        self.dataset_path = dataset_path
        self.model = SentenceTransformer(model_name)
        self.df = pd.read_csv(dataset_path)
        self.df["embedding"] = self.df["question"].apply(
            lambda x: self.model.encode(x, convert_to_tensor=True)
        )

    def retrieve(self, user_query):
        query_embedding = self.model.encode(user_query, convert_to_tensor=True)
        self.df["similarity"] = self.df["embedding"].apply(
            lambda x: util.pytorch_cos_sim(query_embedding, x).item()
        )
        most_similar_row = self.df.loc[self.df["similarity"].idxmax()]
        return (
            most_similar_row["id"],
            most_similar_row["question"],
            most_similar_row["similarity"],
        )


if __name__ == "__main__":
    # Load the dataset
    dataset_path = "./data/ehr_null_data.csv"
    test_data, _ = load_data(TEST_DATA_PATH, None, is_test=True)
    retriever = Retriever(dataset_path)

    thres = 0.75
    cnt = 0
    for item in test_data["data"]:
        str_id, question = item["id"], item["question"]
        most_similar_id, most_similar_question, similarity_score = retriever.retrieve(
            question
        )
        if similarity_score >= thres:
            cnt += 1
            print("===============")
            print(question)
            print(f"Most similar question ID: {most_similar_id}")
            print(f"Most similar question: {most_similar_question}")
            print(f"Similarity score: {similarity_score:.2f}")

    print(f"Total: {len(test_data['data'])}, Matched: {cnt}")
