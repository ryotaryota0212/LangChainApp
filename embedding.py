import os
import openai
from sklearn.metrics.pairwise import cosine_similarity


def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        model=model,
        input=text
    )
    return [response.data[0].embedding]


api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = api_key

# テキストの埋め込みベクトルを取得
vector1 = get_embedding("こんにちわ")
vector2 = get_embedding("こんばんわ")

# コサイン類似度を計算
result = cosine_similarity(vector1, vector2)

print(result)
