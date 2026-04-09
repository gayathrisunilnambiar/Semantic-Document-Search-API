from __future__ import annotations

import numpy as np
import faiss

from model import encode, encode_single

class FAISSIndex:


    def __init__(self, dimension: int = 384) -> None:
        
        self.dimension = dimension

        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)

        self.documents: list[str] = []


    def build(self, documents: list[str]) -> None:
       
        embeddings = encode(documents)

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        self.index.add(embeddings)

        self.documents.extend(documents)

        print(
            f"[search] Indexed {len(documents)} documents "
            f"(total: {self.index.ntotal})"
        )


    def search(self, query: str, top_k: int = 3) -> list[dict]:

        if self.is_empty():
            return []

        query_vec = encode_single(query).reshape(1, -1).astype(np.float32)
        scores, ids = self.index.search(query_vec, top_k)

        results: list[dict] = []
        for score, doc_id in zip(scores[0], ids[0]):

            if doc_id == -1:
                continue
            results.append({
                "text": self.documents[doc_id],
                "score": float(score),
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    

    def is_empty(self) -> bool:

        return self.index.ntotal == 0

    def reset(self) -> None:

        self.index.reset()
        self.documents.clear()

    def __len__(self) -> int:

        return self.index.ntotal

    def __repr__(self) -> str:
        return f"FAISSIndex(dimension={self.dimension}, n_docs={len(self)})"



if __name__ == "__main__":

    sample_docs = [
        "Python is a versatile programming language used in web development and data science.",
        "Machine learning models learn patterns from data to make predictions.",
        "The Eiffel Tower is a famous landmark located in Paris, France.",
        "Neural networks are inspired by the structure of the human brain.",
        "Climate change is causing rising sea levels and extreme weather events.",
    ]

    idx = FAISSIndex(dimension=384)
    idx.build(sample_docs)

    print(f"\n{idx}\n")

    test_query = "How do artificial intelligence systems learn?"
    print(f'Query: "{test_query}"\n')

    results = idx.search(test_query, top_k=3)

    for rank, hit in enumerate(results, start=1):
        print(f"  {rank}. [score={hit['score']:.4f}] {hit['text']}")
