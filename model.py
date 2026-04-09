import functools

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@functools.lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    
    print(f"[model] Loading SentenceTransformer '{MODEL_NAME}' …")
    model = SentenceTransformer(MODEL_NAME)
    print("[model] Model loaded successfully.")
    return model

def encode(texts: list[str]) -> np.ndarray:
    
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings  # type: ignore[return-value]


def encode_single(text: str) -> np.ndarray:
    
    return encode([text])[0]



if __name__ == "__main__":
    
    sentences = [
        "The weather is lovely today.",
        "It is a beautiful day outside.",
    ]

    print("Encoding two test sentences …")
    embeddings = encode(sentences)

    print(f"  Shape       : {embeddings.shape}")
    print(f"  Dtype       : {embeddings.dtype}")

    cosine_sim = np.dot(embeddings[0], embeddings[1])

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  Norms       : {norms}  (should be ≈ 1.0)")
    print(f"  Cosine sim  : {cosine_sim:.4f}")
    print()

    single = encode_single("Hello world")
    print(f"  Single shape: {single.shape}")
    print(f"  Single norm : {np.linalg.norm(single):.6f}")
