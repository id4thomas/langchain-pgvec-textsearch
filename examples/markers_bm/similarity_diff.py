"""
Cosine Similarity Distribution Analysis for markers_bm
- Positive pairs: query vs correct corpus (from qrels)
- Negative pairs: query vs random corpus (not in qrels)
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datasets import load_dataset
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

from config import settings

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_data(data_dir: str, split: str = "test"):
    """Load qrels, queries, and corpus for markers_bm."""
    # Load qrels (ground truth)
    qrels_ds = load_dataset(data_dir, "default")
    qrels = qrels_ds[split]

    # Load queries
    queries_ds = load_dataset(data_dir, "queries")
    queries = queries_ds["queries"]

    # Load corpus
    corpus_ds = load_dataset(data_dir, "corpus")
    corpus = corpus_ds["corpus"]

    # Build query_id -> relevant corpus_ids mapping
    qrels_dict = defaultdict(set)
    for item in qrels:
        query_id = item["query-id"]
        corpus_id = item["corpus-id"]
        qrels_dict[query_id].add(corpus_id)

    # Build dictionaries
    queries_dict = {q["_id"]: q["text"] for q in queries}
    corpus_dict = {c["_id"]: c["text"] for c in corpus}
    corpus_ids = list(corpus_dict.keys())

    return qrels_dict, queries_dict, corpus_dict, corpus_ids


def main(num_samples: int = 500):
    """
    Sample queries and compute cosine similarity distributions.

    Args:
        num_samples: Number of query samples to use
    """
    print("Loading data...")
    qrels_dict, queries_dict, corpus_dict, corpus_ids = load_data(
        settings.data_dir, split="test"
    )

    # Filter queries that have qrels
    valid_query_ids = [qid for qid in qrels_dict.keys() if qid in queries_dict]
    print(f"Total queries with qrels: {len(valid_query_ids)}")

    # Sample queries
    if num_samples < len(valid_query_ids):
        sampled_query_ids = random.sample(valid_query_ids, num_samples)
    else:
        sampled_query_ids = valid_query_ids
    print(f"Sampled queries: {len(sampled_query_ids)}")

    # Initialize embedding model
    print(f"Initializing embedding model: {settings.embedding_model}")
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        dimensions=settings.embedding_dim,
        check_embedding_ctx_length=False
    )

    # Collect texts to embed
    query_texts = []
    positive_corpus_texts = []
    negative_corpus_texts = []

    for query_id in sampled_query_ids:
        query_text = queries_dict[query_id]
        relevant_ids = qrels_dict[query_id]

        # Get one positive corpus (first relevant one)
        pos_corpus_id = next(iter(relevant_ids))
        pos_corpus_text = corpus_dict[pos_corpus_id]

        # Get one random negative corpus (not in relevant set)
        while True:
            neg_corpus_id = random.choice(corpus_ids)
            if neg_corpus_id not in relevant_ids:
                break
        neg_corpus_text = corpus_dict[neg_corpus_id]

        # Add "Query: " prefix for query embedding
        query_texts.append(f"Query: {query_text}")
        positive_corpus_texts.append(pos_corpus_text)
        negative_corpus_texts.append(neg_corpus_text)

    # Embed all texts
    print("Embedding queries...")
    query_embeddings = embeddings.embed_documents(query_texts)

    print("Embedding positive corpus...")
    positive_embeddings = embeddings.embed_documents(positive_corpus_texts)

    print("Embedding negative corpus...")
    negative_embeddings = embeddings.embed_documents(negative_corpus_texts)

    # Compute cosine similarities
    print("Computing cosine similarities...")
    positive_similarities = []
    negative_similarities = []

    for i in tqdm(range(len(query_embeddings)), desc="Computing similarities"):
        q_emb = np.array(query_embeddings[i])
        pos_emb = np.array(positive_embeddings[i])
        neg_emb = np.array(negative_embeddings[i])

        positive_similarities.append(cosine_similarity(q_emb, pos_emb))
        negative_similarities.append(cosine_similarity(q_emb, neg_emb))

    # Statistics
    print("\n=== Cosine Similarity Statistics ===")
    print(f"Positive (query-correct): mean={np.mean(positive_similarities):.4f}, "
          f"std={np.std(positive_similarities):.4f}, "
          f"min={np.min(positive_similarities):.4f}, "
          f"max={np.max(positive_similarities):.4f}")
    print(f"Negative (query-random):  mean={np.mean(negative_similarities):.4f}, "
          f"std={np.std(negative_similarities):.4f}, "
          f"min={np.min(negative_similarities):.4f}, "
          f"max={np.max(negative_similarities):.4f}")

    # Plot histogram
    plt.figure(figsize=(10, 6))

    bins = np.linspace(0, 1, 50)

    plt.hist(positive_similarities, bins=bins, alpha=0.7, label="Positive (query-correct)", color="green")
    plt.hist(negative_similarities, bins=bins, alpha=0.7, label="Negative (query-random)", color="red")

    plt.axvline(np.mean(positive_similarities), color="darkgreen", linestyle="--",
                label=f"Positive mean: {np.mean(positive_similarities):.3f}")
    plt.axvline(np.mean(negative_similarities), color="darkred", linestyle="--",
                label=f"Negative mean: {np.mean(negative_similarities):.3f}")

    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title(f"Cosine Similarity Distribution (OpenAI Embedding: {settings.embedding_model})\n"
              f"markers_bm Dataset (n={len(sampled_query_ids)})")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("diff.png", dpi=150)
    print("\nHistogram saved to diff.png")
    plt.show()


if __name__ == "__main__":
    main(num_samples=500)
