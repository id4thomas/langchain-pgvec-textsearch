# %%
"""
Retrieval Evaluation Script for Ko-StrategyQA Dataset (Korean) - pgvector dense only

Uses langchain-postgres official package for comparison.
Dense vector search only (no BM25/hybrid).

Metrics:
- Recall@k, Precision@k, nDCG@k, MRR, MAP
"""
import asyncio
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from datasets import load_dataset
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from tqdm.asyncio import tqdm as atqdm
import pandas as pd

from config import settings

DATA_NAME = "KoStrategyQA"
TABLE_NAME = f"{DATA_NAME}-pgvec-documents"
QUERY_LANG = "ko"

# %%
DATABASE_URL = "postgresql+psycopg://{}:{}@{}:{}/{}".format(
    settings.postgres_user,
    settings.postgres_password,
    settings.postgres_ip,
    settings.postgres_port,
    settings.postgres_db,
)


# %% [markdown]
# # Evaluation Metrics

# %%
@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""
    recall: float
    precision: float
    ndcg: float
    mrr: float
    ap: float


def compute_dcg(relevances: list[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at k."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    return dcg


def compute_ndcg(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Normalized DCG at k."""
    relevances = [1.0 if doc_id in relevant_ids else 0.0 for doc_id in retrieved_ids[:k]]
    dcg = compute_dcg(relevances, k)

    num_relevant = min(len(relevant_ids), k)
    ideal_relevances = [1.0] * num_relevant + [0.0] * (k - num_relevant)
    idcg = compute_dcg(ideal_relevances, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_average_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute Average Precision for a single query."""
    if not relevant_ids:
        return 0.0

    num_relevant_seen = 0
    precision_sum = 0.0

    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            num_relevant_seen += 1
            precision_at_i = num_relevant_seen / (i + 1)
            precision_sum += precision_at_i

    if num_relevant_seen == 0:
        return 0.0

    return precision_sum / len(relevant_ids)


def compute_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int
) -> RetrievalMetrics:
    """Compute all retrieval metrics for a single query."""
    retrieved_at_k = retrieved_ids[:k]
    retrieved_set = set(retrieved_at_k)

    tp = len(retrieved_set & relevant_ids)
    recall = tp / len(relevant_ids) if relevant_ids else 0.0
    precision = tp / k if k > 0 else 0.0
    ndcg = compute_ndcg(retrieved_ids, relevant_ids, k)

    mrr = 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            mrr = 1.0 / (i + 1)
            break

    ap = compute_average_precision(retrieved_ids, relevant_ids)

    return RetrievalMetrics(
        recall=recall,
        precision=precision,
        ndcg=ndcg,
        mrr=mrr,
        ap=ap,
    )


# %% [markdown]
# # Load Data

# %%
def load_evaluation_data(data_dir: str, split: str = "dev", query_lang: str = "ko"):
    """Load qrels and queries for evaluation."""
    # Load qrels (ground truth)
    qrels_ds = load_dataset(data_dir, "default")
    qrels = qrels_ds[split]

    # Load queries (Korean)
    queries_ds = load_dataset(data_dir, "queries")
    queries = queries_ds[query_lang]

    # Build query_id -> relevant corpus_ids mapping
    qrels_dict = defaultdict(set)
    for item in qrels:
        query_id = item["query-id"]
        corpus_id = item["corpus-id"]
        qrels_dict[query_id].add(corpus_id)

    # Build query_id -> query_text mapping
    queries_dict = {q["_id"]: q["text"] for q in queries}

    return qrels_dict, queries_dict


# %% [markdown]
# # Evaluation Loop

# %%
async def evaluate(
    engine: PGEngine,
    embedding_service,
    qrels_dict: dict[str, set[str]],
    queries_dict: dict[str, str],
    k_values: list[int],
) -> dict[str, dict[str, float]]:
    """Evaluate dense search across all queries."""
    store = await PGVectorStore.create(
        engine=engine,
        embedding_service=embedding_service,
        table_name=TABLE_NAME,
    )

    results_by_k = {k: defaultdict(list) for k in k_values}
    max_k = max(k_values)

    valid_query_ids = [qid for qid in qrels_dict.keys() if qid in queries_dict]

    print(f"\nEvaluating pgvector dense ({len(valid_query_ids)} queries)...")

    for query_id in atqdm(valid_query_ids, desc="pgvector dense"):
        query_text = f"Query: {queries_dict[query_id]}"
        relevant_ids = qrels_dict[query_id]

        results = await store.asimilarity_search_with_score(query_text, k=max_k)
        retrieved_ids = [doc.metadata.get("corpus_id") for doc, _ in results]

        for k in k_values:
            metrics = compute_metrics(retrieved_ids, relevant_ids, k)
            results_by_k[k]["recall"].append(metrics.recall)
            results_by_k[k]["precision"].append(metrics.precision)
            results_by_k[k]["ndcg"].append(metrics.ndcg)
            results_by_k[k]["mrr"].append(metrics.mrr)
            results_by_k[k]["ap"].append(metrics.ap)

    aggregated = {}
    for k in k_values:
        aggregated[k] = {
            f"Recall@{k}": np.mean(results_by_k[k]["recall"]),
            f"Precision@{k}": np.mean(results_by_k[k]["precision"]),
            f"nDCG@{k}": np.mean(results_by_k[k]["ndcg"]),
            "MRR": np.mean(results_by_k[k]["mrr"]),
            "MAP": np.mean(results_by_k[k]["ap"]),
        }

    return aggregated


# %% [markdown]
# # Main Evaluation

# %%
async def main(k_values: list[int] = [5, 10, 20], split: str = "dev"):
    """Run evaluation."""
    engine = PGEngine.from_connection_string(DATABASE_URL)

    print(f"=== Ko-StrategyQA pgvector Evaluation ===")
    print(f"Table: {TABLE_NAME}")
    print(f"Query language: {QUERY_LANG}")
    print(f"EMBEDDING: {settings.embedding_model}")

    embedding_service = OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        dimensions=settings.embedding_dim,
        check_embedding_ctx_length=False
    )

    print(f"\nLoading evaluation data from {settings.data_dir} (split: {split})...")
    qrels_dict, queries_dict = load_evaluation_data(
        settings.data_dir, split, query_lang=QUERY_LANG
    )
    print(f"Loaded {len(qrels_dict)} queries with relevance judgments")

    results = await evaluate(
        engine=engine,
        embedding_service=embedding_service,
        qrels_dict=qrels_dict,
        queries_dict=queries_dict,
        k_values=k_values,
    )

    return {"pgvector dense": results}


def print_results(all_results: dict, k_values: list[int]):
    """Print results in formatted tables."""
    print(f"\n{'='*70}")
    print(f"Ko-StrategyQA pgvector EVALUATION RESULTS")
    print(f"{'='*70}")

    for k in k_values:
        print(f"\n--- Results @ k={k} ---")
        rows = []
        for config_name, results in all_results.items():
            row = {"Config": config_name}
            row.update(results[k])
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.set_index("Config")
        print(df.to_string(float_format=lambda x: f"{x:.4f}"))

    # Summary table
    print(f"\n{'='*70}")
    print("Summary Table")
    print(f"{'='*70}")

    summary_rows = []
    for config_name, results in all_results.items():
        row = {"Config": config_name}
        for k in k_values:
            row[f"R@{k}"] = results[k][f"Recall@{k}"]
            row[f"nDCG@{k}"] = results[k][f"nDCG@{k}"]
        row["MRR"] = results[k_values[0]]["MRR"]
        row["MAP"] = results[k_values[0]]["MAP"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.set_index("Config")
    print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))

    return summary_df


# %%
if __name__ == "__main__":
    K_VALUES = [5, 10, 20]
    SPLIT = "dev"

    results = asyncio.run(main(k_values=K_VALUES, split=SPLIT))
    print_results(results, K_VALUES)

# %%
# For Jupyter notebook:
# results = await main(k_values=[5, 10, 20], split="dev")
# summary_df = print_results(results, [5, 10, 20])
