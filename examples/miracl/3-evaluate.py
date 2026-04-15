# %%
"""
Retrieval Evaluation Script for MIRACL Dataset

Metrics:
- Recall@k, Precision@k, nDCG@k, MRR, MAP

Usage:
    python 3-evaluate.py --config configs/ko-qwen3.yaml --eval-config configs/evaluate.yaml
    python 3-evaluate.py --config configs/en-qwen3.yaml --eval-config configs/evaluate.yaml
"""
import argparse
import asyncio
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_pgvec_textsearch import (
    PGVecTextSearchStore,
    AsyncPGVecTextSearchEngine,
    TableConfig,
    SearchConfig,
)
from tqdm.asyncio import tqdm as atqdm
import pandas as pd

from config import (
    settings,
    load_experiment_config,
    build_search_configs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval on MIRACL")
    parser.add_argument("--index-config", type=str, required=True, help="Path to index YAML config (model/lang specific)")
    parser.add_argument("--eval-config", type=str, default="configs/evaluate.yaml", help="Path to evaluate YAML config (default: configs/evaluate.yaml)")
    return parser.parse_args()


# %% [markdown]
# # Data Loading Functions

# %%
def load_topics(filepath: str) -> dict[str, str]:
    queries = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                qid, query = parts[0], parts[1]
                queries[qid] = query
    return queries


def load_qrels(filepath: str) -> dict[str, set[str]]:
    qrels = defaultdict(set)
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                if rel == 1:
                    qrels[qid].add(docid)
    return dict(qrels)


def load_evaluation_data(data_dir: str, lang: str, split: str = "dev"):
    base_path = Path(data_dir) / f"miracl-v1.0-{lang}"
    topics_file = base_path / "topics" / f"topics.miracl-v1.0-{lang}-{split}.tsv"
    qrels_file = base_path / "qrels" / f"qrels.miracl-v1.0-{lang}-{split}.tsv"
    queries_dict = load_topics(str(topics_file))
    qrels_dict = load_qrels(str(qrels_file))
    return qrels_dict, queries_dict


# %% [markdown]
# # Evaluation Metrics

# %%
@dataclass
class RetrievalMetrics:
    recall: float
    precision: float
    ndcg: float
    mrr: float
    ap: float


def compute_dcg(relevances: list[float], k: int) -> float:
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def compute_ndcg(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    relevances = [1.0 if doc_id in relevant_ids else 0.0 for doc_id in retrieved_ids[:k]]
    dcg = compute_dcg(relevances, k)
    num_relevant = min(len(relevant_ids), k)
    ideal_relevances = [1.0] * num_relevant + [0.0] * (k - num_relevant)
    idcg = compute_dcg(ideal_relevances, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_average_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    num_relevant_seen = 0
    precision_sum = 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            num_relevant_seen += 1
            precision_sum += num_relevant_seen / (i + 1)
    if num_relevant_seen == 0:
        return 0.0
    return precision_sum / len(relevant_ids)


def compute_metrics(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> RetrievalMetrics:
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
    return RetrievalMetrics(recall=recall, precision=precision, ndcg=ndcg, mrr=mrr, ap=ap)


# %% [markdown]
# # Evaluation Loop

# %%
async def evaluate_config(
    engine: AsyncPGVecTextSearchEngine,
    embedding_service,
    table_config: TableConfig,
    config_name: str,
    config: SearchConfig,
    qrels_dict: dict[str, set[str]],
    queries_dict: dict[str, str],
    k_values: list[int],
) -> dict[str, dict[str, float]]:
    store = await PGVecTextSearchStore.create(
        engine=engine,
        embedding_service=embedding_service,
        table_config=table_config,
        search_config=config,
    )

    results_by_k = {k: defaultdict(list) for k in k_values}
    max_k = max(k_values)
    valid_query_ids = [qid for qid in qrels_dict.keys() if qid in queries_dict]

    print(f"\nEvaluating {config_name} ({len(valid_query_ids)} queries)...")

    for query_id in atqdm(valid_query_ids, desc=config_name):
        query_text = f"Query: {queries_dict[query_id]}"
        relevant_ids = qrels_dict[query_id]
        results = await store.asimilarity_search_with_score(query_text, k=max_k)
        retrieved_ids = [doc.metadata.get("docid") for doc, _ in results]

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
async def main(index_cfg: dict, eval_cfg: dict):
    ds = index_cfg["dataset"]
    emb = index_cfg.get("embedding", {})

    table_name = ds["table_name"]
    lang = ds["lang"]
    text_config = ds["text_config"]
    k_values = eval_cfg.get("k_values", [5, 10, 20])
    split = eval_cfg.get("split", "dev")

    database_url = "postgresql+asyncpg://{}:{}@{}:{}/{}".format(
        settings.postgres_user,
        settings.postgres_password,
        settings.postgres_ip,
        settings.postgres_port,
        settings.postgres_db,
    )

    engine = AsyncPGVecTextSearchEngine.from_connection_string(database_url)

    embedding_dim = emb.get("dim", settings.embedding_dim)
    table_config = TableConfig(
        table_name=table_name,
        vector_size=embedding_dim,
    )

    embedding_model = emb.get("model", settings.embedding_model)
    embedding_base_url = emb.get("base_url", settings.embedding_base_url)
    embedding_api_key = emb.get("api_key", settings.embedding_api_key)

    print(f"=== MIRACL Evaluation (text_config: {text_config}) ===")
    print(f"Table: {table_name}")
    print(f"Language: {lang}")
    print(f"EMBEDDING: {embedding_model}")

    embedding_service = OpenAIEmbeddings(
        model=embedding_model,
        base_url=embedding_base_url,
        api_key=embedding_api_key,
        dimensions=embedding_dim,
        check_embedding_ctx_length=False,
    )

    print(f"\nLoading evaluation data from {settings.data_dir} (split: {split})...")
    qrels_dict, queries_dict = load_evaluation_data(settings.data_dir, lang, split)
    print(f"Loaded {len(queries_dict)} queries, {len(qrels_dict)} with relevance judgments")

    configs = build_search_configs(index_cfg, eval_cfg)

    all_results = {}
    for config_name, config in configs.items():
        results = await evaluate_config(
            engine=engine,
            embedding_service=embedding_service,
            table_config=table_config,
            config_name=config_name,
            config=config,
            qrels_dict=qrels_dict,
            queries_dict=queries_dict,
            k_values=k_values,
        )
        all_results[config_name] = results

    await engine.close()
    return all_results, k_values


def print_results(all_results: dict, k_values: list[int]):
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS")
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
    args = parse_args()
    index_cfg = load_experiment_config(args.index_config)
    eval_cfg = load_experiment_config(args.eval_config)

    all_results, k_values = asyncio.run(main(index_cfg, eval_cfg))
    print_results(all_results, k_values)

# %%
# For Jupyter notebook:
# index_cfg = load_experiment_config("configs/ko-qwen3.yaml")
# eval_cfg = load_experiment_config("configs/evaluate.yaml")
# all_results, k_values = await main(index_cfg, eval_cfg)
# summary_df = print_results(all_results, k_values)
