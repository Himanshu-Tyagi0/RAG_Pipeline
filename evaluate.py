import json
import hashlib
from statistics import mean
from RAG_full_pipeline import EmbeddingManager, VectorStoreManager, HybridRetriever


def content_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------- Metrics ----------
def recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    return len(retrieved_set & relevant_set) / len(relevant_set)


def precision_at_k(retrieved_ids, relevant_ids, k):
    if k == 0:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / k


def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant_set = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / i
    return 0.0


# ---------- Main eval ----------
def evaluate(retriever, dataset, mode, k=5):
    recalls, precisions, rrs = [], [], []

    for item in dataset:
        query = item["query"]
        relevant = item["relevant_ids"]

        results = retriever.retrieve(query, top_k=k, mode=mode)

        # Match by content hash — works across all three modes
        retrieved_ids = [content_id(r["documents"]) for r in results]

        recalls.append(recall_at_k(retrieved_ids, relevant, k))
        precisions.append(precision_at_k(retrieved_ids, relevant, k))
        rrs.append(reciprocal_rank(retrieved_ids, relevant))

    return {
        "mode": mode,
        "k": k,
        "recall": mean(recalls),
        "precision": mean(precisions),
        "mrr": mean(rrs),
        "num_queries": len(dataset),
    }


def print_results(results):
    print(f"\n{'=' * 72}")
    print(f"{'Mode':<12} {'Recall@' + str(results[0]['k']):<12} "
          f"{'Precision@' + str(results[0]['k']):<14} {'MRR':<10}")
    print(f"{'-' * 72}")
    for r in results:
        print(f"{r['mode']:<12} {r['recall']:<12.4f} "
              f"{r['precision']:<14.4f} {r['mrr']:<10.4f}")
    print(f"{'=' * 72}")
    print(f"Queries evaluated: {results[0]['num_queries']}\n")


def main():
    # Load data
    with open("eval_queries.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} eval queries.")

    # Build retriever (reuses your existing pipeline)
    em = EmbeddingManager()
    vs = VectorStoreManager()
    retriever = HybridRetriever(em, vs)

    # Run all three modes
    results = []
    for mode in ["vector", "bm25", "hybrid"]:
        print(f"\n--- Evaluating mode: {mode} ---")
        results.append(evaluate(retriever, dataset, mode=mode, k=5))

    print_results(results)

    # ─── Weight sweep (with error tracking) ───
    print("\n--- Weight sweep (hybrid only) ---")
    weight_configs = [
        (0.5, 0.5),
        (0.4, 0.6),
        (0.3, 0.7),
        (0.6, 0.4),
        (0.7, 0.3),
    ]
    print(f"Running {len(weight_configs)} weight configs...")

    for bm25_w, vec_w in weight_configs:
        print(f"\nTrying weights=(bm25={bm25_w}, vec={vec_w}) ...", end=" ")
        try:
            tuned = HybridRetriever(em, vs, weights=(bm25_w, vec_w))
            r = evaluate(tuned, dataset, mode="hybrid", k=5)
            print(f"OK — Recall={r['recall']:.4f}  Precision={r['precision']:.4f}  MRR={r['mrr']:.4f}")
        except Exception as e:
            print(f"FAILED — {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()