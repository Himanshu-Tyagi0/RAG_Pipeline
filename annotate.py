import json


def score_chunk(chunk_text, keywords):
    """Count how many expected keywords appear in the chunk."""
    text_lower = chunk_text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def annotate(template_path="eval_quaries_template.json",
             chunks_path="chunks_dump.json",
             out_path="eval_queries.json",
             min_score_ratio=0.5,
             top_n=3):
    with open(template_path, encoding="utf-8") as f:
        template = json.load(f)
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(template)} queries, {len(chunks)} chunks.\n")

    output = []
    needs_review_count = 0

    for item in template:
        query = item["query"]
        keywords = item["expected_keywords"]
        min_score = max(1, int(len(keywords) * min_score_ratio))

        scored = [(score_chunk(c["full_text"], keywords), c) for c in chunks]
        scored.sort(key=lambda x: x[0], reverse=True)

        matches = [(s, c) for s, c in scored if s >= min_score][:top_n]
        relevant_ids = [c["id"] for _, c in matches]

        needs_review = len(relevant_ids) == 0
        if needs_review:
            needs_review_count += 1

        output.append({
            "query": query,
            "relevant_ids": relevant_ids,
            "expected_keywords": keywords,
            "match_scores": [s for s, _ in matches],
            "needs_review": needs_review,
        })

        status = "⚠ NO MATCH" if needs_review else f"✓ {len(relevant_ids)} matched"
        print(f"[{status}] {query}")
        for s, c in matches:
            print(f"    score={s:<3} id={c['id']} | {c['preview'][:80]}")
        print()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"✓ Wrote {len(output)} queries to {out_path}")
    print(f"⚠ {needs_review_count} queries have no matches — "
          f"open {out_path} and fill their relevant_ids manually.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    annotate()