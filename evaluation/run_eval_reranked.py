import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.embeddings.huggingface import HuggingFaceEmbeddingProvider  # noqa: E402
from app.generation.answerer import Answerer  # noqa: E402
from app.generation.groq_provider import GroqLLMProvider  # noqa: E402
from app.generation.prompts import NO_ANSWER_PHRASE  # noqa: E402
from app.vectorstore.chroma import ChromaVectorStore  # noqa: E402
from evaluation.metrics import hit_at_k, reciprocal_rank  # noqa: E402
from app.retrieval.retriever import RerankingRetriever
from app.retrieval.retriever import RerankingRetriever
from app.retrieval.reranker import CrossEncoderReranker

DATASET_PATH = Path("evaluation/datasets/golden_qa.json")
RESULTS_PATH = Path("evaluation/last_run_results_reranked.json")
TOP_K = 5


def load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text())


def load_existing_results() -> dict[str, dict]:
    if RESULTS_PATH.exists():
        existing = json.loads(RESULTS_PATH.read_text())
        return {r["id"]: r for r in existing}
    return {}


def run_evaluation(resume: bool = True) -> None:
    provider = HuggingFaceEmbeddingProvider()
    store = ChromaVectorStore(provider)
    llm = GroqLLMProvider()
    reranker = CrossEncoderReranker()
    retriever = RerankingRetriever(store, reranker, fetch_k=15, top_n=TOP_K, min_top_score=-1.0)
    answerer = Answerer(retriever, llm, top_k=TOP_K)

    questions = load_dataset()
    existing = load_existing_results() if resume else {}
    results = []

    for q in questions:
        if q["id"] in existing:
            print(f"[{q['id']}] already done, skipping")
            results.append(existing[q["id"]])
            continue

        hits = retriever.retrieve(q["question"])
        retrieved_sources = [h["metadata"]["source_name"] for h in hits]

        answer_result = answerer.answer(q["question"])
        refused = answer_result["answer"].strip().startswith(NO_ANSWER_PHRASE)

        row = {"id": q["id"], "question": q["question"], "difficulty": q["difficulty"]}

        if q["answerable"]:
            expected = q["expected_source"]
            if expected == "cross-document":
                row["hit"] = None
                row["rr"] = None
                row["retrieved_top1"] = retrieved_sources[0] if retrieved_sources else None
                row["incorrectly_refused"] = None
            else:
                row["hit"] = hit_at_k(retrieved_sources, expected)
                row["rr"] = reciprocal_rank(retrieved_sources, expected)
                row["incorrectly_refused"] = refused
            row["correctly_refused"] = None
        else:
            row["hit"] = None
            row["rr"] = None
            row["correctly_refused"] = refused
            row["incorrectly_refused"] = None

        row["answer_preview"] = answer_result["answer"][:150]
        results.append(row)
        print(f"[{q['id']}] done")

        # Save after every question, not just at the end — so a crash loses at most one question's work.
        RESULTS_PATH.write_text(json.dumps(results, indent=2))
        time.sleep(2)

    print_report(results)


def print_report(results: list[dict]) -> None:
    answerable = [r for r in results if r["hit"] is not None]
    unanswerable = [r for r in results if r["correctly_refused"] is not None]

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    if answerable:
        hit_rate = sum(1 for r in answerable if r["hit"]) / len(answerable)
        mrr = sum(r["rr"] for r in answerable) / len(answerable)
        print(f"\nRetrieval (n={len(answerable)} scoreable questions):")
        print(f"  Hit Rate: {hit_rate:.1%}")
        print(f"  MRR:      {mrr:.3f}")

        false_refusals = sum(1 for r in results if r.get("incorrectly_refused"))
        print(f"  False refusals (answerable Q refused): {false_refusals}")

    if unanswerable:
        refusal_rate = sum(1 for r in unanswerable if r["correctly_refused"]) / len(unanswerable)
        print(f"\nRefusal correctness (n={len(unanswerable)} unanswerable questions):")
        print(f"  Correctly refused: {refusal_rate:.1%}")

    print("\nPer-question detail:")
    for r in results:
        if r["hit"] is not None:
            status = "HIT" if r["hit"] else "MISS"
            if r.get("incorrectly_refused"):
                status += " (but refused to answer!)"
        elif r["correctly_refused"] is not None:
            status = "REFUSED (correct)" if r["correctly_refused"] else "ANSWERED (should have refused)"
        else:
            status = f"cross-doc, top1={r['retrieved_top1']}"
        print(f"  [{r['id']}] {status} — {r['question'][:60]}")


if __name__ == "__main__":
    run_evaluation()