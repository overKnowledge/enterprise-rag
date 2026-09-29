# Evaluation methodology and findings

## Dataset

27 hand-written questions across four documents and one web article:
20 scored answerable questions (easy, medium and hard, including paraphrased,
negated and vocabulary-avoidant phrasing), 2 cross-document synthesis questions
(logged for manual review, not scored — there's no single ground-truth source),
and 5 deliberately unanswerable or adversarial questions that must be refused.

Two adversarial questions matter more than the rest: one asks something *near* the
corpus's own content but misstates how it's framed, and one asks for a specific detail
(a repository URL) that sounds plausible but was never in the source. Both are designed
to tempt a model into confidently making something up rather than admitting it doesn't
know.

## Metrics

- **Hit Rate**: did the correct source document appear anywhere in the top-5 retrieved
  chunks?
- **MRR**: how highly was it ranked when found? (1.0 = always rank 1)
- **Refusal correctness**: for the 5 adversarial questions, did the system correctly
  decline to answer, with no fabricated content or citation?

## Baseline result

| Metric | Value |
|---|---|
| Hit Rate (20 scored questions) | 100% |
| MRR | 0.950 |
| False refusals (answerable question refused) | 0 |
| Correct refusals (of 5 adversarial) | 5/5 |

## Finding 1: hallucination with a false citation

Asked *"What model won the WMT 2014 competition outright, according to this paper?"*,
the system answered that the big Transformer "won the competition outright" and
attached a page citation. The actual source text says the model *outperformed prior
published results and established a new state-of-the-art* — it never frames this as a
competition or a win. The citation pointed at a real page, but not at text supporting
the claim as stated.

**Fix**: the grounding prompt was rewritten with a concrete negative example (don't
upgrade "outperformed prior results" into "won a competition") and an explicit
citation-integrity rule. Retested on the original model (`gpt-oss-20b`): the
fabricated language was removed, but the system still answered a question built on a
false premise instead of rejecting it outright. Retested on `openai/gpt-oss-120b`
with the same prompt: it correctly refused. **The prompt fix and the larger model each
closed part of the gap; neither alone fully fixed it.**

## Finding 2: reranking regression, and a miscalibrated fix

Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`, fetch top-15 then rerank to top-5)
was added and evaluated as an alternative retrieval path.

| Configuration | Hit Rate | MRR | False refusals | Correct refusals (of 5) |
|---|---|---|---|---|
| Baseline (vector search only) | 100% | 0.950 | 0 | 5 |
| + rerank, no confidence filter | 100% | 0.950 | 0 | **4** |
| + rerank, absolute threshold (score ≥ 0) | 90% | 0.850 | **2** | 5 |
| + rerank, relative threshold (top score ≥ -1.0) | 95% | 0.900 | **1** | 5 |

Adding reranking with no filtering caused the repository-URL adversarial question to
get an invented answer. Inspection showed every retrieved candidate scored negative on
the cross-encoder, a signal the pipeline wasn't using. Filtering out any chunk scoring
below 0 fixed that case, but introduced two new false refusals: correct chunks for
indirectly-worded questions legitimately score negative on this model, so an absolute
cutoff discarded valid evidence. Replacing it with a relative rule (only reject the
whole result set if even the single best candidate is weak, not each middling one)
recovered most of the false refusals, but one deliberately vocabulary-avoidant question
still fails at any reasonable threshold.

**Decision**: reranking is implemented, threshold-tuned and evaluated, but it does not
beat plain vector search on this corpus, so it is not enabled by default in
`POST /v1/chat`. It's kept in the codebase (`app/retrieval/`) as an evaluated, optional
path — the honest result of the experiment, not a forced win.

## Reproducing this

```bash
python evaluation/run_eval.py            # baseline, writes evaluation/last_run_results.json
python evaluation/run_eval_reranked.py   # reranked, writes evaluation/last_run_results_reranked.json
```

Both are resumable and rate-limit aware (Groq's free tier caps tokens per minute and
per day); a crashed run picks up where it left off on the next invocation. Requires
your own ingested documents (`evaluation/datasets/golden_qa.json` references source
filenames, so questions will only score against a corpus containing matching content).