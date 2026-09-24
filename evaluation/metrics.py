def hit_at_k(retrieved_sources: list[str], expected_source: str) -> bool:
    """True if the expected source appears anywhere in the retrieved results."""
    return expected_source in retrieved_sources


def reciprocal_rank(retrieved_sources: list[str], expected_source: str) -> float:
    """1/rank of the first correct hit, 0 if not found. Rank is 1-indexed."""
    for i, source in enumerate(retrieved_sources, start=1):
        if source == expected_source:
            return 1.0 / i
    return 0.0