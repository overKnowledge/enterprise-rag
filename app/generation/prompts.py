NO_ANSWER_PHRASE = "I don't know based on the available documents."

SYSTEM_PROMPT = f"""You are a precise research assistant. Answer the user's question using ONLY the provided context excerpts below. Follow these rules strictly:

1. Base your answer entirely on the given context. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, respond exactly with: "{NO_ANSWER_PHRASE}"
3. Cite your sources inline using the format [Source: <name>, page <n>] after each claim that relies on a specific excerpt.
4. Be concise and direct. Do not pad your answer with generic disclaimers.
"""


def build_user_prompt(question: str, context_chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(context_chunks, start=1):
        meta = chunk["metadata"]
        source_label = f"{meta['source_name']}, page {meta['page_start']}"
        context_blocks.append(f"[Excerpt {i} - {source_label}]\n{chunk['text']}")

    context_text = "\n\n".join(context_blocks)

    return f"""Context excerpts:

{context_text}

Question: {question}

Answer the question using only the context above, with inline citations."""