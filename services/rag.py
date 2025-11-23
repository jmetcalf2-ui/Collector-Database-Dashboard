# services/rag.py
from __future__ import annotations
import os
from typing import List, Dict, Any, Tuple
from openai import OpenAI

# rag_chunks embeddings are 1536-dim vectors (text-embedding-3-small)
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


def _get_openai() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)


def embed_query(text: str, model: str | None = None) -> List[float]:
    """
    Create an OpenAI embedding vector for a given query string.
    """
    client = _get_openai()
    mdl = model or DEFAULT_EMBEDDING_MODEL
    resp = client.embeddings.create(model=mdl, input=text)
    return resp.data[0].embedding


def semantic_search_rag_chunks(
    supabase,
    query: str,
    match_count: int = 10,
    min_similarity: float = 0.05,
    embedding_model: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Run a similarity search against public.rag_chunks via RPC, fall back to text search.
    """
    qvec = embed_query(query, model=embedding_model)

    rpc_payload = {
        "query_embedding": qvec,
        "match_count": match_count,
        "min_similarity": min_similarity,
    }

    # Try common RPC function names
    for fn in (
        "public.match_rag_chunks",
        "match_public_rag_chunks",
        "match_rag_chunks",
        "ai.match_rag_chunks",  # legacy
    ):
        try:
            res = supabase.rpc(fn, rpc_payload).execute()
            if res and getattr(res, "data", None):
                return res.data or []
        except Exception:
            continue

    # Fallback: text search
    try:
        res = (
            supabase.table("rag_chunks")
            .select("lead_id, full_name, city, country, notes, chunk_text, content, text")
            .or_(
                f"notes.ilike.%{query}%,chunk_text.ilike.%{query}%,text.ilike.%{query}%"
            )
            .limit(match_count)
            .execute()
        )
        return res.data or []
    except Exception:
        return []


def _trim_context(
    chunks: List[Dict[str, Any]], max_chars: int = 3500
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Merge top retrieved rows into a text context for GPT.
    """
    def extract_text(row: Dict[str, Any]) -> str:
        # Recognize the column your database actually uses: chunk_text
        preferred_keys = (
            "notes",
            "chunk_text",  # 👈 critical addition
            "chunk",
            "content",
            "text",
            "content_text",
            "body",
            "document",
            "snippet",
        )
        for key in preferred_keys:
            val = row.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        # Last fallback: any long-ish string that isn't metadata
        skip = {"full_name", "city", "country", "lead_id", "id", "similarity", "chunk_id"}
        for key, val in row.items():
            if key in skip:
                continue
            if isinstance(val, str):
                trimmed = val.strip()
                if len(trimmed) > 12:  # avoid tiny fields
                    return trimmed
        return ""

    used: List[Dict[str, Any]] = []
    buf: List[str] = []
    total = 0

    for row in chunks:
        piece = extract_text(row)
        if not piece:
            continue

        # Build contextual entry
        entry = (
            f"{row.get('full_name', 'Unknown')} "
            f"({row.get('city', '')}, {row.get('country', '')}):\n{piece}"
        )

        if total + len(entry) + 2 > max_chars:
            break

        buf.append(entry)
        total += len(entry) + 2
        used.append(row)

    return "\n\n".join(buf), used


def answer_with_context(
    supabase,
    question: str,
    system_prompt: str,
    match_count: int = 10,
    min_similarity: float = 0.15,
    chat_model: str | None = None,
    embedding_model: str | None = None,
    max_context_chars: int = 3500,
) -> Dict[str, Any]:
    """
    Retrieve context from public.rag_chunks and ask GPT for an answer.
    """
    # 1. Fetch relevant chunks
    chunks = semantic_search_rag_chunks(
        supabase,
        query=question,
        match_count=match_count,
        min_similarity=min_similarity,
        embedding_model=embedding_model,
    )

    # 2. Format context
    context_text, used_chunks = _trim_context(chunks, max_chars=max_context_chars)

    # 3. Build GPT prompt
    client = _get_openai()
    mdl = chat_model or DEFAULT_CHAT_MODEL
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Answer the question below using only the provided context.\n\n"
                f"Context:\n{context_text}\n\nQuestion:\n{question}"
            ),
        },
    ]

    completion = client.chat.completions.create(
        model=mdl,
        messages=messages,
        temperature=0.3,
    )

    answer = (
        completion.choices[0].message.content.strip()
        if completion.choices
        else "(no answer)"
    )

    return {
        "answer": answer,
        "sources": used_chunks,
        "meta": {
            "chat_model": mdl,
            "embedding_model": embedding_model or DEFAULT_EMBEDDING_MODEL,
            "match_count": match_count,
            "min_similarity": min_similarity,
            "context_chars": len(context_text),
            "context_snippet": context_text[:200],
        },
    }
