# services/rag.py
from __future__ import annotations
import os
from typing import List, Dict, Any, Tuple
from openai import OpenAI


DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")


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
    return resp.data[0].embedding  # type: ignore[no-any-return]


def semantic_search_rag_chunks(
    supabase,
    query: str,
    match_count: int = 10,
    min_similarity: float = 0.15,
    embedding_model: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Runs a similarity search directly against ai.rag_chunks.
    Requires the `match_rag_chunks` function (below) in your Supabase DB:
    
        create or replace function ai.match_rag_chunks(
            query_embedding vector(1536),
            match_count int default 10,
            min_similarity float default 0.15
        )
        returns table (
            lead_id uuid,
            full_name text,
            city text,
            country text,
            notes text,
            similarity float
        )
        language plpgsql as $$
        begin
            return query
            select
                lead_id,
                full_name,
                city,
                country,
                notes,
                1 - (ai.embedding <=> query_embedding) as similarity
            from ai.rag_chunks
            where 1 - (ai.rag_chunks.embeddings <=> query_embedding) > min_similarity
            order by ai.rag_chunks.embeddings <=> query_embedding
            limit match_count;
        end;
        $$;
    """
    qvec = embed_query(query, model=embedding_model)
    res = supabase.rpc(
        "match_rag_chunks",
        {
            "query_embedding": qvec,
            "match_count": match_count,
            "min_similarity": min_similarity,
        },
        schema="ai",
    ).execute()
    return res.data or []


def _trim_context(chunks: List[Dict[str, Any]], max_chars: int = 3500) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Merge top retrieved rows into a text context for GPT.
    """
    used: List[Dict[str, Any]] = []
    buf: List[str] = []
    total = 0
    for row in chunks:
        piece = row.get("notes") or ""
        if not piece:
            continue
        entry = f"{row.get('full_name', 'Unknown')}, {row.get('city', '')}, {row.get('country', '')}:\n{piece.strip()}"
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
    Retrieve context from ai.rag_chunks and ask GPT for an answer.
    """
    # 1. Fetch relevant chunks
    chunks = semantic_search_rag_chunks(
        supabase,
        question,
        match_count=match_count,
        min_similarity=min_similarity,
        embedding_model=embedding_model,
    )

    # 2. Format context
    context_text, used_chunks = _trim_context(chunks, max_chars=max_context_chars)

    # 3. Query OpenAI
    client = _get_openai()
    mdl = chat_model or DEFAULT_CHAT_MODEL
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Answer the question below using only the provided context.\n\n"
                f"Context:\n{context_text}\n\nQuestion:\n{question}"
            ),
        },
    ]
    completion = client.chat.completions.create(model=mdl, messages=messages)
    answer = completion.choices[0].message.content

    return {
        "answer": answer,
        "sources": used_chunks,
        "meta": {
            "chat_model": mdl,
            "embedding_model": embedding_model or DEFAULT_EMBEDDING_MODEL,
            "match_count": match_count,
            "min_similarity": min_similarity,
            "context_chars": len(context_text),
        },
    }
