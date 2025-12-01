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
    Run a similarity search against public.rag_chunks via RPC.
    """
    qvec = embed_query(query, model=embedding_model)

    try:
        # Call the RPC function
        res = supabase.rpc(
            "match_rag_chunks",
            {
                "query_embedding": qvec,
                "match_count": match_count,
                "min_similarity": min_similarity,
            }
        ).execute()
        
        if res and res.data:
            print(f"✓ Found {len(res.data)} chunks with similarity > {min_similarity}")
            return res.data
        else:
            print(f"✗ No chunks found above similarity threshold {min_similarity}")
            return []
            
    except Exception as e:
        print(f"✗ RPC call failed: {e}")
        # Fallback to simple text search
        try:
            res = (
                supabase.table("rag_chunks")
                .select("chunk_id, lead_id, full_name, chunk_text, source_url, source_title")
                .ilike("chunk_text", f"%{query}%")
                .limit(match_count)
                .execute()
            )
            print(f"✓ Fallback text search found {len(res.data or [])} results")
            return res.data or []
        except Exception as fallback_error:
            print(f"✗ Fallback also failed: {fallback_error}")
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
            "chunk_text",
            "notes",
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
        skip = {"full_name", "lead_id", "chunk_id", "id", "similarity", "source_url", "source_title"}
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

        # Build contextual entry without city/country since they're not in rag_chunks
        entry = f"{row.get('full_name', 'Unknown')}:\n{piece}"

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
    match_count: int = 20,
    min_similarity: float = 0.10,
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
    
    # Enhanced system prompt
    enhanced_system = (
        f"{system_prompt}\n\n"
        "IMPORTANT: When answering about collectors:\n"
        "- List specific collector names from the context\n"
        "- Explain WHY each collector would be interested based on their documented history\n"
        "- Reference specific artists, movements, or institutions they support\n"
        "- Be concrete and factual - avoid generalizations"
    )
    
    messages = [
        {"role": "system", "content": enhanced_system},
        {
            "role": "user",
            "content": (
                "Answer using ONLY the collectors mentioned in the context below.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question:\n{question}"
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
