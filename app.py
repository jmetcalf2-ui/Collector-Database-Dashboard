import streamlit as st
from supabase_client import get_supabase
from components import inject_css
import os
from openai import OpenAI
from datetime import datetime
from pathlib import Path
import pandas as pd
from services.rag import semantic_search_rag_chunks, _trim_context

st.set_page_config(page_title="Dashboard", layout="wide")
inject_css()

st.markdown("""
<style>
div[data-testid="stButton"] > button {
    text-align: left !important;
    justify-content: flex-start !important;
    border: 1px solid #ddd !important;
    border-radius: 6px !important;
    background-color: #f9f9f9 !important;
    padding: 8px 12px !important;
    margin-bottom: 6px !important;
    font-weight: 500 !important;
    color: #222 !important;
    white-space: pre-wrap !important;
}
div[data-testid="stButton"] > button:hover {
    background-color: #f1f1f1 !important;
    border-color: #ccc !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.write(" ")

if "selected_leads" not in st.session_state:
    st.session_state.selected_leads = []
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []
if "active_chat" not in st.session_state:
    st.session_state.active_chat = []
if "chat_sources" not in st.session_state:
    st.session_state.chat_sources = []
if "current_chat_open" not in st.session_state:
    st.session_state.current_chat_open = None
if "current_session_title" not in st.session_state:
    st.session_state.current_session_title = ""
if "current_session_summary" not in st.session_state:
    st.session_state.current_session_summary = ""

st.markdown("""
<style>
div[data-testid="column"]:first-child {
    background-color: #fafafa;
    border-right: 1px solid #eee;
    padding-right: 12px;
}
div[data-testid="stButton"] > button.chat-btn {
    text-align: left;
    border: 1px solid #ddd;
    border-radius: 6px;
    background-color: #f9f9f9;
    padding: 8px 10px;
    margin-bottom: 6px;
    font-weight: 500;
    color: #222;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Dashboard</h1>", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_supabase():
    return get_supabase()

try:
    supabase = load_supabase()
except Exception as e:
    st.error(f"⚠️ Supabase connection failed: {e}")
    supabase = None

@st.cache_data(show_spinner=False)
def get_total_leads_count():
    if not supabase:
        return 0
    try:
        res = supabase.table("leads").select("*", count="exact").limit(1).execute()
        return getattr(res, "count", 0)
    except:
        return 0

@st.cache_data(show_spinner=False)
def get_full_grid_page(page, per_page):
    if not supabase:
        return []
    offset = page * per_page
    try:
        res = (
            supabase.table("leads")
            .select("lead_id, full_name, email, tier, primary_role, city, country, notes")
            .order("full_name", desc=False)
            .range(offset, offset + per_page - 1)
            .execute()
        )
        return res.data or []
    except:
        return []

@st.cache_data(show_spinner=False)
def get_contacts_page(page, per_page):
    if not supabase:
        return []
    offset = page * per_page
    try:
        res = (
            supabase.table("leads")
            .select("full_name, email, tier, primary_role, city, country, notes")
            .order("created_at", desc=True)
            .range(offset, offset + per_page - 1)
            .execute()
        )
        return res.data or []
    except:
        return []

@st.cache_data(show_spinner=False)
def load_chat_sessions():
    if not supabase:
        return []
    try:
        res = (
            supabase.table("chat_sessions")
            .select("*")
            .order("id", desc=True)
            .execute()
        )
        return res.data or []
    except:
        return []

@st.cache_data(show_spinner=False)
def get_query_embedding_cached(query):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Missing key")
    client = OpenAI(api_key=key)
    emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    ).data[0].embedding
    return emb

@st.cache_data(show_spinner=False)
def summarize_collector(lead_id, combined_notes):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return "⚠️ Missing OPENAI_API_KEY"
    if not combined_notes.strip():
        return "⚠️ No notes found"

    try:
        client = OpenAI(api_key=key)
        prompt = f"""
Summarize collector notes into 4–6 bullet points.
Avoid adjectives. Focus on concrete data.

NOTES:
{combined_notes}
"""
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize collectors factually."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"


@st.cache_data(show_spinner=False)
def load_lead_supplements(lead_id: str):
    """Cached fetch of supplemental notes for a lead to avoid repeated round-trips."""
    if not supabase:
        return []
    try:
        res = (
            supabase.table("leads_supplements")
            .select("notes")
            .eq("lead_id", lead_id)
            .execute()
        )
        return res.data or []
    except Exception:
        return []


def prefetch_page(cache_fn, page: int, per_page: int, direction: int = 1):
    """
    Warm the cache for an adjacent page so pagination feels instant.
    """
    target = page + direction
    if target < 0:
        return
    try:
        cache_fn(target, per_page)
    except Exception:
        pass


def render_lead_detail(lead, summary_prefix: str):
    """Single detail pane for a lead to reduce re-render load."""
    lead_id = str(lead.get("lead_id"))
    name = lead.get("full_name", "Unnamed")
    role_val = lead.get("primary_role", "—")
    tier_val = lead.get("tier", "—")
    city_val = (lead.get("city") or "").strip()
    country_val = (lead.get("country") or "").strip()
    email_val = lead.get("email", "—")
    notes_val = (lead.get("notes") or "").strip()

    location = ", ".join([p for p in [city_val, country_val] if p]) or "—"

    st.markdown(f"### {name}")
    st.caption(f"{role_val} • Tier {tier_val} • {location}")
    st.write(email_val)

    if notes_val:
        st.markdown("**Notes**")
        st.write(notes_val)

    summary_key = f"summary_{summary_prefix}_{lead_id}"
    if summary_key not in st.session_state:
        if st.button(
            f"Summarize {name}",
            key=f"sum_{summary_prefix}_{lead_id}",
            use_container_width=True,
        ):
            with st.spinner("Summarizing notes..."):
                supplements = load_lead_supplements(lead_id)
                supplement_notes = "\n\n".join(
                    (s.get("notes") or "").strip() for s in supplements
                )

                combined = (
                    notes_val
                    + ("\n\n" if notes_val and supplement_notes else "")
                    + supplement_notes
                ).strip()

                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                prompt = f"""
Summarize these collector notes into 5–7 factual bullet points.
Avoid adjectives. Focus on artists collected, museum affiliations, geography,
philanthropy, acquisitions, and collecting tendencies.

NOTES:
{combined}
"""

                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You summarize art collectors factually.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=500,
                )

                st.session_state[summary_key] = resp.choices[0].message.content.strip()
                st.rerun()
    else:
        st.markdown("**Summary**")
        st.markdown(st.session_state[summary_key], unsafe_allow_html=True)


def build_context_from_rag(
    question: str,
    max_chars: int = 3200,
    match_count: int = 12,
    min_similarity: float = 0.1,
):
    """Pull contextual notes from ai.rag_chunks/public.leads for chat grounding."""
    if not supabase or not question.strip():
        return "", []
    rows = []
    # Primary: use ai.rag_chunks RPC
    try:
        rows = semantic_search_rag_chunks(
            supabase,
            query=question,
            match_count=match_count,
            min_similarity=min_similarity,
        )
    except Exception:
        rows = []

    # Fallback 1: semantic search on leads/leads_supplements
    if not rows:
        try:
            emb = get_query_embedding_cached(question)
            rpc = supabase.rpc(
                "rpc_semantic_search_leads_supplements",
                {
                    "query_embedding": emb,
                    "match_count": match_count,
                    "min_score": min_similarity,
                },
            ).execute()
            matched_ids = []
            for r in rpc.data or []:
                lid = r.get("lead_id")
                if lid:
                    matched_ids.append(str(lid))
            if matched_ids:
                res = (
                    supabase.table("leads")
                    .select("lead_id, full_name, city, country, notes")
                    .in_("lead_id", matched_ids)
                    .execute()
                )
                rows = res.data or []
        except Exception:
            rows = []

    # Fallback 2: simple text search on notes when embeddings/rpc fail
    if not rows:
        try:
            res = (
                supabase.table("leads")
                .select("lead_id, full_name, city, country, notes")
                .ilike("notes", f"%{question}%")
                .limit(match_count)
                .execute()
            )
            rows = res.data or []
        except Exception:
            rows = []

    context_text, used_rows = _trim_context(rows, max_chars=max_chars)
    return context_text, used_rows

tabs = st.tabs(["Search", "Contacts", "Saved Sets"])

# ========================
# SEARCH TAB
# ========================
with tabs[0]:
    st.markdown("## Search")

    st.markdown("""
    <style>
    input[data-testid="stTextInput"]::placeholder {
        color:#999 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns([2.2,1.2,1.2,1.2,1.2])
    with col1:
        keyword = st.text_input("Keyword")
    with col2:
        city = st.text_input("City")
    with col3:
        country = st.text_input("Country")
    with col4:
        tier = st.selectbox("Tier", ["", "A", "B", "C"])
    with col5:
        role = st.text_input("Primary Role")

    semantic_query = st.text_input("Semantic Search")
    semantic_min_score = 0.35
    semantic_max_matches = 80
    min_semantic_chars = 8

    def semantic_filter_passes(lead_row):
        """Apply field-level filters to semantic results for extra precision."""
        def contains(value, term):
            return term.lower() in (value or "").lower()

        if keyword and not (
            contains(lead_row.get("full_name", ""), keyword)
            or contains(lead_row.get("email", ""), keyword)
            or contains(lead_row.get("primary_role", ""), keyword)
        ):
            return False
        if city and not contains(lead_row.get("city", ""), city):
            return False
        if country and not contains(lead_row.get("country", ""), country):
            return False
        if tier and lead_row.get("tier") != tier:
            return False
        if role and not contains(lead_row.get("primary_role", ""), role):
            return False
        return True

    if (
        keyword.strip()=="" and city.strip()=="" and country.strip()=="" and
        (tier=="" or tier is None) and role.strip()=="" and
        semantic_query.strip()==""  
    ):
        st.session_state["search_results"] = None
        st.session_state["search_page"] = 0

    if st.button("Search Leads") and supabase:
        with st.spinner("Searching..."):

            semantic_query_clean = semantic_query.strip()

            if semantic_query_clean:
                if len(semantic_query_clean) < min_semantic_chars:
                    st.warning("Please provide a more detailed semantic query (8+ characters) for accurate matches.")
                    st.session_state["search_results"] = []
                else:
                    try:
                        emb = get_query_embedding_cached(semantic_query_clean)
                        rpc = supabase.rpc(
                            "rpc_semantic_search_leads_supplements",
                            {
                                "query_embedding": emb,
                                "match_count": semantic_max_matches,
                                "min_score": semantic_min_score,
                            },
                        ).execute()
                        rows = rpc.data or []

                        def extract_score(row):
                            for key in ("similarity", "score", "match_score", "distance"):
                                val = row.get(key)
                                if isinstance(val, (int, float)):
                                    return float(val)
                            return None

                        scored_ids = []
                        fallback_ids = []
                        for r in rows:
                            lid = r.get("lead_id")
                            if not lid:
                                continue
                            fallback_ids.append(str(lid))
                            score_val = extract_score(r)
                            if score_val is not None and score_val >= semantic_min_score:
                                scored_ids.append((str(lid), score_val))

                        if scored_ids:
                            scored_ids.sort(key=lambda x: x[1], reverse=True)
                            lead_ids = [lid for lid, _ in scored_ids][:semantic_max_matches]
                        else:
                            lead_ids = list(dict.fromkeys(fallback_ids[:20]))

                        if lead_ids:
                            res = (
                                supabase.table("leads")
                                .select("lead_id, full_name, email, tier, primary_role, city, country, notes")
                                .in_("lead_id", lead_ids)
                                .execute()
                            )
                            raw_results = res.data or []
                            score_lookup = {lid: score for lid, score in scored_ids}
                            filtered_results = [
                                r for r in raw_results if semantic_filter_passes(r)
                            ]
                            filtered_results.sort(
                                key=lambda r: score_lookup.get(str(r.get("lead_id")), 0),
                                reverse=True,
                            )
                            st.session_state["search_results"] = filtered_results
                            if not filtered_results:
                                st.info("No matches cleared the similarity threshold with the given filters. Try refining the query.")
                        else:
                            st.session_state["search_results"] = []
                            st.info("No semantic matches above the accuracy threshold. Try adding more detail or different terms.")
                    except Exception as e:
                        st.error(f"Semantic search error: {e}")
                        st.session_state["search_results"] = []

            else:
                try:
                    q = supabase.table("leads").select(
                        "lead_id, full_name, email, tier, primary_role, city, country, notes"
                    )
                    if keyword:
                        w = f"%{keyword}%"
                        q = q.or_(f"full_name.ilike.{w},email.ilike.{w},primary_role.ilike.{w}")
                    if city:
                        q = q.ilike("city", f"%{city}%")
                    if country:
                        q = q.ilike("country", f"%{country}%")
                    if tier:
                        q = q.eq("tier", tier)
                    if role:
                        q = q.ilike("primary_role", f"%{role}%")

                    res = q.limit(2000).execute()
                    st.session_state["search_results"] = res.data or []
                except:
                    st.session_state["search_results"] = []

        st.session_state["search_page"] = 0

    # Divider between search controls and chat
    st.markdown("---")

    # ------------------------------
    # Inline AI chat bar (no labels)
    # ------------------------------
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.warning("Set OPENAI_API_KEY to use the AI chat.")
    else:
        system_prompt_path = Path("prompts/system_prompt.md")
        if system_prompt_path.exists():
            system_prompt = system_prompt_path.read_text().strip()
        else:
            system_prompt = (
                "You are an art-market specialist AI. "
                "Use concise, factual reasoning and stay on collectors, artists, museums, and art-market dynamics."
            )

        client = OpenAI(api_key=api_key)

        if st.session_state.current_chat_open and supabase:
            st.session_state.chat_sessions = load_chat_sessions()

        if st.session_state.active_chat:
            st.markdown(
                """
                <div style="
                    border:1px solid #e0e0e0;
                    border-radius:12px;
                    padding:12px;
                    margin:12px 0 8px;
                    background-color:#fafafa;
                    overflow:hidden;">
                """,
                unsafe_allow_html=True,
            )

        for msg in st.session_state.active_chat:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style="
                        background-color:#f5f5f5;
                        padding:10px 14px;
                        border-radius:12px;
                        margin:6px 0;
                        text-align:right;
                        max-width:75%;
                        float:right;
                        clear:both;">
                        {msg["content"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="
                        background-color:#ffffff;
                        padding:10px 14px;
                        border-radius:12px;
                        margin:6px 0;
                        text-align:left;
                        max-width:75%;
                        float:left;
                        clear:both;">
                        {msg["content"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if st.session_state.active_chat:
            st.markdown("<div style='clear:both;'></div></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='clear:both;'></div>", unsafe_allow_html=True)

        if st.session_state.chat_sources:
            st.markdown(
                "###### Sources used",
                unsafe_allow_html=True,
            )
            for i, src in enumerate(st.session_state.chat_sources[:5]):
                name = src.get("full_name") or "Unknown"
                city = (src.get("city") or "").strip()
                country = (src.get("country") or "").strip()
                loc = ", ".join([p for p in [city, country] if p])
                snippet = (src.get("notes") or "").strip()
                if len(snippet) > 160:
                    snippet = snippet[:157] + "..."
                st.caption(f"{i+1}. {name}" + (f" — {loc}" if loc else ""))
                if snippet:
                    st.caption(f"   “{snippet}”")

        user_input = st.chat_input(placeholder="Ask about collectors, artists, or the art market...", key="collector_chat_bar")

        if user_input:
            st.session_state.active_chat.append(
                {"role": "user", "content": user_input}
            )
            st.session_state.chat_sources = []

            if st.session_state.current_chat_open and supabase:
                supabase.table("chat_messages").insert(
                    {
                        "session_id": st.session_state.current_chat_open,
                        "role": "user",
                        "content": user_input,
                    }
                ).execute()

            with st.spinner("Thinking..."):
                try:
                    context_text, used_chunks = build_context_from_rag(user_input)

                    messages = [
                        {
                            "role": "system",
                            "content": (
                                system_prompt
                                + " Use only the provided database context to answer. "
                                "If no relevant context is available, say you could not find matching collectors."
                            ),
                        }
                    ]

                    if context_text:
                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    "Database context from ai.rag_chunks and public.leads:\n"
                                    f"{context_text}"
                                ),
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    "No database context found for this query. "
                                    "Respond that no matching collectors were found in the database."
                                ),
                            }
                        )

                    messages.extend(st.session_state.active_chat)

                    if not context_text:
                        response_text = (
                            "I couldn't find matching collectors in the database for that query. "
                            "Try refining names, artists, or locations."
                        )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages,
                            temperature=0.2,
                            max_tokens=600,
                        )

                        response_text = completion.choices[0].message.content.strip()

                    st.session_state.active_chat.append(
                        {"role": "assistant", "content": response_text}
                    )
                    st.session_state.chat_sources = used_chunks

                    if st.session_state.current_chat_open and supabase:
                        supabase.table("chat_messages").insert(
                    {
                        "session_id": st.session_state.current_chat_open,
                        "role": "assistant",
                        "content": response_text,
                        }
                    ).execute()

                    st.rerun()
                except Exception as e:
                    st.error(f"Chat failed: {e}")

        if st.session_state.active_chat:
            if st.button("New Chat", use_container_width=True, key="collector_chat_new"):
                st.session_state.active_chat = []
                st.session_state.current_chat_open = None
                st.session_state.current_session_summary = ""
                st.session_state.current_session_title = ""
                st.rerun()

    # Divider between chat and results grid
    st.markdown("---")

    # ==========================================
    # SHOW SEARCH OR FULL GRID
    # ==========================================
    search_results = st.session_state.get("search_results", None)
    show_search_grid = search_results is not None

    # ------------------------------
    # FULL GRID (NO SEARCH APPLIED)
    # ------------------------------
    if not show_search_grid and supabase:
        per_page = 50
        if "full_grid_page" not in st.session_state:
            st.session_state.full_grid_page = 0

        total_full = get_total_leads_count()
        total_pages = max(1, (total_full + per_page - 1) // per_page)

        leads = get_full_grid_page(st.session_state.full_grid_page, per_page)
        prefetch_page(get_full_grid_page, st.session_state.full_grid_page, per_page)

        st.write(f"Showing {len(leads)} of {total_full} collectors")

        if leads:
            df = pd.DataFrame(leads)
            if not df.empty:
                df["location"] = df.apply(
                    lambda r: ", ".join(
                        [p for p in [str(r.get("city") or "").strip(), str(r.get("country") or "").strip()] if p]
                    ),
                    axis=1,
                )
                cols = [
                    "full_name",
                    "location",
                    "primary_role",
                    "tier",
                    "email",
                    "notes",
                ]
                existing_cols = [c for c in cols if c in df.columns]
                df_display = df[existing_cols].copy()
                if "notes" in df_display.columns:
                    df_display["notes"] = df_display["notes"].fillna("").apply(
                        lambda x: (x[:160] + "…") if len(x) > 160 else x
                    )
                st.dataframe(
                    df_display,
                    use_container_width=True,
                    height=520,
                )

            lead_options = [str(r.get("lead_id")) for r in leads if r.get("lead_id")]
            name_lookup = {str(r.get("lead_id")): r.get("full_name", "Unnamed") for r in leads}
            default_id = st.session_state.get("active_full_lead") or (lead_options[0] if lead_options else None)
            if lead_options and default_id not in lead_options:
                default_id = lead_options[0]

            selected_lead_id = st.selectbox(
                "Open collector",
                options=lead_options,
                index=lead_options.index(default_id) if lead_options and default_id else 0,
                format_func=lambda lid: name_lookup.get(lid, lid),
                key="full_detail_selector",
            ) if lead_options else None

            if selected_lead_id:
                st.session_state["active_full_lead"] = selected_lead_id
                selected_lead = next(
                    (r for r in leads if str(r.get("lead_id")) == selected_lead_id),
                    None,
                )
                if selected_lead:
                    with st.container():
                        render_lead_detail(selected_lead, summary_prefix="full")

        col_space_left, prev_col, next_col, col_space_right = st.columns([2, 1, 1, 2])
        with prev_col:
            if st.button("Prev Page", disabled=st.session_state.full_grid_page == 0):
                st.session_state.full_grid_page -= 1
                st.rerun()
        with next_col:
            if st.button(
                "Next Page",
                disabled=st.session_state.full_grid_page >= total_pages - 1,
            ):
                st.session_state.full_grid_page += 1
                st.rerun()

    # ------------------------------
    # SEARCH GRID (HAS RESULTS)
    # ------------------------------
    elif show_search_grid:
        results = search_results or []
        per_page = 50

        if "search_page" not in st.session_state:
            st.session_state.search_page = 0

        total_results = len(results)
        total_pages = max(1, (total_results + per_page - 1) // per_page)

        start = st.session_state.search_page * per_page
        end = start + per_page
        page_results = results[start:end]

        st.write(f"Showing {len(page_results)} of {total_results} results")

        if page_results:
            df = pd.DataFrame(page_results)
            if not df.empty:
                df["location"] = df.apply(
                    lambda r: ", ".join(
                        [p for p in [str(r.get("city") or "").strip(), str(r.get("country") or "").strip()] if p]
                    ),
                    axis=1,
                )
                display_cols = [
                    "full_name",
                    "location",
                    "primary_role",
                    "tier",
                    "email",
                    "notes",
                ]
                existing_cols = [c for c in display_cols if c in df.columns]
                df_display = df[existing_cols].copy()
                if "notes" in df_display.columns:
                    df_display["notes"] = df_display["notes"].fillna("").apply(
                        lambda x: (x[:160] + "…") if len(x) > 160 else x
                    )
                st.dataframe(
                    df_display,
                    use_container_width=True,
                    height=520,
                )

            lead_options = [str(r.get("lead_id")) for r in page_results if r.get("lead_id")]
            name_lookup = {
                str(r.get("lead_id")): r.get("full_name", "Unnamed") for r in page_results
            }
            default_id = st.session_state.get("active_search_lead") or (lead_options[0] if lead_options else None)
            if lead_options and default_id not in lead_options:
                default_id = lead_options[0]

            selected_lead_id = st.selectbox(
                "Open collector",
                options=lead_options,
                index=lead_options.index(default_id) if lead_options and default_id else 0,
                format_func=lambda lid: name_lookup.get(lid, lid),
                key="search_detail_selector",
            ) if lead_options else None

            if selected_lead_id:
                st.session_state["active_search_lead"] = selected_lead_id
                selected_lead = next(
                    (r for r in page_results if str(r.get("lead_id")) == selected_lead_id),
                    None,
                )
                if selected_lead:
                    with st.container():
                        render_lead_detail(selected_lead, summary_prefix="search")
        else:
            st.info("No results on this page.")

        prev_col, next_col = st.columns([1, 1])
        with prev_col:
            if st.button("Prev Results", disabled=st.session_state.search_page == 0):
                st.session_state.search_page -= 1
                st.rerun()
        with next_col:
            if st.button(
                "Next Results",
                disabled=st.session_state.search_page >= total_pages - 1,
            ):
                st.session_state.search_page += 1
                st.rerun()

# ============================================================
# CONTACTS TAB
# ============================================================
with tabs[1]:
    st.markdown("## Contacts")

    # --- Create a new contact form ---
    with st.expander("Create a Contact", expanded=False):
        with st.form("create_contact_form"):
            st.markdown("Enter contact details to add a new record to the leads table:")

            full_name = st.text_input("Full Name")
            email = st.text_input("Email")
            primary_role = st.text_input("Primary Role")
            city = st.text_input("City")
            country = st.text_input("Country")
            tier = st.selectbox("Tier", ["A", "B", "C", "—"], index=3)
            notes = st.text_area("Notes", height=100)

            submitted = st.form_submit_button("Create Contact")
            if submitted:
                if not full_name or not email:
                    st.warning("Please provide at least a name and email.")
                else:
                    try:
                        response = (
                            supabase.table("leads")
                            .insert(
                                {
                                    "full_name": full_name.strip(),
                                    "email": email.strip(),
                                    "primary_role": primary_role.strip()
                                    if primary_role
                                    else None,
                                    "city": city.strip() if city else None,
                                    "country": country.strip() if country else None,
                                    "tier": None if tier == "—" else tier,
                                    "notes": notes.strip() if notes else None,
                                }
                            )
                            .execute()
                        )
                        if getattr(response, "status_code", 400) < 300:
                            get_total_leads_count.clear()
                            get_contacts_page.clear()
                            get_full_grid_page.clear()
                            st.success(f"{full_name} has been added to your contacts.")
                            st.rerun()
                        else:
                            st.error(f"Insert failed: {response}")
                    except Exception as e:
                        st.error(f"Error creating contact: {e}")

    # --- Spacing before filter + table ---
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    if not supabase:
        st.warning("Database unavailable.")
    else:
        # --------------------------
        # Filter + Export controls
        # --------------------------
        st.markdown("### Filter Contacts")

        filter_text = st.text_input(
            "Search contacts by any visible field",
            placeholder="e.g. Nauman, Minimalism, France, curator…",
            key="contacts_filter_text",
        )

        export_btn = st.button("Export Filtered Results as CSV", key="contacts_export_btn")

        def filter_dataframe(df: pd.DataFrame, text: str) -> pd.DataFrame:
            if not text or not text.strip():
                return df
            text = text.lower()
            return df[
                df.apply(
                    lambda row: any(
                        text in str(row[col]).lower() for col in df.columns
                    ),
                    axis=1,
                )
            ]

        # --------------------------
        # Pagination + data fetch
        # --------------------------
        per_page = 200
        if "data_page" not in st.session_state:
            st.session_state.data_page = 0

        total_count = get_total_leads_count()
        total_pages = max(1, (total_count + per_page - 1) // per_page)
        st.caption(
            f"Page {st.session_state.data_page + 1} of {total_pages} — {total_count} total leads"
        )

        leads = get_contacts_page(st.session_state.data_page, per_page)
        prefetch_page(get_contacts_page, st.session_state.data_page, per_page)

        if leads:
            df = pd.DataFrame(leads)
            desired_cols = [
                "full_name",
                "email",
                "tier",
                "primary_role",
                "city",
                "country",
                "notes",
            ]
            existing_cols = [c for c in desired_cols if c in df.columns]
            df = df[existing_cols]

            # Apply filter
            filtered_df = filter_dataframe(df, filter_text)

            # Show table
            st.dataframe(filtered_df, use_container_width=True)

            # Export CSV for filtered rows
            if export_btn:
                if not filtered_df.empty:
                    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download filtered contacts as CSV",
                        data=csv_bytes,
                        file_name="filtered_contacts.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("No rows to export for this filter.")

            # Pagination controls
            st.markdown("---")
            col_space_left, col_prev, col_next, col_space_right = st.columns(
                [2, 1, 1, 2]
            )
            with col_prev:
                if st.button(
                    "Previous",
                    use_container_width=True,
                    disabled=st.session_state.data_page == 0,
                ):
                    st.session_state.data_page -= 1
                    st.rerun()
            with col_next:
                if st.button(
                    "Next",
                    use_container_width=True,
                    disabled=st.session_state.data_page >= total_pages - 1,
                ):
                    st.session_state.data_page += 1
                    st.rerun()
        else:
            st.info("No leads found.")

# ============================================================
# SAVED SETS TAB
# ============================================================
with tabs[2]:
    st.markdown("## Saved Sets")
    if not supabase:
        st.warning("Database unavailable.")
    else:
        sets = (
            supabase.table("saved_sets")
            .select("id, name, description, created_at")
            .order("created_at", desc=True)
            .execute()
            .data
            or []
        )
        if not sets:
            st.info("No saved sets yet.")
        else:
            for s in sets:
                with st.expander(f"{s['name']}"):
                    st.write(f"**Description:** {s.get('description', '—')}")
                    st.write(f"**Created:** {s.get('created_at', '—')}")
