import streamlit as st
from supabase_client import get_supabase
from components import inject_css
import os
from openai import OpenAI
from datetime import datetime
from pathlib import Path
import pandas as pd
import re
import requests
from services.rag import answer_with_context


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
if 'email_searches' not in st.session_state:
    st.session_state.email_searches = []
if 'verified_emails' not in st.session_state:
    st.session_state.verified_emails = []

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

# Helper functions for Research tab
def extract_domain(url):
    """Extract domain from URL"""
    pattern = r'(?:https?://)?(?:www\.)?([^/]+)'
    match = re.search(pattern, url)
    return match.group(1) if match else url

def generate_email_patterns(domain, first_name, last_name):
    """Generate common email patterns"""
    patterns = [
        f"{first_name.lower()}.{last_name.lower()}@{domain}",
        f"{first_name.lower()}{last_name.lower()}@{domain}",
        f"{first_name[0].lower()}{last_name.lower()}@{domain}",
        f"{first_name.lower()}@{domain}",
        f"{last_name.lower()}@{domain}",
        f"{first_name[0].lower()}.{last_name.lower()}@{domain}",
    ]
    return patterns

tabs = st.tabs(["Search", "Contacts", "Saved Sets", "Research"])

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

    st.markdown("<hr>", unsafe_allow_html=True)

    # ------------------------------
    # Inline AI chat bar (no labels)
    # ------------------------------
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.warning("Set OPENAI_API_KEY to use the AI chat.")
    else:
        # --- One-time CSS for chat alignment + removing heavy chat divider look ---
        if "collector_chat_css" not in st.session_state:
            st.session_state.collector_chat_css = True
            st.markdown(
                """
                <style>
                /* Soften / remove the heavy chat-input "divider" look */
                div[data-testid="stChatInput"] > div {
                    box-shadow: none !important;
                    border: 1px solid #e0e0e0 !important;
                    border-radius: 8px !important;
                    background: #ffffff !important;
                }
                /* Optional: tighten the vertical gap above the chat input */
                div[data-testid="stChatInput"] {
                    margin-top: 0.5rem !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
    
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
    
        # ------------------------------
        # Chat history display
        # ------------------------------
        if st.session_state.active_chat:
            # Outer card wrapper
            st.markdown(
                """
                <div style="
                    border:1px solid #e0e0e0;
                    border-radius:12px;
                    padding:12px;
                    margin:12px 0 8px;
                    background-color:#fafafa;">
                """,
                unsafe_allow_html=True,
            )
    
            # Each message as a flex row so alignment is consistent
            for msg in st.session_state.active_chat:
                is_user = msg["role"] == "user"
                justify = "flex-end" if is_user else "flex-start"
                bg = "#f5f5f5" if is_user else "#ffffff"
                align = "right" if is_user else "left"
    
                st.markdown(
                    f"""
                    <div style="
                        display:flex;
                        justify-content:{justify};
                        margin:6px 0;">
                        <div style="
                            background-color:{bg};
                            padding:10px 14px;
                            border-radius:12px;
                            max-width:75%;
                            border:1px solid #e0e0e0;
                            text-align:{align};
                            font-size:0.95rem;
                            line-height:1.5;">
                            {msg["content"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    
            # Close outer wrapper
            st.markdown("</div>", unsafe_allow_html=True)
    
        else:
            # No chat yet – just make sure layout is cleared
            st.markdown("<div style='clear:both;'></div>", unsafe_allow_html=True)
    
        # ------------------------------
        # Sources used (only if there is an active chat)
        # ------------------------------
        if st.session_state.active_chat and st.session_state.chat_sources:
            st.markdown("###### Sources used", unsafe_allow_html=True)
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
    
        # ------------------------------
        # Chat input
        # ------------------------------
        user_input = st.chat_input(
            placeholder="Ask about collectors, artists, or the art market...",
            key="collector_chat_bar",
        )
    
        if user_input:
            st.session_state.active_chat.append(
                {"role": "user", "content": user_input}
            )
            # Clear sources for the new turn (you already had this line)
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
                    # Use unified RAG pipeline
                    result = answer_with_context(
                        supabase=supabase,
                        question=user_input,
                        system_prompt=system_prompt,
                        match_count=20,
                        min_similarity=0.10,
                    )
    
                    response_text = result.get("answer", "").strip() or (
                        "I couldn't find matching collectors in the database for that query. "
                        "Try refining names, artists, or locations."
                    )
                    used_chunks = result.get("sources", []) or []
    
                    # Append assistant reply to chat state
                    st.session_state.active_chat.append(
                        {"role": "assistant", "content": response_text}
                    )
                    st.session_state.chat_sources = used_chunks
    
                    # Persist assistant message if a DB chat session is open
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
    
        # ------------------------------
        # New Chat button
        # ------------------------------
        if st.session_state.active_chat:
            if st.button("New Chat", use_container_width=True, key="collector_chat_new"):
                st.session_state.active_chat = []
                st.session_state.chat_sources = []          # ← clear sources too
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
    
        st.write(f"Showing {len(leads)} of {total_full} collectors")
    
        left_col, right_col = st.columns(2)
    
        for i, lead in enumerate(leads):
            col = left_col if i % 2 == 0 else right_col
    
            name = lead.get("full_name", "Unnamed")
            city_val = (lead.get("city") or "").strip()
            label = f"{name} — {city_val}" if city_val else name
            lead_id = str(lead.get("lead_id"))
    
            with col:
                expander_key = f"expander_full_{lead_id}"
                with st.expander(label, key=expander_key):
            
                    tier_val = lead.get("tier", "—")
                    role_val = lead.get("primary_role", "—")
                    email_val = lead.get("email", "—")
                    country_val = (lead.get("country") or "").strip()
            
                    if city_val or country_val:
                        st.caption(f"{city_val}, {country_val}".strip(", "))
                    st.caption(f"{role_val} | Tier {tier_val}")
                    st.write(email_val)

                    sum_col, _ = st.columns([3, 1])
                    summary_key = f"summary_{lead_id}"
    
                    with sum_col:
                        if summary_key not in st.session_state:
                            if st.button(f"Summarize {name}", key=f"sum_full_{lead_id}"):
                                with st.spinner("Summarizing notes..."):
    
                                    # -----------------------
                                    # Research table fetch
                                    # -----------------------
                                    supplements = (
                                        supabase.table("research")
                                        .select("content_text, source_url, source_title, created_at")
                                        .eq("lead_id", lead_id)
                                        .execute()
                                        .data
                                        or []
                                    )
    
                                    # Base notes: if your leads table has its own stored notes
                                    # (otherwise set base_notes = "")
                                    base_notes = lead.get("content_text") or ""
    
                                    # Supplement notes: now correctly referencing "content_text"
                                    supplement_notes = "\n\n".join(
                                        (s.get("content_text") or "").strip()
                                        for s in supplements
                                    )
    
                                    combined = (
                                        base_notes
                                        + ("\n\n" if base_notes and supplement_notes else "")
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
                                            {"role": "system", "content": "You summarize art collectors factually."},
                                            {"role": "user", "content": prompt},
                                        ],
                                        temperature=0.2,
                                        max_tokens=500,
                                    )
    
                                    st.session_state[summary_key] = resp.choices[0].message.content.strip()
    
                        else:
                            st.markdown("**Summary:**")
                            st.markdown(st.session_state[summary_key], unsafe_allow_html=True)
    
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

        left_col, right_col = st.columns(2)

        for i, lead in enumerate(page_results):
            col = left_col if i % 2 == 0 else right_col
            lead_id = str(lead.get("lead_id"))

            name = lead.get("full_name", "Unnamed")
            city_val = (lead.get("city") or "").strip()
            label = f"{name} — {city_val}" if city_val else name

            with col:
                expander_key = f"expander_{lead_id}"
                with st.expander(label, key=expander_key):
                    st.markdown(f"**{name}**")
                    st.caption(f"{lead.get('primary_role', '—')} | Tier {lead.get('tier', '—')}")
                    st.write(lead.get("email", "—"))

                    sum_col, _ = st.columns([3, 1])
                    summary_key = f"summary_{lead_id}"

                    with sum_col:
                        if summary_key not in st.session_state:
                            if st.button(f"Summarize {name}", key=f"sum_search_{lead_id}"):
                                with st.spinner("Summarizing notes..."):
                                    supplements = (
                                        supabase.table("leads_supplements")
                                        .select("notes")
                                        .eq("lead_id", lead_id)
                                        .execute()
                                        .data
                                        or []
                                    )

                                    base_notes = lead.get("notes") or ""
                                    supplement_notes = "\n\n".join(
                                        (s.get("notes") or "").strip()
                                        for s in supplements
                                    )

                                    combined = (
                                        base_notes
                                        + ("\n\n" if base_notes and supplement_notes else "")
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
                                            {"role": "system", "content": "You summarize art collectors factually."},
                                            {"role": "user", "content": prompt},
                                        ],
                                        temperature=0.2,
                                        max_tokens=500,
                                    )

                                    st.session_state[summary_key] = (
                                        resp.choices[0].message.content.strip()
                                    )
                        else:
                            st.markdown("**Summary:**")
                            st.markdown(st.session_state[summary_key], unsafe_allow_html=True)

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
            .select("*")
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

# ============================================================
# RESEARCH TAB - WITH REAL HUNTER.IO API INTEGRATION
# ============================================================
with tabs[3]:
    st.markdown("## Research")
    
    # Hide warning and info banners
    st.markdown("""
    <style>
    div[data-testid="stAlert"] {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check for Hunter.io API key
    hunter_api_key = os.getenv("HUNTER_API_KEY")
    
    if not supabase:
        st.warning("Database unavailable.")
    else:
        # Tool selector
        tool_option = st.radio(
            "Select Tool",
            ["Domain Search", "Email Finder", "Email Verifier"],
            horizontal=True
        )
        
        st.markdown("---")
        
        # Domain Search Tool
        if tool_option == "Domain Search":
            st.markdown("### Domain Search")
            st.caption("Find all email addresses associated with a domain using Hunter.io API")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                domain_input = st.text_input("Domain", placeholder="stripe.com", label_visibility="collapsed", key="domain_search_input")
            with col2:
                search_button = st.button("Search", type="primary", use_container_width=True, key="domain_search_btn")
            
            if search_button and domain_input:
                domain = extract_domain(domain_input)
                st.session_state.email_searches.append({
                    "domain": domain, 
                    "timestamp": datetime.now()
                })
                
                if hunter_api_key:
                    with st.spinner("Searching Hunter.io..."):
                        try:
                            # Call Hunter.io Domain Search API
                            url = f"https://api.hunter.io/v2/domain-search?domain={domain}&api_key={hunter_api_key}"
                            response = requests.get(url)
                            
                            if response.status_code == 200:
                                data = response.json()
                                emails = data.get('data', {}).get('emails', [])
                                
                                if emails:
                                    st.write(f"Found {len(emails)} email addresses for {domain}")
                                    
                                    left_col, right_col = st.columns(2)
                                    
                                    for i, email_data in enumerate(emails):
                                        col = left_col if i % 2 == 0 else right_col
                                        
                                        email = email_data.get('value', 'Unknown')
                                        first_name = email_data.get('first_name', '')
                                        last_name = email_data.get('last_name', '')
                                        name = f"{first_name} {last_name}".strip() or "Unknown"
                                        position = email_data.get('position', 'Position not listed')
                                        confidence = email_data.get('confidence', 0)
                                        
                                        with col:
                                            with st.expander(f"{name} ({confidence}% confidence)"):
                                                st.markdown(f"**{name}**")
                                                st.caption(position)
                                                st.code(email, language=None)
                                                
                                                # Confidence score visualization
                                                if confidence >= 90:
                                                    st.success(f"Confidence: {confidence}%")
                                                elif confidence >= 70:
                                                    st.info(f"Confidence: {confidence}%")
                                                else:
                                                    st.warning(f"Confidence: {confidence}%")
                                                
                                                # Show sources
                                                sources = email_data.get('sources', [])
                                                if sources:
                                                    st.caption(f"Found on {len(sources)} source(s)")
                                                    for source in sources[:3]:
                                                        st.caption(f"• {source.get('domain', 'Unknown')}")
                                else:
                                    st.info(f"No email addresses found for {domain} in Hunter.io database")
                            elif response.status_code == 401:
                                st.error("Invalid API key. Please check your HUNTER_API_KEY")
                            elif response.status_code == 429:
                                st.error("Rate limit exceeded. Please wait a moment and try again")
                            else:
                                st.error(f"API error: {response.status_code} - {response.text}")
                        
                        except Exception as e:
                            st.error(f"Error calling Hunter.io API: {e}")
                else:
                    st.warning("Please set HUNTER_API_KEY to use this feature")
        
        # Email Finder Tool
        elif tool_option == "Email Finder":
            st.markdown("### Email Finder")
            st.caption("Find someone's email address using their name and company domain")
            
            col1, col2 = st.columns(2)
            with col1:
                first_name = st.text_input("First Name", placeholder="Patrick", key="finder_first")
                domain = st.text_input("Company Domain", placeholder="stripe.com", key="finder_domain")
            with col2:
                last_name = st.text_input("Last Name", placeholder="Collison", key="finder_last")
                
            find_button = st.button("Find Email", type="primary", use_container_width=True, key="finder_btn")
            
            if find_button and first_name and last_name and domain:
                domain = extract_domain(domain)
                
                if hunter_api_key:
                    with st.spinner("Finding email with Hunter.io..."):
                        try:
                            # Call Hunter.io Email Finder API
                            url = f"https://api.hunter.io/v2/email-finder?domain={domain}&first_name={first_name}&last_name={last_name}&api_key={hunter_api_key}"
                            response = requests.get(url)
                            
                            if response.status_code == 200:
                                data = response.json()
                                email_data = data.get('data', {})
                                
                                email = email_data.get('email')
                                score = email_data.get('score', 0)
                                position = email_data.get('position')
                                
                                if email:
                                    st.markdown("### Found Email")
                                    
                                    with st.expander(f"{first_name} {last_name}", expanded=True):
                                        st.code(email, language=None)
                                        
                                        if position:
                                            st.caption(f"Position: {position}")
                                        
                                        # Confidence visualization
                                        st.markdown(f"**Confidence Score: {score}%**")
                                        st.progress(score / 100)
                                        
                                        if score >= 90:
                                            st.success("Very high confidence")
                                        elif score >= 70:
                                            st.info("Good confidence")
                                        elif score >= 50:
                                            st.warning("Moderate confidence")
                                        else:
                                            st.error("Low confidence")
                                        
                                        # Show sources
                                        sources = email_data.get('sources', [])
                                        if sources:
                                            st.markdown("**Sources:**")
                                            for source in sources[:5]:
                                                st.caption(f"• {source.get('uri', 'Unknown source')}")
                                else:
                                    st.info("No email found with high confidence. Try different variations of the name.")
                            
                            elif response.status_code == 401:
                                st.error("Invalid API key. Please check your HUNTER_API_KEY")
                            elif response.status_code == 429:
                                st.error("Rate limit exceeded. Please wait a moment and try again")
                            else:
                                st.error(f"API error: {response.status_code}")
                        
                        except Exception as e:
                            st.error(f"Error calling Hunter.io API: {e}")
                else:
                    st.warning("Please set HUNTER_API_KEY to use this feature")
        
        # Email Verifier Tool
        else:  # tool_option == "Email Verifier"
            st.markdown("### Email Verifier")
            st.caption("Verify if an email address is valid and deliverable using Hunter.io")
            
            email_to_verify = st.text_input("Email Address", placeholder="patrick@stripe.com", key="verify_email")
            verify_button = st.button("Verify Email", type="primary", use_container_width=True, key="verify_btn")
            
            if verify_button and email_to_verify:
                if hunter_api_key:
                    with st.spinner("Verifying email with Hunter.io..."):
                        try:
                            # Call Hunter.io Email Verifier API
                            url = f"https://api.hunter.io/v2/email-verifier?email={email_to_verify}&api_key={hunter_api_key}"
                            response = requests.get(url)
                            
                            if response.status_code == 200:
                                data = response.json()
                                result_data = data.get('data', {})
                                
                                status = result_data.get('status', 'unknown')
                                score = result_data.get('score', 0)
                                
                                st.session_state.verified_emails.append({
                                    "email": email_to_verify,
                                    "status": status,
                                    "score": score,
                                    "timestamp": datetime.now()
                                })
                                
                                st.markdown("### Verification Results")
                                
                                with st.expander(email_to_verify, expanded=True):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        # Status display
                                        if status == "valid":
                                            st.success(f"Status: {status.upper()}")
                                        elif status == "risky":
                                            st.warning(f"Status: {status.upper()}")
                                        else:
                                            st.error(f"Status: {status.upper()}")
                                        
                                        # Score
                                        st.markdown(f"**Deliverability Score: {score}%**")
                                        st.progress(score / 100)
                                    
                                    with col2:
                                        st.caption("Verification Details")
                                        checks = [
                                            ("Format", result_data.get('regexp', False)),
                                            ("MX Records", result_data.get('mx_records', False)),
                                            ("SMTP Server", result_data.get('smtp_server', False)),
                                            ("SMTP Check", result_data.get('smtp_check', False)),
                                            ("Not Disposable", not result_data.get('disposable', True)),
                                            ("Not Gibberish", not result_data.get('gibberish', True))
                                        ]
                                        
                                        for check_name, passed in checks:
                                            icon = "✅" if passed else "❌"
                                            st.write(f"{icon} {check_name}")
                                
                                # Show sources if found
                                sources = result_data.get('sources', [])
                                if sources:
                                    st.markdown("**Email found on:**")
                                    for source in sources[:5]:
                                        st.caption(f"• {source.get('domain', 'Unknown')}")
                            
                            elif response.status_code == 401:
                                st.error("Invalid API key. Please check your HUNTER_API_KEY")
                            elif response.status_code == 429:
                                st.error("Rate limit exceeded. Please wait a moment and try again")
                            else:
                                st.error(f"API error: {response.status_code}")
                        
                        except Exception as e:
                            st.error(f"Error calling Hunter.io API: {e}")
                else:
                    st.warning("Please set HUNTER_API_KEY to use this feature")
        

