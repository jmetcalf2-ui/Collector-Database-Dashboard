import streamlit as st
from supabase_client import get_supabase
from components import inject_css
import os
from openai import OpenAI
from datetime import datetime
from pathlib import Path

# --- Page setup ---
st.set_page_config(page_title="Dashboard", layout="wide")
inject_css()

# --- Full-width but centered layout ---
st.markdown("""
<style>
/* Left-align all buttons (chat history) */
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

/* Subtle hover effect */
div[data-testid="stButton"] > button:hover {
    background-color: #f1f1f1 !important;
    border-color: #ccc !important;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.write(" ")

# --- Initialize session state ---
if "selected_leads" not in st.session_state:
    st.session_state.selected_leads = []
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []
if "active_chat" not in st.session_state:
    st.session_state.active_chat = []

# --- Global style ---
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

# --- Main content ---
st.markdown("<h1>Dashboard</h1>", unsafe_allow_html=True)

# --- Connect Supabase (silent connect, no banner) ---
try:
    supabase = get_supabase()
except Exception as e:
    st.error(f"⚠️ Supabase connection failed: {e}")
    supabase = None


# --- Tabs ---
tabs = st.tabs(["Search", "Data", "Saved Sets", "Chat"])

# --- Cached AI summarization helper ---
@st.cache_data(show_spinner=False)
def summarize_collector(lead_id: str, combined_notes: str) -> str:
    """
    Summarizes collector intelligence notes into bullet points using OpenAI.
    Prints debug info if key or data missing.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return "⚠️ Missing OPENAI_API_KEY — add it to your environment."

    if not combined_notes.strip():
        return "⚠️ No notes found for this lead."

    try:
        client = OpenAI(api_key=key)
        prompt = f"""
        You are an expert art-market researcher creating collector intelligence summaries.
        Write 4–6 short bullet points summarizing this collector's data factually.
        Focus on specifics like:
        - Artists collected or recently purchased
        - Museum/institutional boards or affiliations
        - Geography (city or region)
        - Collecting tendencies or philanthropy
        - Notable sales, acquisitions, or foundations
        Avoid adjectives like 'important' or 'renowned'.
        Example: 'Collects Glenn Ligon and Kara Walker; MoMA trustee; founded Art for Justice Fund.'

        NOTES:
        {combined_notes}
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize art collectors factually and concisely."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"

# ======================================================================
# === SEARCH TAB ===
# ======================================================================
with tabs[0]:
    st.markdown("## Search")

    st.markdown("""
    <style>
    input[data-testid="stTextInput"]::placeholder {
        color: #888 !important;
        opacity: 1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Unified Search Layout ---
    col1, col2, col3, col4, col5 = st.columns([2.2, 1.2, 1.2, 1.2, 1.2])
    with col1:
        keyword = st.text_input("Keyword", placeholder="Name, email, interests, etc.")
    with col2:
        city = st.text_input("City")
    with col3:
        country = st.text_input("Country")
    with col4:
        tier = st.selectbox("Tier", ["", "A", "B", "C"], index=0)
    with col5:
        role = st.text_input("Primary Role")

    # --- Semantic Search Field (inline, cohesive look) ---
    semantic_query = st.text_input(
        "Semantic Search",
        placeholder="e.g. Minimalism collectors or those following Bruce Nauman",
    )

    # --- Button ---
    if st.button("Search Leads") and supabase:
        with st.spinner("Searching..."):
            results = []

            if semantic_query.strip():
                # --- Run semantic search if semantic query entered ---
                try:
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    emb = client.embeddings.create(
                        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
                        input=semantic_query,
                    ).data[0].embedding

                    res = supabase.rpc(
                        "rpc_semantic_search_leads_supplements",
                        {
                            "query_embedding": list(map(float, emb)),
                            "match_count": 25,
                            "min_score": 0.15,
                        },
                    ).execute()

                    results = res.data or []
                    st.caption("Showing semantic matches")
                except Exception as e:
                    st.error(f"Semantic search failed: {e}")
            else:
                # --- Fallback to regular search ---
                query = supabase.table("leads").select("*")
                if keyword:
                    query = query.ilike("full_name", f"%{keyword}%")
                if city:
                    query = query.ilike("city", f"%{city}%")
                if country:
                    query = query.ilike("country", f"%{country}%")
                if tier:
                    query = query.eq("tier", tier)
                if role:
                    query = query.ilike("primary_role", f"%{role}%")
                results = query.limit(100).execute().data or []

        # --- Display results (shared logic) ---
        if results:
            st.success(f"Found {len(results)} results")
            for lead in results:
                with st.expander(f"{lead.get('full_name', 'Unnamed')}{' — ' + lead['city'] if lead.get('city') else ''}"):
                    st.write(f"**Email:** {lead.get('email','—')}")
                    st.write(f"**Tier:** {lead.get('tier','—')}")
                    st.write(f"**Role:** {lead.get('primary_role','—')}")

                    try:
                        lead_pk = lead.get("lead_id") or lead.get("id")
                        if not lead_pk:
                            continue
                        supplements = (
                            supabase.table("leads_supplements")
                            .select("notes")
                            .eq("lead_id", str(lead_pk))
                            .execute()
                            .data
                            or []
                        )

                        base_notes = lead.get("notes") or ""
                        supplement_notes = "\n\n".join(
                            (s.get("notes") or "").strip() for s in supplements if isinstance(s, dict)
                        )
                        combined_notes = (
                            base_notes
                            + ("\n\n" if base_notes and supplement_notes else "")
                            + supplement_notes
                        ).strip()

                        summary = summarize_collector(str(lead_pk), combined_notes)
                        st.markdown("**Notes:**")
                        st.markdown(summary, unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown("**Notes:**")
                        st.write(f"⚠️ Failed to summarize: {e}")
                        st.write((lead.get("notes") or "")[:600])
        else:
            st.info("No leads found.")

# ======================================================================
# === DATA TAB ===
# ======================================================================
with tabs[1]:
    st.markdown("## Data Overview")

    # --- Pagination setup ---
    per_page = 25
    if "data_page" not in st.session_state:
        st.session_state.data_page = 0

    offset = st.session_state.data_page * per_page

    # --- Get total count safely (no ID column required) ---
    try:
        total_response = supabase.table("leads").select("*", count="exact").limit(1).execute()
        total_count = getattr(total_response, "count", None) or 0
    except Exception as e:
        st.error(f"Could not fetch total lead count: {e}")
        total_count = 0

    total_pages = max(1, (total_count + per_page - 1) // per_page)

    # --- Fetch paginated data ---
    try:
        data_response = (
            supabase.table("leads")
            .select("full_name, email, city, country, tier, primary_role, created_at")
            .order("created_at", desc=True)
            .range(offset, offset + per_page - 1)
            .execute()
        )
        data = getattr(data_response, "data", []) or []
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        data = []

    # --- Display results ---
    if data:
        st.dataframe(
            data,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(f"Page {st.session_state.data_page + 1} of {total_pages} — {total_count} total leads")

        col_prev, col_next = st.columns([1, 1])
        with col_prev:
            if st.button("Previous", disabled=st.session_state.data_page == 0):
                st.session_state.data_page -= 1
                st.rerun()
        with col_next:
            if st.button("Next", disabled=st.session_state.data_page >= total_pages - 1):
                st.session_state.data_page += 1
                st.rerun()
    else:
        st.info("No data found.")

# ======================================================================
# === SAVED SETS TAB ===
# ======================================================================
with tabs[2]:
    st.markdown("## Saved Sets")
    if not supabase:
        st.warning("Database unavailable.")
    else:
        sets = supabase.table("saved_sets").select("*").order("created_at", desc=True).execute().data or []
        if not sets:
            st.info("No saved sets yet.")
        else:
            for s in sets:
                with st.expander(f"{s['name']}"):
                    st.write(f"**Description:** {s.get('description', '—')}")
                    st.write(f"**Created:** {s.get('created_at', '—')}")

# ======================================================================
# === CHAT TAB ===
# ======================================================================
with tabs[3]:

    # --- System prompt setup ---
    system_prompt_path = Path("prompts/system_prompt.md")
    if system_prompt_path.exists():
        system_prompt = system_prompt_path.read_text().strip()
    else:
        system_prompt = (
            "You are CollectorGPT — a helpful art-market assistant. "
            "You answer questions conversationally, referencing collectors, artists, galleries, and market trends when relevant. "
            "Keep responses factual, concise, and well-reasoned."
        )

    # --- Initialize OpenAI client ---
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # --- Layout (match width with Search tab) ---
    chat_container_full = st.container()
    with chat_container_full:
        st.markdown(
            """
            <style>
            [data-testid="stHorizontalBlock"] {
                width: 100% !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        left, right = st.columns([2.4, 6.6], gap="large")

    # --- LEFT COLUMN: Chat History ---
    with left:
        st.markdown("### Chats")
        if not st.session_state.chat_sessions:
            st.info("No previous chats yet.")
        else:
            for i, session in enumerate(reversed(st.session_state.chat_sessions)):
                summary = session.get("summary", "Untitled chat")
                # Show only the chat title (summary) — no timestamp
                label = f"{summary}"
                if st.button(label, key=f"chat_open_{i}", use_container_width=True, type="secondary"):
                    st.session_state.active_chat = session["history"].copy()
                    st.rerun()

    # --- RIGHT COLUMN: Active Chat ---
    with right:
        st.markdown("#### Current Chat")

        chat_container = st.container()

        # --- Clean chat message rendering (no icons) ---
        for msg in st.session_state.active_chat:
            role_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
            chat_container.markdown(
                f"""
                <div class="{role_class}">
                    {msg["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )

                # --- Custom CSS for chat bubbles (auto-width based on text length) ---
        st.markdown("""
        <style>
        .user-msg {
            background-color: #f5f5f5;
            padding: 10px 14px;
            border-radius: 14px;
            margin: 6px 0;
            display: inline-flex;
            align-self: flex-end;
            justify-content: flex-end;
            text-align: right;
            color: #111;
            word-break: break-word;
            max-width: 75%;
        }
        .assistant-msg {
            background-color: #ffffff;
            border: 1px solid #e8e8e8;
            padding: 10px 14px;
            border-radius: 14px;
            margin: 6px 0;
            display: inline-flex;
            align-self: flex-start;
            text-align: left;
            color: #111;
            word-break: break-word;
            max-width: 75%;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .user-msg, .assistant-msg {
            animation: fadeIn 0.2s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """, unsafe_allow_html=True)


        # --- Chat input ---
        user_input = st.chat_input("Ask about collectors, regions, or interests...")

        if user_input:
            st.session_state.active_chat.append({"role": "user", "content": user_input})
            chat_container.markdown(f"<div class='user-msg'>{user_input}</div>", unsafe_allow_html=True)

            with st.spinner("Thinking..."):
                try:
                    messages = [{"role": "system", "content": system_prompt}]
                    messages.extend(st.session_state.active_chat)
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        temperature=0.5,
                        max_tokens=700,
                    )
                    response_text = completion.choices[0].message.content.strip()
                    chat_container.markdown(f"<div class='assistant-msg'>{response_text}</div>", unsafe_allow_html=True)
                    st.session_state.active_chat.append(
                        {"role": "assistant", "content": response_text}
                    )
                except Exception as e:
                    st.error(f"Chat failed: {e}")

        # --- New Chat Button ---
        if st.session_state.active_chat:
            st.divider()
            if st.button("New Chat", use_container_width=True):
                try:
                    preview_text = " ".join(
                        [m["content"] for m in st.session_state.active_chat if m["role"] == "user"]
                    )[:600]
                    summary_prompt = (
                        "Summarize this conversation in 3–5 plain words, no emojis or punctuation. "
                        "Example: 'European collectors trends'.\n\n"
                        f"{preview_text}"
                    )
                    summary_resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": summary_prompt}],
                        max_tokens=20,
                        temperature=0.5,
                    )
                    summary_text = summary_resp.choices[0].message.content.strip()
                except Exception:
                    summary_text = "Untitled chat"

                st.session_state.chat_sessions.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "summary": summary_text,
                    "history": st.session_state.active_chat.copy(),
                })
                st.session_state.active_chat = []
                st.rerun()
