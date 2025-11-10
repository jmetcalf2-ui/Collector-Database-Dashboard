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

# --- Connect Supabase ---
try:
    supabase = get_supabase()
    st.success("Connected to Supabase")
except Exception as e:
    st.error(f"Connection failed: {e}")
    supabase = None

# --- Tabs ---
tabs = st.tabs(["Search", "Saved Sets", "Chat"])

# ======================================================================
# === SEARCH TAB ===
# ======================================================================
with tabs[0]:
    st.markdown("## Search")

    col1, col2, col3, col4, col5 = st.columns([2, 1.2, 1.2, 1.2, 1.2])
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

    if st.button("Search Leads") and supabase:
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
        data = query.limit(100).execute().data or []
        if data:
            st.write(f"Found {len(data)} results")
            for lead in data:
                with st.expander(f"{lead.get('full_name', 'Unnamed')} — {lead.get('city', 'Unknown')}"):
                    st.write(f"**Email:** {lead.get('email','—')}")
                    st.write(f"**Tier:** {lead.get('tier','—')}")
                    st.write(f"**Role:** {lead.get('primary_role','—')}")
                    st.write(f"**Notes:** {lead.get('notes','')[:250]}")
        else:
            st.info("No leads found.")
    else:
        st.empty()

    # --- Divider ---
    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

    # --- Semantic Search Section ---
    st.markdown("### Semantic Search (Notes & Content)")

    query_text = st.text_input(
        "Describe the type of collector or interest you’re looking for",
        placeholder="e.g. Minimalism collectors or those following Bruce Nauman",
        key="semantic_query_text",
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        top_k = st.slider("Number of results", 5, 100, 25, key="semantic_top_k")
    with col_b:
        min_similarity = st.slider("Minimum similarity", 0.0, 1.0, 0.15, 0.01, key="semantic_similarity")

    run_semantic = st.button("Run Semantic Search", key="semantic_search_button")

    if run_semantic and query_text.strip():
        with st.spinner("Embedding query and searching..."):
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

            try:
                emb = client.embeddings.create(model=model, input=query_text).data[0].embedding
                res = supabase.rpc(
                    "ai.semantic_search_lead_supplements",
                    {
                        "query_embedding": emb,
                        "match_count": top_k,
                        "min_similarity": min_similarity,
                    },
                ).execute()

                results = res.data or []
                if results:
                    st.success(f"Found {len(results)} semantic matches")
                    st.dataframe(results, use_container_width=True, hide_index=True)
                else:
                    st.info("No semantic matches found for that query.")
            except Exception as e:
                st.error(f"Semantic search failed: {e}")
    else:
        st.caption("Enter a description and click **Run Semantic Search** to start.")


# ======================================================================
# === SAVED SETS TAB ===
# ======================================================================
with tabs[1]:
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
with tabs[2]:
    st.markdown("## ", unsafe_allow_html=True)

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

    # --- Layout ---
    left, right = st.columns([2.2, 5])

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
