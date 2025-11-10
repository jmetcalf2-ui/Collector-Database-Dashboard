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
    st.markdown("## Chat with Collector Intelligence")
    from services.rag import answer_with_context

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    system_prompt_path = Path("prompts/system_prompt.md")
    system_prompt = system_prompt_path.read_text().strip() if system_prompt_path.exists() else (
        "You are a helpful art-market assistant. Answer based only on the provided context."
    )

    left, right = st.columns([2.2, 5])

    # --- LEFT COLUMN: Chat History ---
    with left:
        st.markdown("### Chats")
        if not st.session_state.chat_sessions:
            st.info("No previous chats yet.")
        else:
            for i, session in enumerate(reversed(st.session_state.chat_sessions)):
                label = f"{session.get('summary', 'Untitled chat')}\n{session['timestamp']}"
                if st.button(label, key=f"chat_open_{i}", use_container_width=True):
                    st.session_state.active_chat = session["history"].copy()
                    st.rerun()

    # --- RIGHT COLUMN: Active Chat ---
       # --- RIGHT COLUMN: Active Chat ---
    with right:
        st.markdown("#### Current Chat")

        # --- Wrap in scrollable container to prevent stacking
        chat_area = st.container()

        # If there are active messages, show them
        if st.session_state.active_chat:
            with chat_area:
                for msg in st.session_state.active_chat:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
        else:
            # Render placeholder when no active chat
            with chat_area:
                st.info("Start a new chat or select one from the left.")

        # --- Input pinned to bottom ---
        user_input = st.chat_input("Ask about collectors, regions, or interests...")

        # --- Handle user input
        if user_input and supabase:
            # Add user message
            st.session_state.active_chat.append({"role": "user", "content": user_input})

            # Display user message immediately
            with chat_area.chat_message("user"):
                st.markdown(user_input)

            # Generate assistant reply
            with chat_area.chat_message("assistant"):
                with st.spinner("Consulting database and reasoning..."):
                    from services.rag import answer_with_context
                    try:
                        res = answer_with_context(
                            supabase,
                            question=user_input,
                            system_prompt=system_prompt,
                            match_count=10,
                            min_similarity=0.15,
                        )
                        st.markdown(res["answer"])
                        st.session_state.active_chat.append(
                            {"role": "assistant", "content": res["answer"]}
                        )
                    except Exception as e:
                        st.error(f"Chat failed: {e}")

        # --- New Chat Button ---
        if st.session_state.active_chat:
            st.divider()
            if st.button("New Chat", use_container_width=True):
                # Generate a short title before clearing
                try:
                    preview_text = " ".join(
                        [m["content"] for m in st.session_state.active_chat if m["role"] == "user"]
                    )[:600]
                    summary_prompt = (
                        "Summarize this chat in 3–5 plain words, no emojis or punctuation. "
                        "Example: Top collectors in Europe.\n\n"
                        f"{preview_text}"
                    )
                    summary_resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": summary_prompt}],
                        max_tokens=20,
                        temperature=0.4,
                    )
                    summary_text = summary_resp.choices[0].message.content.strip() or "Untitled chat"
                except Exception:
                    summary_text = "Untitled chat"

                # Save and reset chat safely
                st.session_state.chat_sessions.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "summary": summary_text,
                    "history": st.session_state.active_chat.copy(),
                })
                st.session_state.active_chat = []
                st.experimental_rerun()

        # --- NEW CHAT BUTTON ---
        if st.session_state.active_chat:
            st.divider()
            if st.button("New Chat", use_container_width=True):
                try:
                    preview_text = " ".join(
                        [m["content"] for m in st.session_state.active_chat if m["role"] == "user"]
                    )[:600]

                    # Force summary creation before rerun
                    summary_prompt = (
                        "Summarize this chat in 3–5 plain words, no emojis or punctuation. "
                        "Example: Top collectors in Europe.\n\n"
                        f"{preview_text}"
                    )
                    summary_resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": summary_prompt}],
                        max_tokens=20,
                        temperature=0.4,
                    )
                    summary_text = summary_resp.choices[0].message.content.strip()
                    if not summary_text:
                        summary_text = "Untitled chat"
                except Exception:
                    summary_text = "Untitled chat"

                # Save session safely
                st.session_state.chat_sessions.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "summary": summary_text,
                    "history": st.session_state.active_chat.copy(),
                })

                # Reset chat
                st.session_state.active_chat = []
                st.rerun()
