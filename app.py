import streamlit as st
from supabase_client import get_supabase
from components import inject_css
import os
from openai import OpenAI
from datetime import datetime
from pathlib import Path
import pandas as pd  # NEW: for spreadsheet-style Contacts table

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
tabs = st.tabs(["Search", "Contacts", "Saved Sets", "Chat"])

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

        # --- Display results using PROFILE GRID (moved from Contacts) ---
        if results:
            st.success(f"Found {len(results)} results")

            cols = st.columns(2)
            for i, lead in enumerate(results):
                col = cols[i % 2]
                with col:
                    name = lead.get("full_name", "Unnamed")
                    tier_val = lead.get("tier", "—")
                    role_val = lead.get("primary_role", "—")
                    email_val = lead.get("email", "—")
                    city_val = (lead.get("city") or "").strip()
                    country_val = (lead.get("country") or "").strip()

                    expander_label = name
                    lead_key = str(lead.get("lead_id") or lead.get("id") or "")
                    summary_key = f"summary_{lead_key}" if lead_key else None

                    with st.expander(expander_label):
                        st.markdown(f"**{name}**")

                        if city_val or country_val:
                            location = f"{city_val}, {country_val}".strip(", ")
                            st.caption(location)

                        st.caption(f"{role_val if role_val else '—'} | Tier {tier_val if tier_val else '—'}")
                        st.write(email_val)

                        # --- Row with Summarize and Delete buttons ---
                        sum_col, del_col = st.columns([3, 1])

                        # Summarize block (reused logic)
                        with sum_col:
                            if lead_key and summary_key:
                                if summary_key not in st.session_state:
                                    if st.button(f"Summarize {name}", key=f"sum_{lead_key}"):
                                        with st.spinner("Summarizing notes..."):
                                            try:
                                                supplements = (
                                                    supabase.table("leads_supplements")
                                                    .select("notes")
                                                    .eq("lead_id", lead_key)
                                                    .execute()
                                                    .data
                                                    or []
                                                )

                                                base_notes = lead.get("notes") or ""
                                                supplement_notes = "\n\n".join(
                                                    (s.get("notes") or "").strip()
                                                    for s in supplements
                                                    if isinstance(s, dict)
                                                )
                                                combined_notes = (
                                                    base_notes
                                                    + ("\n\n" if base_notes and supplement_notes else "")
                                                    + supplement_notes
                                                ).strip()

                                                summary = summarize_collector(lead_key, combined_notes)
                                                st.session_state[summary_key] = summary
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Summarization failed: {e}")
                                else:
                                    st.markdown("**Notes:**")
                                    st.markdown(st.session_state[summary_key], unsafe_allow_html=True)
                            else:
                                st.caption("No ID available to summarize notes.")

                        # Delete block (reused logic)
                        with del_col:
                            if lead_key:
                                if st.button("Delete", key=f"del_{lead_key}"):
                                    st.session_state[f"confirm_delete_{lead_key}"] = True

                                if st.session_state.get(f"confirm_delete_{lead_key}", False):
                                    st.warning(f"Are you sure you want to delete {name}?")
                                    confirm = st.button("Yes, delete", key=f"confirm_del_{lead_key}")
                                    cancel = st.button("Cancel", key=f"cancel_del_{lead_key}")

                                    if confirm:
                                        try:
                                            supabase.table("leads").delete().eq("lead_id", lead_key).execute()
                                            st.success(f"{name} has been deleted.")
                                            st.session_state[f"confirm_delete_{lead_key}"] = False
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error deleting contact: {e}")

                                    if cancel:
                                        st.session_state[f"confirm_delete_{lead_key}"] = False
                                        st.rerun()
        else:
            st.info("No leads found.")

# ======================================================================
# === CONTACTS TAB ===
# ======================================================================
with tabs[1]:
    st.markdown("## Contacts")

    # --- Create a new contact form ---
    with st.expander("Create a Contact", expanded=False):
        with st.form("create_contact_form"):
            st.markdown("Enter contact details to add a new record to the leads table:")

            # Core lead fields — adjust to your Supabase schema
            full_name = st.text_input("Full Name")
            email = st.text_input("Email")
            primary_role = st.text_input("Primary Role")
            city = st.text_input("City")
            country = st.text_input("Country")
            tier = st.selectbox("Tier", ["A", "B", "C", "—"], index=3)
            notes = st.text_area("Notes", height=100)

            # Submit
            submitted = st.form_submit_button("Create Contact")
            if submitted:
                if not full_name or not email:
                    st.warning("Please provide at least a name and email.")
                else:
                    try:
                        response = (
                            supabase.table("leads")
                            .insert({
                                "full_name": full_name.strip(),
                                "email": email.strip(),
                                "primary_role": primary_role.strip() if primary_role else None,
                                "city": city.strip() if city else None,
                                "country": country.strip() if country else None,
                                "tier": None if tier == "—" else tier,
                                "notes": notes.strip() if notes else None,
                            })
                            .execute()
                        )
                        if getattr(response, "status_code", 400) < 300:
                            st.success(f"{full_name} has been added to your contacts.")
                            st.rerun()
                        else:
                            st.error(f"Insert failed: {response}")
                    except Exception as e:
                        st.error(f"Error creating contact: {e}")

    # --- Small spacing before table ---
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    if not supabase:
        st.warning("Database unavailable.")
    else:
        # --- Pagination setup for spreadsheet view ---
        per_page = 20
        if "data_page" not in st.session_state:
            st.session_state.data_page = 0

        offset = st.session_state.data_page * per_page

        # --- Count total leads ---
        try:
            total_response = supabase.table("leads").select("*", count="exact").limit(1).execute()
            total_count = getattr(total_response, "count", None) or 0
        except Exception as e:
            st.error(f"Could not fetch total lead count: {e}")
            total_count = 0

        total_pages = max(1, (total_count + per_page - 1) // per_page)
        st.caption(f"Page {st.session_state.data_page + 1} of {total_pages} — {total_count} total leads")

        # --- Fetch paginated leads ---
        try:
            leads = (
                supabase.table("leads")
                .select("lead_id, full_name, email, tier, primary_role, city, country, notes")
                .order("created_at", desc=True)
                .range(offset, offset + per_page - 1)
                .execute()
                .data
                or []
            )
        except Exception as e:
            st.error(f"Failed to fetch leads: {e}")
            leads = []

        # --- Display leads in spreadsheet-style table ---
        if leads:
            df = pd.DataFrame(leads)

            # Reorder columns to something sensible
            desired_cols = ["lead_id", "full_name", "email", "tier", "primary_role", "city", "country", "notes"]
            existing_cols = [c for c in desired_cols if c in df.columns]
            df = df[existing_cols]

            st.dataframe(df, use_container_width=True)

            # --- Pagination controls ---
            st.markdown("---")
            col_space_left, col_prev, col_next, col_space_right = st.columns([2, 1, 1, 2])

            with col_prev:
                if st.button("Previous", use_container_width=True, disabled=st.session_state.data_page == 0):
                    st.session_state.data_page -= 1
                    st.rerun()

            with col_next:
                if st.button("Next", use_container_width=True, disabled=st.session_state.data_page >= total_pages - 1):
                    st.session_state.data_page += 1
                    st.rerun()
        else:
            st.info("No leads found.")



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
    
        # --- Render chat messages (clean markdown style) ---
        for msg in st.session_state.active_chat:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style="
                        background-color:#f5f5f5;
                        color:#111;
                        padding:10px 14px;
                        border-radius:12px;
                        margin:6px 0;
                        text-align:right;
                        max-width:75%;
                        float:right;
                        clear:both;
                        box-shadow:0 1px 2px rgba(0,0,0,0.1);
                        word-break:break-word;">
                        {msg["content"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="
                        background-color:#ffffff;
                        color:#111;
                        padding:10px 14px;
                        border-radius:12px;
                        margin:6px 0;
                        text-align:left;
                        max-width:75%;
                        float:left;
                        clear:both;
                        box-shadow:0 1px 3px rgba(0,0,0,0.08);
                        word-break:break-word;">
                        {msg["content"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
        # --- Chat input ---
        st.markdown("<div style='clear:both'></div>", unsafe_allow_html=True)
        user_input = st.chat_input("Ask about collectors, regions, or interests...")
    
        if user_input:
            st.session_state.active_chat.append({"role": "user", "content": user_input})
    
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
    
                    # Append response and re-render
                    st.session_state.active_chat.append({"role": "assistant", "content": response_text})
                    st.rerun()
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
