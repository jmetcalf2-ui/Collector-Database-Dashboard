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
</style>
""", unsafe_allow_html=True)

# --- Main content ---
st.markdown("<h1>Dashboard</h1>", unsafe_allow_html=True)

# --- Additional layout polish ---
st.markdown("""
<style>
section.main > div.block-container { padding-top: 2rem !important; }
.streamlit-expanderHeader { font-weight: 600 !important; font-size: 1rem !important; }
.streamlit-expanderContent { padding: 0.5rem 0.2rem !important; }
</style>
""", unsafe_allow_html=True)

# --- Connect Supabase ---
try:
    supabase = get_supabase()
except Exception as e:
    st.error(f"⚠️ Supabase connection failed: {e}")
    supabase = None

# --- Tabs ---
tabs = st.tabs(["Search", "Contacts", "Saved Sets", "Chat"])

# --- Summarization helper ---
@st.cache_data(show_spinner=False)
def summarize_collector(lead_id: str, combined_notes: str) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return "⚠️ Missing OPENAI_API_KEY"

    if not combined_notes.strip():
        return "⚠️ No notes found."

    try:
        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize art collectors factually."},
                {"role": "user", "content": f"NOTES:\n{combined_notes}"}
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

    col1, col2, col3, col4, col5 = st.columns([2.2, 1.2, 1.2, 1.2, 1.2])
    with col1:
        keyword = st.text_input("Keyword", placeholder="Name, email, interests…")
    with col2:
        city = st.text_input("City")
    with col3:
        country = st.text_input("Country")
    with col4:
        tier = st.selectbox("Tier", ["", "A", "B", "C"], index=0)
    with col5:
        role = st.text_input("Primary Role")

    semantic_query = st.text_input("Semantic Search", placeholder="e.g. Minimalism collectors")

    if st.button("Search Leads") and supabase:
        with st.spinner("Searching…"):
            results = []

            if semantic_query.strip():
                try:
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    emb = client.embeddings.create(
                        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
                        input=semantic_query,
                    ).data[0].embedding

                    res = supabase.rpc(
                        "rpc_semantic_search_leads_supplements",
                        {"query_embedding": emb, "match_count": 25, "min_score": 0.15},
                    ).execute()

                    results = res.data or []
                except Exception as e:
                    st.error(f"Semantic search failed: {e}")

            else:
                q = supabase.table("leads").select("*")
                if keyword:
                    q = q.ilike("full_name", f"%{keyword}%")
                if city:
                    q = q.ilike("city", f"%{city}%")
                if country:
                    q = q.ilike("country", f"%{country}%")
                if tier:
                    q = q.eq("tier", tier)
                if role:
                    q = q.ilike("primary_role", f"%{role}%")
                results = q.limit(100).execute().data or []

        if results:
            st.success(f"Found {len(results)} results")
            for lead in results:
                with st.expander(lead.get("full_name", "Unnamed")):
                    st.write(f"**Email:** {lead.get('email', '—')}")
                    st.write(f"**Tier:** {lead.get('tier', '—')}")
                    st.write(f"**Role:** {lead.get('primary_role', '—')}")

                    # Summaries
                    lead_pk = lead.get("lead_id") or lead.get("id")
                    supplements = supabase.table("leads_supplements").select("notes").eq(
                        "lead_id", str(lead_pk)
                    ).execute().data or []
                    base_notes = lead.get("notes") or ""
                    supp_notes = "\n\n".join(
                        (s.get("notes") or "").strip() for s in supplements
                    )
                    combined = f"{base_notes}\n\n{supp_notes}".strip()

                    summary = summarize_collector(str(lead_pk), combined)
                    st.markdown("**Notes:**")
                    st.markdown(summary, unsafe_allow_html=True)
        else:
            st.info("No leads found.")

# ======================================================================
# === CONTACTS TAB — PART 1
# ======================================================================
with tabs[1]:
    st.markdown("<h2>Contacts</h2>", unsafe_allow_html=True)

    with st.expander("Create a Contact", expanded=False):
        with st.form("create_contact_form"):
            st.markdown("Enter contact details to add a new record:")

            st.markdown("<div style='padding: 0.8rem 0;'>", unsafe_allow_html=True)

            full_name = st.text_input("Full Name")
            email = st.text_input("Email")
            primary_role = st.text_input("Primary Role")
            city = st.text_input("City")
            country = st.text_input("Country")
            tier = st.selectbox("Tier", ["A", "B", "C", "—"], index=3)
            notes = st.text_area("Notes", height=100)

            st.markdown("</div>", unsafe_allow_html=True)

            if st.form_submit_button("Create Contact"):
                if not full_name or not email:
                    st.warning("Name + email required.")
                else:
                    try:
                        supabase.table("leads").insert({
                            "full_name": full_name,
                            "email": email,
                            "primary_role": primary_role or None,
                            "city": city or None,
                            "country": country or None,
                            "tier": None if tier == "—" else tier,
                            "notes": notes or None,
                        }).execute()
                        st.success(f"{full_name} added.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Insert failed: {e}")
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    # ===============================
    # CONTACT LIST
    # ===============================
    if not supabase:
        st.warning("Database unavailable.")
    else:
        per_page = 20
        if "data_page" not in st.session_state:
            st.session_state.data_page = 0
        offset = st.session_state.data_page * per_page

        try:
            total_response = (
                supabase.table("leads")
                .select("*", count="exact")
                .limit(1)
                .execute()
            )
            total_count = total_response.count or 0
        except:
            total_count = 0

        total_pages = max(1, (total_count + per_page - 1) // per_page)
        st.caption(f"Page {st.session_state.data_page+1} of {total_pages}")

        try:
            leads = (
                supabase.table("leads")
                .select("lead_id, full_name, email, tier, primary_role, city, country, notes")
                .order("created_at", desc=True)
                .range(offset, offset + per_page - 1)
                .execute().data or []
            )
        except:
            leads = []

        if leads:
            cols = st.columns(2)
            for i, lead in enumerate(leads):
                col = cols[i % 2]
                with col:
                    name = lead.get("full_name", "Unnamed")
                    role = lead.get("primary_role", "—")
                    tier_val = lead.get("tier", "—")
                    email_val = lead.get("email", "—")
                    city = (lead.get("city") or "").strip()
                    country = (lead.get("country") or "").strip()

                    lead_key = str(lead["lead_id"])
                    summary_key = f"summary_{lead_key}"

                    with st.container():
                        st.markdown("<div class='contact-card'>", unsafe_allow_html=True)

                        with st.expander(name):

                            if city or country:
                                st.caption(f"{city}, {country}".strip(", "))

                            st.caption(f"{role} | Tier {tier_val}")
                            st.write(email_val)

                            sum_col, del_col = st.columns([3, 1])

                            with sum_col:
                                if summary_key not in st.session_state:
                                    if st.button(f"Summarize {name}", key=f"sum_{lead_key}"):
                                        with st.spinner("Summarizing…"):
                                            supplements = (
                                                supabase.table("leads_supplements")
                                                .select("notes")
                                                .eq("lead_id", lead_key)
                                                .execute().data or []
                                            )
                                            base = lead.get("notes") or ""
                                            supp = "\n\n".join(
                                                (s.get("notes") or "").strip()
                                                for s in supplements
                                            )
                                            combined = f"{base}\n\n{supp}".strip()
                                            st.session_state[summary_key] = summarize_collector(
                                                lead_key, combined
                                            )
                                            st.rerun()
                                else:
                                    st.markdown("**Notes:**")
                                    st.markdown(st.session_state[summary_key], unsafe_allow_html=True)

                            with del_col:
                                if st.button("Delete", key=f"del_{lead_key}"):
                                    st.session_state[f"confirm_delete_{lead_key}"] = True

                                if st.session_state.get(f"confirm_delete_{lead_key}", False):
                                    st.warning(f"Delete {name}?")
                                    if st.button("Yes", key=f"yes_{lead_key}"):
                                        supabase.table("leads").delete().eq(
                                            "lead_id", lead_key
                                        ).execute()
                                        st.success(f"{name} deleted.")
                                        st.session_state[f"confirm_delete_{lead_key}"] = False
                                        st.rerun()
                                    if st.button("Cancel", key=f"cancel_{lead_key}"):
                                        st.session_state[f"confirm_delete_{lead_key}"] = False
                                        st.rerun()

                        st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
            colL, colPrev, colNext, colR = st.columns([2, 1, 1, 2])

            with colPrev:
                if st.button("Previous", disabled=st.session_state.data_page == 0):
                    st.session_state.data_page -= 1
                    st.rerun()

            with colNext:
                if st.button("Next", disabled=st.session_state.data_page >= total_pages - 1):
                    st.session_state.data_page += 1
                    st.rerun()

        else:
            st.info("No leads found.")

# ======================================================================
# === SAVED SETS TAB
# ======================================================================
with tabs[2]:
    st.markdown("## Saved Sets")
    if not supabase:
        st.warning("Database unavailable.")
    else:
        sets = (
            supabase.table("saved_sets")
            .select("*")
            .order("created_at", desc=True)
            .execute().data or []
        )
        if not sets:
            st.info("No saved sets yet.")
        else:
            for s in sets:
                with st.expander(s["name"]):
                    st.write(f"**Description:** {s.get('description', '—')}")
                    st.write(f"**Created:** {s.get('created_at', '—')}")

# ======================================================================
# === CHAT TAB
# ======================================================================
with tabs[3]:
    system_prompt_path = Path("prompts/system_prompt.md")
    if system_prompt_path.exists():
        system_prompt = system_prompt_path.read_text().strip()
    else:
        system_prompt = (
            "You are CollectorGPT — a helpful art-market assistant."
        )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    left, right = st.columns([2.4, 6.6], gap="large")

    # HISTORY
    with left:
        st.markdown("### Chats")
        if not st.session_state.chat_sessions:
            st.info("No previous chats.")
        else:
            for i, session in enumerate(reversed(st.session_state.chat_sessions)):
                if st.button(session.get("summary", "Untitled"), key=f"chat_{i}", use_container_width=True):
                    st.session_state.active_chat = session["history"].copy()
                    st.rerun()

    # ACTIVE CHAT
    with right:
        st.markdown("#### Current Chat")

        for msg in st.session_state.active_chat:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style='background:#e8eef8;padding:12px;border-radius:14px;float:right;margin:10px 0;max-width:75%;clear:both;'>
                        {msg['content']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style='background:#fff;padding:12px;border-radius:14px;border:1px solid #ececec;float:left;margin:10px 0;max-width:75%;clear:both;'>
                        {msg['content']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("<div style='clear:both;'></div>", unsafe_allow_html=True)

        user_input = st.chat_input("Ask a question…")
        if user_input:
            st.session_state.active_chat.append({"role": "user", "content": user_input})

            with st.spinner("Thinking…"):
                try:
                    messages = [{"role": "system", "content": system_prompt}]
                    messages.extend(st.session_state.active_chat)

                    result = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        temperature=0.5,
                        max_tokens=700,
                    )

                    st.session_state.active_chat.append({
                        "role": "assistant",
                        "content": result.choices[0].message.content.strip()
                    })
                    st.rerun()

                except Exception as e:
                    st.error(f"Chat failed: {e}")

        if st.session_state.active_chat:
            st.divider()
            if st.button("New Chat", use_container_width=True):
                try:
                    preview = " ".join(
                        msg["content"] for msg in st.session_state.active_chat if msg["role"] == "user"
                    )[:600]
                    summary = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": f"Summarize in 3–5 words:\n\n{preview}"}],
                        max_tokens=20,
                    ).choices[0].message.content.strip()
                except:
                    summary = "Untitled chat"

                st.session_state.chat_sessions.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "summary": summary,
                    "history": st.session_state.active_chat.copy(),
                })
                st.session_state.active_chat = []
                st.rerun()
