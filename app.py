import streamlit as st
from supabase_client import get_supabase
from components import inject_css
import os
from openai import OpenAI
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Dashboard", layout="wide")
inject_css()   # loads style.css

# Sidebar hidden (already removed via CSS)
with st.sidebar:
    st.write(" ")

# Session state init
if "selected_leads" not in st.session_state:
    st.session_state.selected_leads = []
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []
if "active_chat" not in st.session_state:
    st.session_state.active_chat = []

# Header
st.markdown("<h1>Dashboard</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------
# CONNECT SUPABASE
# ---------------------------------------------------------
try:
    supabase = get_supabase()
except Exception as e:
    st.error(f"⚠️ Supabase connection failed: {e}")
    supabase = None

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tabs = st.tabs(["Search", "Contacts", "Saved Sets", "Chat"])

# ---------------------------------------------------------
# OPENAI SUMMARIZER CACHE
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def summarize_collector(lead_id: str, combined_notes: str) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return "⚠️ Missing OPENAI_API_KEY."

    if not combined_notes.strip():
        return "⚠️ No notes found for this lead."

    try:
        client = OpenAI(api_key=key)
        prompt = f"""
        You are an expert art-market researcher creating collector intelligence summaries.
        Write 4–6 short bullet points summarizing this collector's data factually.
        Avoid adjectives. Stick to institutional roles, artist preferences, purchases, philanthropy.
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

# =========================================================
# === SEARCH TAB ==========================================
# =========================================================
with tabs[0]:
    st.markdown("<h2>Search</h2>", unsafe_allow_html=True)

    # Search filters in 5 columns
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

    semantic_query = st.text_input(
        "Semantic Search",
        placeholder="e.g. Minimalism collectors"
    )

    # Perform search
    if st.button("Search Leads") and supabase:
        with st.spinner("Searching…"):
            results = []

            # Semantic search branch
            if semantic_query.strip():
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

            # Filtered lookup branch
            else:
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

        # Display results
        if results:
            st.success(f"Found {len(results)} results")

            for lead in results:
                name = lead.get("full_name", "Unnamed")
                label = name
                if lead.get("city"):
                    label += f" — {lead['city']}"

                with st.expander(label):
                    st.write(f"**Email:** {lead.get('email', '—')}")
                    st.write(f"**Tier:** {lead.get('tier', '—')}")
                    st.write(f"**Role:** {lead.get('primary_role', '—')}")

                    try:
                        lead_pk = lead.get("lead_id") or lead.get("id")
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
                            (s.get("notes") or "").strip() for s in supplements
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
                        st.write(f"⚠️ Could not summarize: {e}")

        else:
            st.info("No leads found.")

# =========================================================
# === CONTACTS TAB ========================================
# =========================================================
with tabs[1]:
    st.markdown("<h2>Contacts</h2>", unsafe_allow_html=True)

    # -------------------------------------------------------
    # CREATE CONTACT FORM
    # -------------------------------------------------------
    with st.expander("Create a Contact", expanded=False):
        with st.form("create_contact_form"):

            st.markdown("Enter contact details below:")

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
                    st.warning("Please provide at least a name and an email.")
                else:
                    try:
                        response = (
                            supabase.table("leads")
                            .insert({
                                "full_name": full_name.strip(),
                                "email": email.strip(),
                                "primary_role": primary_role.strip() or None,
                                "city": city.strip() or None,
                                "country": country.strip() or None,
                                "tier": None if tier == "—" else tier,
                                "notes": notes.strip() or None,
                            })
                            .execute()
                        )
                        if getattr(response, "status_code", 400) < 300:
                            st.success(f"{full_name} added.")
                            st.rerun()
                        else:
                            st.error(f"Insert failed: {response}")
                    except Exception as e:
                        st.error(f"Error creating contact: {e}")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------
    # CONTACT GRID + PAGINATION
    # -------------------------------------------------------
    if not supabase:
        st.warning("Database unavailable.")
    else:
        per_page = 20
        if "data_page" not in st.session_state:
            st.session_state.data_page = 0

        offset = st.session_state.data_page * per_page

        # Count total leads
        try:
            total_response = (
                supabase.table("leads")
                .select("*", count="exact")
                .limit(1)
                .execute()
            )
            total_count = total_response.count or 0
        except Exception as e:
            st.error(f"Failed to count leads: {e}")
            total_count = 0

        total_pages = max(1, (total_count + per_page - 1) // per_page)
        st.caption(f"Page {st.session_state.data_page + 1} of {total_pages} — {total_count} total contacts")

        # Fetch leads
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

        # -------------------------------------------------------
        # DISPLAY GRID
        # -------------------------------------------------------
        if leads:
            cols = st.columns(2)

            for i, lead in enumerate(leads):
                col = cols[i % 2]

                with col:
                    name = lead.get("full_name", "Unnamed")
                    role = lead.get("primary_role") or "—"
                    tier_val = lead.get("tier") or "—"
                    email_v = lead.get("email") or "—"
                    city_v = (lead.get("city") or "").strip()
                    country_v = (lead.get("country") or "").strip()

                    lead_key = str(lead["lead_id"])
                    summary_key = f"summary_{lead_key}"

                    # Clean wrapper (fixed bubble issue)
                    with st.container():
                        st.markdown("<div class='contact-card'>", unsafe_allow_html=True)

                        with st.expander(name):

                            if city_v or country_v:
                                st.caption(f"{city_v}, {country_v}".strip(", "))

                            st.caption(f"{role} | Tier {tier_val}")
                            st.write(email_v)

                            sum_col, del_col = st.columns([3, 1])

                            # SUMMARIZE BUTTON
                            with sum_col:
                                if summary_key not in st.session_state:
                                    if st.button(f"Summarize {name}", key=f"sum_{lead_key}"):
                                        with st.spinner("Summarizing…"):
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
                                                st.error(f"Summarization error: {e}")
                                else:
                                    st.markdown("**Notes:**")
                                    st.markdown(st.session_state[summary_key], unsafe_allow_html=True)

                            # DELETE BUTTON
                            with del_col:
                                if st.button("Delete", key=f"del_{lead_key}"):
                                    st.session_state[f"confirm_delete_{lead_key}"] = True

                                if st.session_state.get(f"confirm_delete_{lead_key}", False):
                                    st.warning(f"Delete {name}?")
                                    yes = st.button("Yes, delete", key=f"yes_{lead_key}")
                                    no = st.button("Cancel", key=f"no_{lead_key}")

                                    if yes:
                                        try:
                                            supabase.table("leads").delete().eq("lead_id", lead_key).execute()
                                            st.success(f"{name} deleted.")
                                            st.session_state[f"confirm_delete_{lead_key}"] = False
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Delete failed: {e}")

                                    if no:
                                        st.session_state[f"confirm_delete_{lead_key}"] = False
                                        st.rerun()

                        st.markdown("</div>", unsafe_allow_html=True)

            # Pagination Controls
            st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
            left, prev, nxt, right = st.columns([2, 1, 1, 2])

            with prev:
                if st.button("Previous", use_container_width=True, disabled=st.session_state.data_page == 0):
                    st.session_state.data_page -= 1
                    st.rerun()

            with nxt:
                if st.button("Next", use_container_width=True, disabled=st.session_state.data_page >= total_pages - 1):
                    st.session_state.data_page += 1
                    st.rerun()

        else:
            st.info("No contacts found.")

# =========================================================
# === SAVED SETS TAB ======================================
# =========================================================
with tabs[2]:
    st.markdown("<h2>Saved Sets</h2>", unsafe_allow_html=True)

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
                with st.expander(s["name"]):
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
            "You answer questions conversationally, referencing collectors, artists, "
            "galleries, and market trends when relevant. Keep responses factual and concise."
        )

    # --- OpenAI client ---
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # --- Layout (two-column like Contacts tab) ---
    chat_container_full = st.container()

    with chat_container_full:
        left, right = st.columns([2.4, 6.6], gap="large")

    # ================================================================
    # LEFT COLUMN — CHAT HISTORY
    # ================================================================
    with left:
        st.markdown("## Chats")   # <-- MATCHES CONTACTS SIZE

        if not st.session_state.chat_sessions:
            st.info("No previous chats.")
        else:
            for i, session in enumerate(reversed(st.session_state.chat_sessions)):
                summary = session.get("summary", "Untitled chat")
                if st.button(summary, key=f"chat_open_{i}", use_container_width=True, type="secondary"):
                    st.session_state.active_chat = session["history"].copy()
                    st.rerun()

    # ================================================================
    # RIGHT COLUMN — CURRENT CHAT
    # ================================================================
    with right:
        st.markdown("## Current Chat")   # <-- FIXED HEADER SIZE

        chat_box = st.container()

        # Render chat history messages
        for msg in st.session_state.active_chat:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style="
                        background:#e8eef8;
                        padding:12px 15px;
                        border-radius:14px;
                        margin:10px 0;
                        max-width:75%;
                        float:right;
                        clear:both;
                    ">
                        {msg['content']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="
                        background:#ffffff;
                        padding:12px 15px;
                        border-radius:14px;
                        margin:10px 0;
                        max-width:75%;
                        float:left;
                        clear:both;
                        border:1px solid #ececec;
                    ">
                        {msg['content']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Clear float
        st.markdown("<div style='clear:both'></div>", unsafe_allow_html=True)

        # ================================================================
        # CHAT INPUT
        # ================================================================
        user_input = st.chat_input("Ask a question…")

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

                    reply = completion.choices[0].message.content.strip()
                    st.session_state.active_chat.append({"role": "assistant", "content": reply})

                    st.rerun()

                except Exception as e:
                    st.error(f"Chat failed: {e}")

        # ================================================================
        # NEW CHAT BUTTON
        # ================================================================
        if st.session_state.active_chat:
            st.divider()
            if st.button("New Chat", use_container_width=True):

                try:
                    preview_text = " ".join(
                        [m["content"] for m in st.session_state.active_chat if m["role"] == "user"]
                    )[:600]

                    summary_prompt = (
                        "Summarize this conversation in 3–5 plain words. No emojis.\n\n"
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
