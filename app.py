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

# ======================================================================
# === CONTACTS TAB ======================================================
# ======================================================================
with tabs[1]:

    # -------------------------
    # Page Title
    # -------------------------
    st.markdown("""
    <h2 style="margin-bottom: 8px;">Contacts</h2>
    """, unsafe_allow_html=True)

    # -------------------------
    # Elegant CSS Styling  
    # -------------------------
    st.markdown("""
    <style>

    /* --- Table structure --- */
    .contact-header, .contact-row {
        display: grid;
        grid-template-columns: 2fr 0.7fr 2fr;
        padding: 10px 14px;
        font-size: 15px;
        align-items: center;
    }

    /* Header */
    .contact-header {
        font-weight: 600;
        border-bottom: 1px solid #e1e1e1;
        color: #333;
        margin-top: 10px;
    }

    /* Rows */
    .contact-row {
        border-bottom: 1px solid #f3f3f3;
        transition: background-color 0.15s ease;
    }

    .contact-row:hover {
        background-color: #fafafa;
    }

    /* Name button */
    .name-button {
        background: none !important;
        border: none !important;
        text-align: left !important;
        padding: 0 !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        color: #0a3a7e !important;
        cursor: pointer;
    }

    .name-button:hover {
        text-decoration: underline;
    }

    /* Tier badge */
    .tier-badge {
        font-weight: 500;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 13px;
        display: inline-block;
    }

    .tier-A { background: #e8f1ff; color: #144a92; }
    .tier-B { background: #f4f4f4; color: #555; }
    .tier-C { background: #f9ecec; color: #7d2d2d; }
    .tier-dash { background: #eee; color: #444; }

    /* Detail panel */
    .detail-container {
        border: 1px solid #e8e8e8;
        border-radius: 8px;
        padding: 18px;
        margin: 10px 0 20px 0;
        background: #fff;
    }

    /* Clean section dividers */
    .divider {
        height: 1px;
        background: #ececec;
        margin: 14px 0;
    }

    </style>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------
    # CREATE CONTACT FORM
    # -------------------------------------------------------
    with st.expander("Create a Contact", expanded=False):
        with st.form("create_contact_form"):
            st.markdown("Enter contact details to add a new record:")

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

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------
    # CONTACT TABLE
    # -------------------------------------------------------
    if not supabase:
        st.warning("Database unavailable.")
    else:
        per_page = 20
        if "data_page" not in st.session_state:
            st.session_state.data_page = 0

        offset = st.session_state.data_page * per_page

        # Count rows
        total_count = supabase.table("leads").select("*", count="exact").limit(1).execute().count or 0
        total_pages = max(1, (total_count + per_page - 1) // per_page)

        st.caption(f"Page {st.session_state.data_page + 1} of {total_pages} — {total_count} total contacts")

        # Fetch leads
        leads = (
            supabase.table("leads")
            .select("lead_id, full_name, email, tier, primary_role, city, country, notes")
            .order("full_name")
            .range(offset, offset + per_page - 1)
            .execute()
            .data or []
        )

        # -------------------------------------------------------
        # HEADER
        # -------------------------------------------------------
        st.markdown("""
        <div class="contact-header">
            <div>Name</div>
            <div>Tier</div>
            <div>Email</div>
        </div>
        """, unsafe_allow_html=True)

        # -------------------------------------------------------
        # ROWS
        # -------------------------------------------------------
        for lead in leads:
            lead_key = str(lead["lead_id"])
            name = lead.get("full_name") or "Unnamed"
            tier = lead.get("tier") or "—"
            email_val = lead.get("email") or "—"
            role = lead.get("primary_role") or "—"
            city = lead.get("city") or ""
            country = lead.get("country") or ""

            # Click state
            open_key = f"open_{lead_key}"
            if open_key not in st.session_state:
                st.session_state[open_key] = False

            # ------------------------------
            # Row Grid
            # ------------------------------
            col1, col2, col3 = st.columns([2, 1, 2])

            # Name = clickable
            with col1:
                if st.button(name, key=f"name_{lead_key}", use_container_width=True):
                    st.session_state[open_key] = not st.session_state[open_key]

            # Tier badge
            with col2:
                badge_class = f"tier-{tier}" if tier in ["A", "B", "C"] else "tier-dash"
                st.markdown(f"<span class='tier-badge {badge_class}'>{tier}</span>", unsafe_allow_html=True)

            # Email
            with col3:
                st.write(email_val)

            # ------------------------------
            # Dropdown panel
            # ------------------------------
            if st.session_state[open_key]:
                with st.container():
                    st.markdown("<div class='detail-container'>", unsafe_allow_html=True)

                    st.markdown(f"<h4 style='margin-top:0'>{name}</h4>", unsafe_allow_html=True)

                    if city or country:
                        st.caption(f"{city}, {country}".strip(", "))

                    st.caption(f"{role} | Tier {tier}")
                    st.write(email_val)

                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

                    # ---------------- Buttons -----------------
                    c1, c2, c3 = st.columns([2, 2, 1])

                    # Summarize
                    with c1:
                        if st.button("Summarize", key=f"summ_{lead_key}"):
                            supplements = (
                                supabase.table("leads_supplements")
                                .select("notes")
                                .eq("lead_id", lead_key)
                                .execute()
                                .data or []
                            )

                            base_notes = lead.get("notes") or ""
                            supplement_notes = "\n\n".join((s.get("notes") or "").strip() for s in supplements)
                            combined_notes = (
                                base_notes
                                + ("\n\n" if base_notes and supplement_notes else "")
                                + supplement_notes
                            ).strip()

                            summary = summarize_collector(lead_key, combined_notes)
                            st.markdown("### Notes")
                            st.markdown(summary, unsafe_allow_html=True)

                    # Save
                    with c2:
                        st.button("Add to Saved Set", key=f"save_{lead_key}")

                    # Delete
                    with c3:
                        if st.button("Delete", key=f"del_{lead_key}"):
                            st.session_state[f"confirm_delete_{lead_key}"] = True

                    # Confirm delete
                    if st.session_state.get(f"confirm_delete_{lead_key}", False):
                        st.warning(f"Delete {name}?")

                        yes = st.button("Yes", key=f"yes_{lead_key}")
                        no = st.button("No", key=f"no_{lead_key}")

                        if yes:
                            supabase.table("leads").delete().eq("lead_id", lead_key).execute()
                            st.success("Deleted.")
                            st.rerun()

                        if no:
                            st.session_state[f"confirm_delete_{lead_key}"] = False
                            st.rerun()

                    st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------------------------------------
        # Pagination
        # -------------------------------------------------------
        st.markdown("<hr>", unsafe_allow_html=True)

        left, prev, nxt, right = st.columns([2, 1, 1, 2])

        with prev:
            if st.button("Previous", disabled=st.session_state.data_page == 0):
                st.session_state.data_page -= 1
                st.rerun()

        with nxt:
            if st.button("Next", disabled=st.session_state.data_page >= total_pages - 1):
                st.session_state.data_page += 1
                st.rerun()

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

    # --- Initialize OpenAI client ---
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # --- Two-column layout (same proportions as before) ---
    left, right = st.columns([2.4, 6.6], gap="large")

    # ------------------------------------------------------
    # LEFT COLUMN: CHAT HISTORY
    # ------------------------------------------------------
    with left:
        # 👇 EXACT same heading style as "Contacts"
        st.markdown("## Chats")

        if not st.session_state.chat_sessions:
            st.info("No previous chats.")
        else:
            for i, session in enumerate(reversed(st.session_state.chat_sessions)):
                summary = session.get("summary", "Untitled chat")
                if st.button(summary, key=f"chat_open_{i}", use_container_width=True, type="secondary"):
                    st.session_state.active_chat = session["history"].copy()
                    st.rerun()

    # ------------------------------------------------------
    # RIGHT COLUMN: CURRENT CHAT
    # ------------------------------------------------------
    with right:
        # 👇 Also an H2, so it matches Contacts 1:1
        st.markdown("## Current Chat")

        # Render chat messages using the CSS bubbles you defined
        for msg in st.session_state.active_chat:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='message-user'>{msg['content']}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='message-assistant'>{msg['content']}</div>",
                    unsafe_allow_html=True
                )

        # Clear floats so the input bar doesn’t overlap bubbles
        st.markdown("<div style='clear:both'></div>", unsafe_allow_html=True)

        # ---------------- Chat input ----------------
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
                    response_text = completion.choices[0].message.content.strip()

                    st.session_state.active_chat.append(
                        {"role": "assistant", "content": response_text}
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Chat failed: {e}")

        # ---------------- New Chat button ----------------
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
