import streamlit as st
from supabase_client import get_supabase
from components import inject_css
import os
from openai import OpenAI
from datetime import datetime
from pathlib import Path
import pandas as pd  # For spreadsheet-style Contacts table

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
# === SEARCH TAB (CLEAN, GRID ALWAYS VISIBLE, NO COLORED ICONS) =========
# ======================================================================
with tabs[0]:
    st.markdown("## Search")

    # --- Clean input style ---
    st.markdown("""
    <style>
        input[data-testid="stTextInput"]::placeholder {
            color:#999 !important;
        }
        .spacer { margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

    # --- SEARCH INPUT FIELDS ---
    col1, col2, col3, col4, col5 = st.columns([2.2,1.2,1.2,1.2,1.2])

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

    semantic_query = st.text_input(
        "Semantic Search",
        placeholder="e.g. Minimalism collectors or those following Bruce Nauman"
    )

    # Auto-clear search when everything is empty
    if (
        keyword.strip()=="" and city.strip()=="" and country.strip()=="" and
        (tier=="" or tier is None) and role.strip()=="" and
        semantic_query.strip()==""
    ):
        st.session_state["search_results"] = None
        st.session_state["search_page"] = 0

    # ============================================================
    # RUN SEARCH
    # ============================================================
    if st.button("Search Leads") and supabase:
        with st.spinner("Searching..."):

            # ---------------- SEMANTIC SEARCH ----------------
            if semantic_query.strip():
                try:
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    emb = client.embeddings.create(
                        model=os.getenv("EMBEDDING_MODEL","text-embedding-3-large"),
                        input=semantic_query
                    ).data[0].embedding

                    rpc = supabase.rpc(
                        "rpc_semantic_search_leads_supplements",
                        {
                            "query_embedding": list(map(float, emb)),
                            "match_count": 300,
                            "min_score": 0.10
                        }
                    ).execute()

                    results = [{
                        "lead_id": r.get("lead_id"),
                        "full_name": r.get("full_name"),
                        "email": r.get("email"),
                        "tier": r.get("tier"),
                        "primary_role": r.get("primary_role"),
                        "city": r.get("city"),
                        "country": r.get("country"),
                        "notes": r.get("notes"),
                    } for r in (rpc.data or [])]

                except:
                    results = []

            # ---------------- REGULAR SEARCH ----------------
            else:
                try:
                    query = supabase.table("leads").select(
                        "lead_id, full_name, email, tier, primary_role, city, country, notes"
                    )

                    if keyword:
                        q = f"%{keyword}%"
                        query = query.or_(
                            f"full_name.ilike.{q},email.ilike.{q},primary_role.ilike.{q}"
                        )
                    if city:
                        query = query.ilike("city", f"%{city}%")
                    if country:
                        query = query.ilike("country", f"%{country}%")
                    if tier:
                        query = query.eq("tier", tier)
                    if role:
                        query = query.ilike("primary_role", f"%{role}%")

                    results = query.limit(2000).execute().data or []

                except:
                    results = []

        st.session_state["search_results"] = results
        st.session_state["search_page"] = 0

    # ============================================================
    # DETERMINE WHICH GRID TO SHOW
    # ============================================================
    search_results = st.session_state.get("search_results", None)
    show_search_grid = search_results is not None

    # ======================================================================
    # === FULL GRID (NO SEARCH APPLIED)
    # ======================================================================
    if not show_search_grid:

        per_page = 50
        if "full_grid_page" not in st.session_state:
            st.session_state.full_grid_page = 0

        # Count
        try:
            total_response = supabase.table("leads").select("*", count="exact").limit(1).execute()
            total_full = total_response.count or 0
        except:
            total_full = 0

        total_pages = max(1, (total_full + per_page - 1)//per_page)
        offset = st.session_state.full_grid_page * per_page

        # Fetch
        leads = (
            supabase.table("leads")
            .select("lead_id, full_name, email, tier, primary_role, city, country, notes")
            .order("full_name", desc=False)
            .range(offset, offset + per_page - 1)
            .execute()
            .data or []
        )

        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        st.write(f"Showing {len(leads)} of {total_full} collectors")

        left, right = st.columns(2)

        for i, lead in enumerate(leads):
            col = left if i % 2 == 0 else right

            with col:
                name = lead.get("full_name","Unnamed")
                city_val = (lead.get("city") or "").strip()
                label = f"{name} — {city_val}" if city_val else name
                lead_id = str(lead.get("lead_id"))   # ✅ FIXED

                with st.expander(label):

                    # Basic info
                    tier_val = lead.get("tier","—")
                    role_val = lead.get("primary_role","—")
                    email_val = lead.get("email","—")
                    country_val = (lead.get("country") or "").strip()

                    if city_val or country_val:
                        st.caption(f"{city_val}, {country_val}".strip(", "))
                    st.caption(f"{role_val} | Tier {tier_val}")
                    st.write(email_val)

                    # -------- Summarize Button (WORKING) --------
                    sum_col, _ = st.columns([3,1])
                    summary_key = f"summary_{lead_id}"

                    with sum_col:
                        if summary_key not in st.session_state:
                            if st.button(f"Summarize {name}", key=f"sum_{lead_id}"):
                                with st.spinner("Summarizing notes..."):

                                    supplements = (
                                        supabase.table("leads_supplements")
                                        .select("notes")
                                        .eq("lead_id", lead_id)
                                        .execute()
                                        .data or []
                                    )

                                    base_notes = lead.get("notes") or ""
                                    supplement_notes = "\n\n".join(
                                        (s.get("notes") or "").strip()
                                        for s in supplements
                                    )

                                    combined = (
                                        base_notes +
                                        ("\n\n" if base_notes and supplement_notes else "") +
                                        supplement_notes
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
                                            {"role": "system", "content":"You summarize art collectors factually."},
                                            {"role": "user", "content": prompt},
                                        ],
                                        temperature=0.2,
                                        max_tokens=500
                                    )

                                    st.session_state[summary_key] = resp.choices[0].message.content.strip()
                                    st.rerun()

                        else:
                            st.markdown("**Summary:**")
                            st.markdown(st.session_state[summary_key], unsafe_allow_html=True)

        # Pagination
        prev_col, next_col = st.columns([1,1])
        with prev_col:
            if st.button("Prev Page", disabled=st.session_state.full_grid_page==0):
                st.session_state.full_grid_page -= 1
                st.rerun()
        with next_col:
            if st.button("Next Page", disabled=st.session_state.full_grid_page>=total_pages-1):
                st.session_state.full_grid_page += 1
                st.rerun()

    # ======================================================================
    # === SEARCH GRID (WHEN RESULTS EXIST)
    # ======================================================================
    else:
        results = search_results
        per_page = 50

        if "search_page" not in st.session_state:
            st.session_state.search_page = 0

        total_results = len(results)
        total_pages = max(1, (total_results + per_page - 1)//per_page)

        start = st.session_state.search_page * per_page
        end = start + per_page
        page_results = results[start:end]

        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        st.write(f"Showing {len(page_results)} of {total_results} results")

        left, right = st.columns(2)

        for i, lead in enumerate(page_results):
            col = left if i % 2 == 0 else right
            lead_id = str(lead.get("lead_id"))

            with col:
                name = lead.get("full_name","Unnamed")
                city_val = (lead.get("city") or "").strip()
                label = f"{name} — {city_val}" if city_val else name

                with st.expander(label):
                    st.markdown(f"**{name}**")
                    st.caption(f"{lead.get('primary_role','—')} | Tier {lead.get('tier','—')}")
                    st.write(lead.get("email","—"))

        prev_col, next_col = st.columns([1,1])
        with prev_col:
            if st.button("Prev Results", disabled=st.session_state.search_page==0):
                st.session_state.search_page -= 1
                st.rerun()
        with next_col:
            if st.button("Next Results", disabled=st.session_state.search_page>=total_pages-1):
                st.session_state.search_page += 1
                st.rerun()

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
        per_page = 200  # 200 entries per page as requested
        if "data_page" not in st.session_state:
            st.session_state.data_page = 0

        offset = st.session_state.data_page * per_page

        # --- Count total leads ---
        try:
            total_response = (
                supabase.table("leads")
                .select("*", count="exact")
                .limit(1)
                .execute()
            )
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
                .select("full_name, email, tier, primary_role, city, country, notes")
                .order("created_at", desc=True)
                .range(offset, offset + per_page - 1)
                .execute()
                .data
                or []
            )
        except Exception as e:
            st.error(f"Failed to fetch leads: {e}")
            leads = []

        # --- Display leads in spreadsheet-style table (no lead_id column) ---
        if leads:
            df = pd.DataFrame(leads)

            # Reorder columns to something sensible
            desired_cols = ["full_name", "email", "tier", "primary_role", "city", "country", "notes"]
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
                if st.button(
                    "Next",
                    use_container_width=True,
                    disabled=st.session_state.data_page >= total_pages - 1,
                ):
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
# ======================================================================
# === CHAT TAB =========================================================
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

    # --- OpenAI client ---
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # --- Session state initialization ---
    st.session_state.chat_sessions = (
        supabase.table("chat_sessions")
        .select("*")
        .order("id", desc=True)
        .execute()
        .data
    )

    if "active_chat" not in st.session_state:
        st.session_state.active_chat = []

    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None

    # --- Layout ---
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


    # ======================================================================
    # === LEFT COLUMN: CHAT HISTORY ========================================
    # ======================================================================
    with left:
        st.markdown("### Chats")
    
        if not st.session_state.chat_sessions:
            st.info("No previous chats yet.")
        else:
            for i, session in enumerate(st.session_state.chat_sessions):
    
                title = session.get("title", "Untitled chat")
                summary = session.get("summary", "")
    
                # ---- GROUP BUTTON + SUMMARY INTO ONE CARD ----
                with st.container():
                    # Chat title button
                    clicked = st.button(
                        title,
                        key=f"chat_button_{i}",
                        use_container_width=True,
                        type="secondary"
                    )
    
                    # Summary expander (kept attached to the button visually)
                    if summary:
                        with st.expander("Summary", expanded=False):
                            st.markdown(summary)
    
                    # Handle click
                    if clicked:
                        session_id = session["id"]
                        st.session_state.current_session_id = session_id
    
                        msgs = (
                            supabase.table("chat_messages")
                            .select("*")
                            .eq("session_id", session_id)
                            .order("id", desc=False)   # FIXED — no asc=True
                            .execute()
                            .data
                        )
    
                        st.session_state.active_chat = [
                            {"role": m["role"], "content": m["content"]}
                            for m in msgs
                        ]
    
                        st.rerun()

    # ======================================================================
    # === RIGHT COLUMN: ACTIVE CHAT WINDOW =================================
    # ======================================================================
    with right:
        st.markdown("#### Current Chat")

        # Render chat bubbles
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
                    unsafe_allow_html=True,
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
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='clear:both;'></div>", unsafe_allow_html=True)

        # --- Chat Input ---
        user_input = st.chat_input("Ask about collectors, regions, or interests...")

        if user_input:

            # Add locally
            st.session_state.active_chat.append({"role": "user", "content": user_input})

            # Save message to Supabase
            if st.session_state.current_session_id:
                supabase.table("chat_messages").insert({
                    "session_id": st.session_state.current_session_id,
                    "role": "user",
                    "content": user_input,
                }).execute()

            # AI response
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
                    st.session_state.active_chat.append({"role": "assistant", "content": response_text})

                    # Save assistant response
                    if st.session_state.current_session_id:
                        supabase.table("chat_messages").insert({
                            "session_id": st.session_state.current_session_id,
                            "role": "assistant",
                            "content": response_text,
                        }).execute()

                    st.rerun()

                except Exception as e:
                    st.error(f"Chat failed: {e}")


        # ==================================================================
        # === NEW CHAT BUTTON =============================================
        # ==================================================================
        if st.session_state.active_chat:
            st.divider()

            if st.button("New Chat", use_container_width=True):

                # Extract user text for summary
                preview_text = " ".join(
                    [m["content"] for m in st.session_state.active_chat if m["role"] == "user"]
                )[:2000]

                # ---------- TITLE ----------
                try:
                    title_prompt = (
                        "Summarize the conversation topic in 3–5 plain words. "
                        "No punctuation, no emojis.\n\n"
                        f"{preview_text}"
                    )
                    title_resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": title_prompt}],
                        max_tokens=15,
                    )
                    title_text = title_resp.choices[0].message.content.strip()
                except:
                    title_text = "Untitled chat"

                # ---------- BULLET SUMMARY ----------
                try:
                    summary_prompt = (
                        "Write a clean bullet-point summary of the user's conversation.\n"
                        "- Use 3–6 bullets.\n"
                        "- Keep each bullet short.\n\n"
                        f"{preview_text}"
                    )
                    summary_resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": summary_prompt}],
                        max_tokens=200,
                    )
                    summary_text = summary_resp.choices[0].message.content.strip()
                except:
                    summary_text = "- No summary available."

                # Save session to Supabase
                result = (
                    supabase.table("chat_sessions")
                    .insert({"title": title_text, "summary": summary_text})
                    .execute()
                )

                new_session_id = result.data[0]["id"]
                st.session_state.current_session_id = new_session_id

                # Reset active messages
                st.session_state.active_chat = []

                # Refresh sidebar
                st.session_state.chat_sessions = (
                    supabase.table("chat_sessions")
                    .select("*")
                    .order("id", desc=True)
                    .execute()
                    .data
                )

                st.rerun()
