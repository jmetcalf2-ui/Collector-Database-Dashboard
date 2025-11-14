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
# === SEARCH TAB — DEFAULT FULL LIST + SEARCH + PAGINATION =============
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

    # -------------------------------
    # SEARCH INPUT FIELDS
    # -------------------------------
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

    semantic_query = st.text_input(
        "Semantic Search",
        placeholder="e.g. Minimalism collectors or those following Bruce Nauman",
    )

    # -------------------------------
    # SESSION STATE
    # -------------------------------
    if "search_mode" not in st.session_state:
        st.session_state.search_mode = False  # False = show all collectors
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "search_message" not in st.session_state:
        st.session_state.search_message = ""

    # Pagination state
    if "search_page" not in st.session_state:
        st.session_state.search_page = 0
    if "default_page" not in st.session_state:
        st.session_state.default_page = 0

    PER_PAGE = 200

    # -------------------------------
    # SEARCH BUTTON
    # -------------------------------
    if st.button("Search Leads"):
        st.session_state.search_mode = True
        st.session_state.search_page = 0  # reset search pagination

        with st.spinner("Searching..."):
            # ---- SEMANTIC SEARCH ----
            if semantic_query.strip():
                try:
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    emb = client.embeddings.create(
                        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
                        input=semantic_query,
                    ).data[0].embedding

                    rpc = supabase.rpc(
                        "rpc_semantic_search_leads_supplements",
                        {
                            "query_embedding": list(map(float, emb)),
                            "match_count": 500,
                            "min_score": 0.12,
                        },
                    ).execute()

                    normalized = []
                    for row in rpc.data or []:
                        normalized.append({
                            "lead_id": row.get("lead_id") or row.get("id"),
                            "full_name": row.get("full_name"),
                            "email": row.get("email"),
                            "tier": row.get("tier"),
                            "primary_role": row.get("primary_role"),
                            "city": row.get("city"),
                            "country": row.get("country"),
                            "notes": row.get("notes"),
                        })

                    st.session_state.search_results = normalized
                    st.session_state.search_message = "Showing semantic matches"

                except Exception as e:
                    st.error("Semantic search failed.")
                    st.code(str(e))

            # ---- REGULAR SEARCH ----
            else:
                try:
                    q = supabase.table("leads").select(
                        "lead_id, full_name, email, tier, primary_role, city, country, notes"
                    )

                    if keyword:
                        w = f"%{keyword}%"
                        q = q.or_(
                            f"full_name.ilike.{w},email.ilike.{w},primary_role.ilike.{w}"
                        )
                    if city:
                        q = q.ilike("city", f"%{city}%")
                    if country:
                        q = q.ilike("country", f"%{country}%")
                    if tier:
                        q = q.eq("tier", tier)
                    if role:
                        q = q.ilike("primary_role", f"%{role}%")

                    data = q.limit(500).execute().data or []

                    st.session_state.search_results = data
                    st.session_state.search_message = "Showing filtered results"

                except Exception as e:
                    st.error("Search failed.")
                    st.code(str(e))

    # -------------------------------
    # CLEAR SEARCH BUTTON
    # -------------------------------
    if st.session_state.search_mode:
        if st.button("Clear Search"):
            st.session_state.search_mode = False
            st.session_state.search_results = []
            st.session_state.search_message = ""
            st.session_state.search_page = 0

    # ======================================================================
    # === FETCH DATA FOR DISPLAY (DEFAULT OR SEARCH MODE)
    # ======================================================================

    if st.session_state.search_mode:
        # SEARCH MODE PAGINATION
        results_all = st.session_state.search_results
        total = len(results_all)

        start = st.session_state.search_page * PER_PAGE
        end = start + PER_PAGE
        results = results_all[start:end]

    else:
        # DEFAULT MODE (SHOW ALL COLLECTORS)
        start = st.session_state.default_page * PER_PAGE
        end = start + PER_PAGE

        try:
            results = (
                supabase.table("leads")
                .select("lead_id, full_name, email, tier, primary_role, city, country, notes")
                .order("full_name")
                .range(start, end - 1)
                .execute()
                .data or []
            )

            # Count
            count_resp = supabase.table("leads").select("*", count="exact").limit(1).execute()
            total = getattr(count_resp, "count", 0)

            st.success(f"Showing {len(results)} of {total} collectors")

        except Exception as e:
            st.error("Failed to load collectors")
            st.code(str(e))
            results = []

    # ======================================================================
    # === RESULTS GRID — TWO COLUMNS (CONTACT-STYLE)
    # ======================================================================
    if results:
        left, right = st.columns(2)
        for i, lead in enumerate(results):
            col = left if i % 2 == 0 else right
            with col:
                name = lead.get("full_name") or "Unnamed"
                tier_val = lead.get("tier") or "—"
                role_val = lead.get("primary_role") or "—"
                email_val = lead.get("email") or "—"
                city_val = (lead.get("city") or "").strip()
                country_val = (lead.get("country") or "").strip()

                lead_id = str(lead.get("lead_id"))
                summary_key = f"summary_{lead_id}"

                label = name if not city_val else f"{name} — {city_val}"

                with st.expander(label):

                    st.markdown(f"**{name}**")
                    if city_val or country_val:
                        st.caption(f"{city_val}, {country_val}".strip(", "))
                    st.caption(f"{role_val} | Tier {tier_val}")
                    st.write(email_val)

                    # Row: summarize / delete
                    c1, c2 = st.columns([3,1])

                    with c1:
                        if summary_key not in st.session_state:
                            if st.button(f"Summarize {name}", key=f"summarize_{lead_id}"):
                                with st.spinner("Summarizing..."):
                                    try:
                                        supp = (
                                            supabase.table("leads_supplements")
                                            .select("notes")
                                            .eq("lead_id", lead_id)
                                            .execute().data or []
                                        )
                                        base = lead.get("notes") or ""
                                        extra = "\n\n".join((s.get("notes") or "") for s in supp)
                                        combined = (base + "\n\n" + extra).strip()

                                        summary = summarize_collector(lead_id, combined)
                                        st.session_state[summary_key] = summary
                                        st.rerun()
                                    except:
                                        st.error("Summarization failed")
                        else:
                            st.markdown("**Notes:**")
                            st.markdown(st.session_state[summary_key], unsafe_allow_html=True)

                    with c2:
                        if st.button("Delete", key=f"del_{lead_id}"):
                            st.session_state[f"confirm_{lead_id}"] = True

                        if st.session_state.get(f"confirm_{lead_id}", False):
                            st.warning(f"Delete {name}?")
                            yes = st.button("Confirm", key=f"confirm_del_{lead_id}")
                            no = st.button("Cancel", key=f"cancel_del_{lead_id}")

                            if yes:
                                supabase.table("leads").delete().eq("lead_id", lead_id).execute()
                                st.success("Deleted.")
                                st.rerun()
                            if no:
                                st.session_state[f"confirm_{lead_id}"] = False
                                st.rerun()

    # ======================================================================
    # === # ======================================================================
# === SEARCH TAB (FINAL VERSION WITH AUTO-CLEAR + PAGINATION) ==========
# ======================================================================
with tabs[0]:
    st.markdown("## Search")

    st.markdown("""
    <style>
    input[data-testid="stTextInput"]::placeholder {
        color: #888 !important;
        opacity: 1 !important;
    }
    .search-spacer { margin-top: 18px; }
    </style>
    """, unsafe_allow_html=True)

    # --- INPUT FIELDS ---------------------------------------------------
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

    semantic_query = st.text_input(
        "Semantic Search",
        placeholder="e.g. Minimalism collectors or those following Bruce Nauman",
    )

    # ------------------------------------------------------------------
    # AUTO-CLEAR SEARCH WHEN FIELDS ARE EMPTY
    # ------------------------------------------------------------------
    if (
        keyword.strip() == "" 
        and city.strip() == "" 
        and country.strip() == "" 
        and role.strip() == "" 
        and (tier == "" or tier is None)
        and semantic_query.strip() == ""
    ):
        st.session_state["search_results"] = None

    # ------------------------------------------------------------------
    # SEARCH LOGIC
    # ------------------------------------------------------------------
    if st.button("Search Leads") and supabase:
        with st.spinner("Searching..."):

            # ----- SEMANTIC SEARCH --------------------------------------
            if semantic_query.strip():
                try:
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    emb = client.embeddings.create(
                        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
                        input=semantic_query,
                    ).data[0].embedding

                    rpc = supabase.rpc(
                        "rpc_semantic_search_leads_supplements",
                        {
                            "query_embedding": list(map(float, emb)),
                            "match_count": 200,
                            "min_score": 0.10,
                        },
                    ).execute()

                    results = [
                        {
                            "lead_id": r.get("lead_id"),
                            "full_name": r.get("full_name"),
                            "email": r.get("email"),
                            "tier": r.get("tier"),
                            "primary_role": r.get("primary_role"),
                            "city": r.get("city"),
                            "country": r.get("country"),
                            "notes": r.get("notes"),
                        }
                        for r in (rpc.data or [])
                    ]

                except Exception as e:
                    st.error("Semantic search failed.")
                    st.code(str(e))
                    results = []

            # ----- REGULAR SEARCH ----------------------------------------
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

                    results = query.limit(200).execute().data or []

                except Exception as e:
                    st.error("Search failed.")
                    st.code(str(e))
                    results = []

        # save to session state
        st.session_state["search_results"] = results

    # ------------------------------------------------------------------
    # GET CURRENT RESULTS
    # ------------------------------------------------------------------
    results = st.session_state.get("search_results", None)

    # ------------------------------------------------------------------
    # PAGINATION
    # ------------------------------------------------------------------
    if results:
        per_page = 20

        if "search_page" not in st.session_state:
            st.session_state.search_page = 0

        total_results = len(results)
        total_pages = max(1, (total_results + per_page - 1) // per_page)

        start = st.session_state.search_page * per_page
        end = start + per_page
        page_results = results[start:end]

        # ------------------------------------------------------------------
        # RESULTS HEADER
        # ------------------------------------------------------------------
        st.markdown(f"<div class='search-spacer'></div>", unsafe_allow_html=True)
        st.write(f"Showing {len(page_results)} of {total_results} results")

        # ------------------------------------------------------------------
        # GRID DISPLAY (2 COLUMNS)
        # ------------------------------------------------------------------
        left_col, right_col = st.columns(2)

        for i, lead in enumerate(page_results):
            col = left_col if i % 2 == 0 else right_col

            with col:
                name = lead.get("full_name", "Unnamed")
                tier_val = lead.get("tier", "—")
                role_val = lead.get("primary_role", "—")
                email_val = lead.get("email", "—")
                city_val = (lead.get("city") or "").strip()
                country_val = (lead.get("country") or "").strip()

                lead_id = str(lead.get("lead_id"))
                summary_key = f"summary_{lead_id}"

                label = f"{name} — {city_val}" if city_val else name

                with st.expander(label):
                    st.markdown(f"**{name}**")
                    if city_val or country_val:
                        st.caption(f"{city_val}, {country_val}".strip(", "))
                    st.caption(f"{role_val} | Tier {tier_val}")
                    st.write(email_val)

                    sum_col, del_col = st.columns([3, 1])

                    # SUMMARY
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
                                        base_notes
                                        + ("\n\n" if base_notes and supplement_notes else "")
                                        + supplement_notes
                                    ).strip()
                                    summary = summarize_collector(lead_id, combined)
                                    st.session_state[summary_key] = summary
                                    st.rerun()
                        else:
                            st.markdown("**Notes:**")
                            st.markdown(st.session_state[summary_key], unsafe_allow_html=True)

                    # DELETE
                    with del_col:
                        if st.button("Delete", key=f"del_{lead_id}"):
                            st.session_state[f"delete_{lead_id}"] = True

                        if st.session_state.get(f"delete_{lead_id}", False):
                            st.warning(f"Delete {name}?")
                            if st.button("Yes", key=f"yes_{lead_id}"):
                                supabase.table("leads").delete().eq("lead_id", lead_id).execute()
                                st.session_state[f"delete_{lead_id}"] = False
                                st.session_state["search_results"] = None
                                st.rerun()
                            if st.button("Cancel", key=f"cancel_{lead_id}"):
                                st.session_state[f"delete_{lead_id}"] = False
                                st.rerun()

        # ------------------------------------------------------------------
        # PAGINATION BUTTONS
        # ------------------------------------------------------------------
        prev_col, next_col = st.columns([1, 1])

        with prev_col:
            if st.button("Prev", disabled=st.session_state.search_page == 0):
                st.session_state.search_page -= 1
                st.rerun()

        with next_col:
            if st.button(
                "Next", disabled=st.session_state.search_page >= total_pages - 1
            ):
                st.session_state.search_page += 1
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
