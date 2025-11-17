import streamlit as st
from supabase_client import get_supabase
from components import inject_css
import os
from openai import OpenAI
from datetime import datetime
from pathlib import Path
import pandas as pd

# ============================================================
# PAGE CONFIG & GLOBAL STYLE
# ============================================================
st.set_page_config(page_title="Collector Intelligence Dashboard", layout="wide")
inject_css()

# --- Global CSS Theme (Hybrid: Supabase + ChatGPT + Notion) ---
st.markdown(
    """
<style>
/* Overall background */
.main {
    background: radial-gradient(circle at top left, #1f2937 0, #020617 55%, #020617 100%);
}

/* Center content, constrain width */
.block-container {
    max-width: 1200px;
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
}

/* App header */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.25rem;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(15,23,42,0.98));
    border: 1px solid rgba(148,163,184,0.35);
    box-shadow: 0 18px 40px rgba(0,0,0,0.45);
    margin-bottom: 1.5rem;
}
.app-header-left {
    display: flex;
    flex-direction: column;
}
.app-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #e5e7eb;
}
.app-subtitle {
    font-size: 0.9rem;
    color: #9ca3af;
}
.app-header-right {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.app-pill {
    font-size: 0.75rem;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.45);
    color: #9ca3af;
}

/* Stat cards row */
.stat-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0,1fr));
    gap: 0.9rem;
    margin-bottom: 1.2rem;
}
.stat-card {
    background: linear-gradient(145deg, #020617, #020617);
    border-radius: 14px;
    padding: 0.75rem 0.95rem;
    border: 1px solid rgba(148,163,184,0.3);
    box-shadow: 0 10px 30px rgba(15,23,42,0.7);
}
.stat-label {
    font-size: 0.78rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: #9ca3af;
}
.stat-value {
    font-size: 1.4rem;
    font-weight: 600;
    color: #e5e7eb;
    margin-top: 0.15rem;
}
.stat-caption {
    font-size: 0.78rem;
    color: #6b7280;
}

/* Card shells used in tabs */
.section-card {
    background: rgba(15,23,42,0.98);
    border-radius: 18px;
    padding: 1.1rem 1.1rem 1.1rem 1.1rem;
    border: 1px solid rgba(148,163,184,0.4);
    box-shadow: 0 16px 35px rgba(0,0,0,0.6);
    margin-bottom: 1rem;
}
.section-title {
    font-size: 1.0rem;
    font-weight: 500;
    color: #e5e7eb;
    margin-bottom: 0.4rem;
}
.section-caption {
    font-size: 0.82rem;
    color: #9ca3af;
    margin-bottom: 0.8rem;
}

/* Tab labels (top nav) */
button[role="tab"] {
    border-radius: 999px !important;
    padding: 0.3rem 0.9rem !important;
    margin-right: 0.25rem !important;
    color: #e5e7eb !important;
    background-color: rgba(15,23,42,0.7) !important;
    border: 1px solid rgba(148,163,184,0.5) !important;
    font-size: 0.85rem !important;
}
button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #22c55e, #6366f1) !important;
    border-color: transparent !important;
    color: #020617 !important;
}

/* Inputs */
div[data-baseweb="input"] input, textarea, select, .stTextInput input {
    background-color: rgba(15,23,42,0.95) !important;
    color: #e5e7eb !important;
    border-radius: 999px !important;
    border: 1px solid rgba(148,163,184,0.7) !important;
}
label {
    color: #cbd5f5 !important;
    font-size: 0.8rem !important;
}

/* Buttons */
div[data-testid="stButton"] > button {
    border-radius: 999px !important;
    padding: 0.35rem 0.9rem !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    border: 1px solid rgba(148,163,184,0.7) !important;
    background: linear-gradient(135deg, #0f172a, #020617) !important;
    color: #e5e7eb !important;
}
div[data-testid="stButton"] > button:hover {
    border-color: #22c55e !important;
    box-shadow: 0 0 0 1px rgba(34,197,94,0.5);
}

/* Dataframe tweaks (contacts) */
.stDataFrame {
    background-color: #020617 !important;
    border-radius: 14px !important;
    border: 1px solid rgba(148,163,184,0.6) !important;
}

/* Chat message bubbles */
.chat-user {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    color: #e5e7eb;
}
.chat-assistant {
    background: linear-gradient(135deg, #111827, #020617);
    color: #e5e7eb;
}

/* Expander styling for collectors */
.streamlit-expanderHeader {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #e5e7eb !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR (Re-activated, minimal)
# ============================================================
with st.sidebar:
    st.markdown("### Collector Intelligence")
    st.caption("Private internal tool for research and outreach.")
    st.markdown("---")
    st.markdown("**Quick stats**")
    st.caption("Live from Supabase")
    # We'll fill the metrics after we can call get_total_leads_count; for now a placeholder.
    # They will be updated in main area with true values.

# ============================================================
# SESSION STATE SETUP
# ============================================================
if "selected_leads" not in st.session_state:
    st.session_state.selected_leads = []
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []
if "active_chat" not in st.session_state:
    st.session_state.active_chat = []

# ============================================================
# SUPABASE CONNECTION
# ============================================================
@st.cache_resource(show_spinner=False)
def load_supabase():
    return get_supabase()

try:
    supabase = load_supabase()
except Exception as e:
    st.error(f"⚠️ Supabase connection failed: {e}")
    supabase = None

# ============================================================
# CACHED HELPERS
# ============================================================
@st.cache_data(show_spinner=False)
def get_total_leads_count() -> int:
    if not supabase:
        return 0
    try:
        res = supabase.table("leads").select("*", count="exact").limit(1).execute()
        return getattr(res, "count", 0) or 0
    except Exception:
        return 0


@st.cache_data(show_spinner=False)
def get_full_grid_page(page_index: int, per_page: int):
    if not supabase:
        return []
    offset = page_index * per_page
    try:
        res = (
            supabase.table("leads")
            .select("lead_id, full_name, email, tier, primary_role, city, country, notes")
            .order("full_name", desc=False)
            .range(offset, offset + per_page - 1)
            .execute()
        )
        return res.data or []
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def get_contacts_page(page_index: int, per_page: int):
    if not supabase:
        return []
    offset = page_index * per_page
    try:
        res = (
            supabase.table("leads")
            .select("full_name, email, tier, primary_role, city, country, notes")
            .order("created_at", desc=True)
            .range(offset, offset + per_page - 1)
            .execute()
        )
        return res.data or []
    except Exception:
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
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def get_query_embedding_cached(query: str):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Missing OPENAI_API_KEY")
    client = OpenAI(api_key=key)
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    return emb


@st.cache_data(show_spinner=False)
def summarize_collector(lead_id: str, combined_notes: str) -> str:
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
Focus on specifics:
- Artists collected or recently purchased
- Museum / institutional boards or affiliations
- Geography (city / region)
- Collecting tendencies or philanthropy
- Notable sales, acquisitions, or foundations

Avoid value adjectives ('important', 'renowned', etc).

NOTES:
{combined_notes}
"""
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize art collectors factually and concisely."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.25,
            max_tokens=450,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"


# ============================================================
# TOP HEADER + STATS
# ============================================================
total_leads = get_total_leads_count()
today = datetime.now().strftime("%b %d, %Y")

st.markdown(
    f"""
<div class="app-header">
  <div class="app-header-left">
    <div class="app-title">Collector Intelligence Dashboard</div>
    <div class="app-subtitle">A private workspace for researching collectors, segmenting contacts, and chatting with your data.</div>
  </div>
  <div class="app-header-right">
    <div class="app-pill">Updated {today}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Stat cards
st.markdown(
    f"""
<div class="stat-row">
  <div class="stat-card">
    <div class="stat-label">Total Leads</div>
    <div class="stat-value">{total_leads}</div>
    <div class="stat-caption">Unique collectors in Supabase</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Workspace Mode</div>
    <div class="stat-value">Research</div>
    <div class="stat-caption">Search, filter, and export contacts</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">AI Engine</div>
    <div class="stat-value">gpt-4o-mini</div>
    <div class="stat-caption">Used for summaries & chat</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# MAIN TABS
# ============================================================
tabs = st.tabs(["Search", "Contacts", "Saved Sets", "Chat"])

# ============================================================
# SEARCH TAB (polished)
# ============================================================
with tabs[0]:
    st.markdown(
        """
<div class="section-card">
  <div class="section-title">Lead Search</div>
  <div class="section-caption">
    Filter by name, geography, tier, or role — or use semantic search to query your RAG notes for concepts like
    <span style="color:#a5b4fc;">“Minimalism collectors”</span> or <span style="color:#a5b4fc;">“Nauman board members”.</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.container():
        col1, col2, col3, col4, col5 = st.columns([2.2, 1.2, 1.2, 1.0, 1.4])
        with col1:
            keyword = st.text_input("Keyword", placeholder="Name, email, notes…")
        with col2:
            city = st.text_input("City", placeholder="e.g. New York")
        with col3:
            country = st.text_input("Country", placeholder="e.g. United States")
        with col4:
            tier = st.selectbox("Tier", ["", "A", "B", "C"], index=0)
        with col5:
            role = st.text_input("Primary Role", placeholder="Collector, curator…")

    st.markdown(
        """
<div class="section-card" style="margin-top:0.8rem;">
  <div class="section-title">Semantic Search</div>
  <div class="section-caption">
    Type a natural language query to search RAG notes. The app embeds your query and calls your Supabase RPC.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    semantic_query = st.text_input(
        "Semantic Search",
        placeholder="e.g. Minimalism collectors or those following Bruce Nauman",
        key="semantic_search_input",
    )

    # Auto-clear search when everything is blank
    if (
        keyword.strip() == ""
        and city.strip() == ""
        and country.strip() == ""
        and (tier == "" or tier is None)
        and role.strip() == ""
        and semantic_query.strip() == ""
    ):
        st.session_state["search_results"] = None
        st.session_state["search_page"] = 0

    # Run search
    run_search = st.button("Run Search", key="search_leads_button")

    if run_search and supabase:
        with st.spinner("Searching collectors…"):
            # Semantic branch
            if semantic_query.strip():
                try:
                    emb = get_query_embedding_cached(semantic_query.strip())
                    rpc = supabase.rpc(
                        "rpc_semantic_search_leads_supplements",
                        {
                            "query_embedding": emb,
                            "match_count": 300,
                            "min_score": 0.10,
                        },
                    ).execute()
                    rows = rpc.data or []
                    ids = [r.get("lead_id") for r in rows if r.get("lead_id")]

                    if ids:
                        res = (
                            supabase.table("leads")
                            .select(
                                "lead_id, full_name, email, tier, primary_role, city, country, notes"
                            )
                            .in_("lead_id", ids)
                            .execute()
                        )
                        st.session_state["search_results"] = res.data or []
                    else:
                        st.session_state["search_results"] = []
                except Exception as e:
                    st.error(f"Semantic search error: {e}")
                    st.session_state["search_results"] = []
            else:
                # Regular filters
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

                    res = q.limit(2000).execute()
                    st.session_state["search_results"] = res.data or []
                except Exception as e:
                    st.error(f"Search error: {e}")
                    st.session_state["search_results"] = []

        st.session_state["search_page"] = 0

    # Decide which grid to show
    search_results = st.session_state.get("search_results", None)
    show_search_grid = search_results is not None

    # ----------------------- FULL GRID (no filters) -----------------------
    if not show_search_grid and supabase:
        per_page = 40
        if "full_grid_page" not in st.session_state:
            st.session_state.full_grid_page = 0

        total_full = total_leads
        total_pages = max(1, (total_full + per_page - 1) // per_page)

        leads = get_full_grid_page(st.session_state.full_grid_page, per_page)

        st.markdown(
            f"<div class='section-caption'>Showing {len(leads)} of {total_full} collectors</div>",
            unsafe_allow_html=True,
        )

        left_col, right_col = st.columns(2)
        for i, lead in enumerate(leads):
            col = left_col if i % 2 == 0 else right_col

            name = lead.get("full_name", "Unnamed")
            city_val = (lead.get("city") or "").strip()
            label = f"{name} — {city_val}" if city_val else name
            lead_id = str(lead.get("lead_id"))

            with col:
                with st.expander(label):
                    tier_val = lead.get("tier", "—")
                    role_val = lead.get("primary_role", "—")
                    email_val = lead.get("email", "—")
                    country_val = (lead.get("country") or "").strip()

                    st.markdown(f"**{name}**")
                    st.caption(
                        f"{role_val} · Tier {tier_val} · {(city_val + ', ' if city_val else '')}{country_val}"
                    )
                    st.write(email_val)

                    sum_col, _ = st.columns([3, 1])
                    summary_key = f"summary_{lead_id}"

                    with sum_col:
                        if summary_key not in st.session_state:
                            if st.button(
                                f"Summarize collector", key=f"sum_full_{lead_id}"
                            ):
                                with st.spinner("Summarizing notes…"):
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
                                        + (
                                            "\n\n"
                                            if base_notes and supplement_notes
                                            else ""
                                        )
                                        + supplement_notes
                                    ).strip()

                                    st.session_state[summary_key] = summarize_collector(
                                        lead_id, combined
                                    )
                                    st.rerun()
                        else:
                            st.markdown("**Summary**")
                            st.markdown(
                                st.session_state[summary_key],
                                unsafe_allow_html=True,
                            )

        # Pagination
        col_space_left, prev_col, next_col, col_space_right = st.columns(
            [2, 1, 1, 2]
        )
        with prev_col:
            if st.button(
                "Previous page", disabled=st.session_state.full_grid_page == 0
            ):
                st.session_state.full_grid_page -= 1
                st.rerun()
        with next_col:
            if st.button(
                "Next page",
                disabled=st.session_state.full_grid_page >= total_pages - 1,
            ):
                st.session_state.full_grid_page += 1
                st.rerun()

    # ----------------------- SEARCH GRID (results) -----------------------
    elif show_search_grid:
        results = search_results or []
        per_page = 40

        if "search_page" not in st.session_state:
            st.session_state.search_page = 0

        total_results = len(results)
        total_pages = max(1, (total_results + per_page - 1) // per_page)

        start = st.session_state.search_page * per_page
        end = start + per_page
        page_results = results[start:end]

        st.markdown(
            f"<div class='section-caption'>Showing {len(page_results)} of {total_results} matching collectors</div>",
            unsafe_allow_html=True,
        )

        left_col, right_col = st.columns(2)
        for i, lead in enumerate(page_results):
            col = left_col if i % 2 == 0 else right_col
            lead_id = str(lead.get("lead_id"))

            name = lead.get("full_name", "Unnamed")
            city_val = (lead.get("city") or "").strip()
            label = f"{name} — {city_val}" if city_val else name

            with col:
                with st.expander(label):
                    tier_val = lead.get("tier", "—")
                    role_val = lead.get("primary_role", "—")
                    email_val = lead.get("email", "—")
                    country_val = (lead.get("country") or "").strip()

                    st.markdown(f"**{name}**")
                    st.caption(
                        f"{role_val} · Tier {tier_val} · {(city_val + ', ' if city_val else '')}{country_val}"
                    )
                    st.write(email_val)

                    sum_col, _ = st.columns([3, 1])
                    summary_key = f"summary_{lead_id}"

                    with sum_col:
                        if summary_key not in st.session_state:
                            if st.button(
                                f"Summarize collector", key=f"sum_search_{lead_id}"
                            ):
                                with st.spinner("Summarizing notes…"):
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
                                        + (
                                            "\n\n"
                                            if base_notes and supplement_notes
                                            else ""
                                        )
                                        + supplement_notes
                                    ).strip()

                                    st.session_state[summary_key] = summarize_collector(
                                        lead_id, combined
                                    )
                                    st.rerun()
                        else:
                            st.markdown("**Summary**")
                            st.markdown(
                                st.session_state[summary_key],
                                unsafe_allow_html=True,
                            )

        prev_col, next_col = st.columns([1, 1])
        with prev_col:
            if st.button(
                "Previous results", disabled=st.session_state.search_page == 0
            ):
                st.session_state.search_page -= 1
                st.rerun()
        with next_col:
            if st.button(
                "Next results",
                disabled=st.session_state.search_page >= total_pages - 1,
            ):
                st.session_state.search_page += 1
                st.rerun()

# ============================================================
# CONTACTS TAB (with filter + CSV, redesigned)
# ============================================================
with tabs[1]:
    st.markdown(
        """
<div class="section-card">
  <div class="section-title">Contacts</div>
  <div class="section-caption">
    Maintain a clean spreadsheet-style view of every lead. Filter by any visible field and export the current slice as CSV.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # --- Create a new contact form ---
    with st.expander("Create a contact", expanded=False):
        with st.form("create_contact_form"):
            st.markdown("Enter details to add a new record to the leads table:")

            col_a, col_b = st.columns(2)
            with col_a:
                full_name = st.text_input("Full Name")
                primary_role = st.text_input("Primary Role")
                city = st.text_input("City")
            with col_b:
                email = st.text_input("Email")
                country = st.text_input("Country")
                tier = st.selectbox("Tier", ["A", "B", "C", "—"], index=3)

            notes = st.text_area("Notes", height=100)

            submitted = st.form_submit_button("Create Contact")
            if submitted:
                if not full_name or not email:
                    st.warning("Please provide at least a name and an email.")
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

    st.markdown("<br>", unsafe_allow_html=True)

    if not supabase:
        st.warning("Database unavailable.")
    else:
        # Filter + Export controls
        st.markdown(
            """
<div class="section-card" style="margin-bottom:0.8rem;">
  <div class="section-title">Filter & Export</div>
  <div class="section-caption">
    This search runs locally on the visible page. Use it to quickly carve out a CSV for outreach or analysis.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        filter_text = st.text_input(
            "Filter visible contacts by any field",
            placeholder="Type to filter: names, countries, tiers, roles, notes…",
            key="contacts_filter_text",
        )
        export_btn = st.button(
            "Export filtered results as CSV",
            key="contacts_export_btn",
        )

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

        # Pagination & data
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

            filtered_df = filter_dataframe(df, filter_text)

            st.dataframe(filtered_df, use_container_width=True)

            # Export CSV
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

            st.markdown("---")
            col_space_left, col_prev, col_next, col_space_right = st.columns(
                [2, 1, 1, 2]
            )
            with col_prev:
                if st.button(
                    "Previous page",
                    use_container_width=True,
                    disabled=st.session_state.data_page == 0,
                ):
                    st.session_state.data_page -= 1
                    st.rerun()
            with col_next:
                if st.button(
                    "Next page",
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
    st.markdown(
        """
<div class="section-card">
  <div class="section-title">Saved Sets</div>
  <div class="section-caption">
    Curated lists of collectors for specific campaigns, previews, or regional pushes. Each set can be exported or used as a targeting list.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

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
# CHAT TAB (CollectorGPT panel)
# ============================================================
with tabs[3]:
    st.markdown(
        """
<div class="section-card">
  <div class="section-title">CollectorGPT Chat</div>
  <div class="section-caption">
    Ask questions about collectors, segments, and strategy. This panel uses the same model configuration as your summaries.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    system_prompt_path = Path("prompts/system_prompt.md")
    if system_prompt_path.exists():
        system_prompt = system_prompt_path.read_text().strip()
    else:
        system_prompt = (
            "You are CollectorGPT — a helpful art-market assistant. "
            "Keep responses factual, concise, and well-reasoned."
        )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Load sessions
    st.session_state.chat_sessions = load_chat_sessions()

    if "active_chat" not in st.session_state:
        st.session_state.active_chat = []
    if "current_chat_open" not in st.session_state:
        st.session_state.current_chat_open = None
    if "current_session_title" not in st.session_state:
        st.session_state.current_session_title = ""
    if "current_session_summary" not in st.session_state:
        st.session_state.current_session_summary = ""

    left, right = st.columns([2.4, 6.6], gap="large")

    # ----------------------- LEFT: Chat list -----------------------
    with left:
        st.markdown("#### History")

        if not st.session_state.chat_sessions:
            st.info("No previous chats yet.")
        else:
            for i, session in enumerate(st.session_state.chat_sessions):
                title = session.get("title", "Untitled chat")
                summary = session.get("summary", "")
                session_id = session["id"]

                clicked = st.button(
                    title,
                    key=f"chat_btn_{i}",
                    use_container_width=True,
                )

                if clicked:
                    if st.session_state.current_chat_open == session_id:
                        st.session_state.current_chat_open = None
                        st.session_state.current_session_summary = ""
                        st.session_state.current_session_title = ""
                    else:
                        st.session_state.current_chat_open = session_id
                        st.session_state.current_session_summary = summary
                        st.session_state.current_session_title = title
                        st.session_state.active_chat = []
                    st.rerun()

    # ----------------------- RIGHT: Chat body -----------------------
    with right:
        if st.session_state.current_chat_open is not None:
            st.markdown(
                f"### {st.session_state.current_session_title}",
            )
            st.markdown(st.session_state.current_session_summary)
        else:
            st.markdown("### Current Chat")

            # Render messages
            for msg in st.session_state.active_chat:
                css_class = "chat-user" if msg["role"] == "user" else "chat-assistant"
                align = "right" if msg["role"] == "user" else "left"

                st.markdown(
                    f"""
<div style="
    background: transparent;
    display:flex;
    justify-content:{align};
    margin: 4px 0;
">
  <div class="{css_class}" style="
      padding:10px 14px;
      border-radius:14px;
      max-width:78%;
      border:1px solid rgba(148,163,184,0.5);
      font-size:0.9rem;
  ">
    {msg["content"]}
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

            user_input = st.chat_input(
                "Ask about collectors, regions, interests, or strategy…"
            )

            if user_input:
                st.session_state.active_chat.append(
                    {"role": "user", "content": user_input}
                )

                if st.session_state.current_chat_open and supabase:
                    supabase.table("chat_messages").insert(
                        {
                            "session_id": st.session_state.current_chat_open,
                            "role": "user",
                            "content": user_input,
                        }
                    ).execute()

                with st.spinner("Thinking…"):
                    try:
                        messages = [{"role": "system", "content": system_prompt}]
                        messages.extend(st.session_state.active_chat)

                        completion = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages,
                            temperature=0.5,
                            max_tokens=600,
                        )
                        response_text = completion.choices[0].message.content.strip()

                        st.session_state.active_chat.append(
                            {"role": "assistant", "content": response_text}
                        )

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

            if st.session_state.active_chat:
                st.divider()
                if st.button("Save as new chat", use_container_width=True):
                    preview_text = " ".join(
                        [
                            m["content"]
                            for m in st.session_state.active_chat
                            if m["role"] == "user"
                        ]
                    )[:2000]

                    try:
                        title_prompt = (
                            "Summarize the conversation topic in 3–5 plain words.\n"
                            "No emojis, no punctuation.\n\n"
                            f"{preview_text}"
                        )
                        title_resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": title_prompt}],
                            max_tokens=15,
                        )
                        title_text = title_resp.choices[0].message.content.strip()
                    except Exception:
                        title_text = "Untitled chat"

                    try:
                        summary_prompt = (
                            "Write a clean bullet-point summary of the user's conversation.\n"
                            "- Use 3–6 bullets.\n"
                            "- Keep them short.\n\n"
                            f"{preview_text}"
                        )
                        summary_resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": summary_prompt}],
                            max_tokens=200,
                        )
                        summary_text = summary_resp.choices[0].message.content.strip()
                    except Exception:
                        summary_text = "- No summary available."

                    if supabase:
                        result = (
                            supabase.table("chat_sessions")
                            .insert({"title": title_text, "summary": summary_text})
                            .execute()
                        )
                        load_chat_sessions.clear()
                        new_session_id = result.data[0]["id"]
                        st.session_state.current_chat_open = new_session_id

                    st.session_state.active_chat = []
                    st.session_state.chat_sessions = load_chat_sessions()
                    st.rerun()
