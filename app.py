import streamlit as st
from supabase_client import get_supabase
from components import inject_css
import os
from openai import OpenAI
from datetime import datetime
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Dashboard", layout="wide")
inject_css()

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

if "selected_leads" not in st.session_state:
    st.session_state.selected_leads = []
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []
if "active_chat" not in st.session_state:
    st.session_state.active_chat = []

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

st.markdown("<h1>Dashboard</h1>", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_supabase():
    return get_supabase()

try:
    supabase = load_supabase()
except Exception as e:
    st.error(f"⚠️ Supabase connection failed: {e}")
    supabase = None

@st.cache_data(show_spinner=False)
def get_total_leads_count():
    if not supabase:
        return 0
    try:
        res = supabase.table("leads").select("*", count="exact").limit(1).execute()
        return getattr(res, "count", 0)
    except:
        return 0

@st.cache_data(show_spinner=False)
def get_full_grid_page(page, per_page):
    if not supabase:
        return []
    offset = page * per_page
    try:
        res = (
            supabase.table("leads")
            .select("lead_id, full_name, email, tier, primary_role, city, country, notes")
            .order("full_name", desc=False)
            .range(offset, offset + per_page - 1)
            .execute()
        )
        return res.data or []
    except:
        return []

@st.cache_data(show_spinner=False)
def get_contacts_page(page, per_page):
    if not supabase:
        return []
    offset = page * per_page
    try:
        res = (
            supabase.table("leads")
            .select("full_name, email, tier, primary_role, city, country, notes")
            .order("created_at", desc=True)
            .range(offset, offset + per_page - 1)
            .execute()
        )
        return res.data or []
    except:
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
    except:
        return []

@st.cache_data(show_spinner=False)
def get_query_embedding_cached(query):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Missing key")
    client = OpenAI(api_key=key)
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    return emb

@st.cache_data(show_spinner=False)
def summarize_collector(lead_id, combined_notes):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return "⚠️ Missing OPENAI_API_KEY"
    if not combined_notes.strip():
        return "⚠️ No notes found"

    try:
        client = OpenAI(api_key=key)
        prompt = f"""
Summarize collector notes into 4–6 bullet points.
Avoid adjectives. Focus on concrete data.

NOTES:
{combined_notes}
"""
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize collectors factually."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"

tabs = st.tabs(["Search", "Contacts", "Saved Sets", "Chat"])

# ========================
# SEARCH TAB
# ========================
with tabs[0]:
    st.markdown("## Search")

    st.markdown("""
    <style>
    input[data-testid="stTextInput"]::placeholder {
        color:#999 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns([2.2,1.2,1.2,1.2,1.2])
    with col1:
        keyword = st.text_input("Keyword")
    with col2:
        city = st.text_input("City")
    with col3:
        country = st.text_input("Country")
    with col4:
        tier = st.selectbox("Tier", ["", "A", "B", "C"])
    with col5:
        role = st.text_input("Primary Role")

    semantic_query = st.text_input("Semantic Search")

    if (
        keyword.strip()=="" and city.strip()=="" and country.strip()=="" and
        (tier=="" or tier is None) and role.strip()=="" and
        semantic_query.strip()==""  
    ):
        st.session_state["search_results"] = None
        st.session_state["search_page"] = 0

    if st.button("Search Leads") and supabase:
        with st.spinner("Searching..."):

            if semantic_query.strip():
                try:
                    emb = get_query_embedding_cached(semantic_query.strip())
                    rpc = supabase.rpc(
                        "rpc_semantic_search_leads_supplements",
                        {
                            "query_embedding": emb,
                            "match_count": 300,
                            "min_score": 0.10
                        }
                    ).execute()
                    rows = rpc.data or []
                    ids = [r["lead_id"] for r in rows if r.get("lead_id")]
                    if ids:
                        res = (
                            supabase.table("leads")
                            .select("lead_id, full_name, email, tier, primary_role, city, country, notes")
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
                try:
                    q = supabase.table("leads").select(
                        "lead_id, full_name, email, tier, primary_role, city, country, notes"
                    )
                    if keyword:
                        w = f"%{keyword}%"
                        q = q.or_(f"full_name.ilike.{w},email.ilike.{w},primary_role.ilike.{w}")
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
                except:
                    st.session_state["search_results"] = []

        st.session_state["search_page"] = 0

    # ==========================================
    # SHOW SEARCH OR FULL GRID
    # ==========================================
    search_results = st.session_state.get("search_results", None)
    show_search_grid = search_results is not None

    # ------------------------------
    # FULL GRID (NO SEARCH APPLIED)
    # ------------------------------
    if not show_search_grid and supabase:
        per_page = 50
        if "full_grid_page" not in st.session_state:
            st.session_state.full_grid_page = 0

        total_full = get_total_leads_count()
        total_pages = max(1, (total_full + per_page - 1) // per_page)

        leads = get_full_grid_page(st.session_state.full_grid_page, per_page)

        st.write(f"Showing {len(leads)} of {total_full} collectors")

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

                    if city_val or country_val:
                        st.caption(f"{city_val}, {country_val}".strip(", "))
                    st.caption(f"{role_val} | Tier {tier_val}")
                    st.write(email_val)

                    sum_col, _ = st.columns([3, 1])
                    summary_key = f"summary_{lead_id}"

                    with sum_col:
                        if summary_key not in st.session_state:
                            if st.button(f"Summarize {name}", key=f"sum_full_{lead_id}"):
                                with st.spinner("Summarizing notes..."):
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
                                        + ("\n\n" if base_notes and supplement_notes else "")
                                        + supplement_notes
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
                                            {"role": "system", "content": "You summarize art collectors factually."},
                                            {"role": "user", "content": prompt},
                                        ],
                                        temperature=0.2,
                                        max_tokens=500,
                                    )

                                    st.session_state[summary_key] = (
                                        resp.choices[0].message.content.strip()
                                    )
                                    st.rerun()
                        else:
                            st.markdown("**Summary:**")
                            st.markdown(st.session_state[summary_key], unsafe_allow_html=True)

        col_space_left, prev_col, next_col, col_space_right = st.columns([2, 1, 1, 2])
        with prev_col:
            if st.button("Prev Page", disabled=st.session_state.full_grid_page == 0):
                st.session_state.full_grid_page -= 1
                st.rerun()
        with next_col:
            if st.button(
                "Next Page",
                disabled=st.session_state.full_grid_page >= total_pages - 1,
            ):
                st.session_state.full_grid_page += 1
                st.rerun()

    # ------------------------------
    # SEARCH GRID (HAS RESULTS)
    # ------------------------------
    elif show_search_grid:
        results = search_results or []
        per_page = 50

        if "search_page" not in st.session_state:
            st.session_state.search_page = 0

        total_results = len(results)
        total_pages = max(1, (total_results + per_page - 1) // per_page)

        start = st.session_state.search_page * per_page
        end = start + per_page
        page_results = results[start:end]

        st.write(f"Showing {len(page_results)} of {total_results} results")

        left_col, right_col = st.columns(2)

        for i, lead in enumerate(page_results):
            col = left_col if i % 2 == 0 else right_col
            lead_id = str(lead.get("lead_id"))

            name = lead.get("full_name", "Unnamed")
            city_val = (lead.get("city") or "").strip()
            label = f"{name} — {city_val}" if city_val else name

            with col:
                with st.expander(label):
                    st.markdown(f"**{name}**")
                    st.caption(f"{lead.get('primary_role', '—')} | Tier {lead.get('tier', '—')}")
                    st.write(lead.get("email", "—"))

                    sum_col, _ = st.columns([3, 1])
                    summary_key = f"summary_{lead_id}"

                    with sum_col:
                        if summary_key not in st.session_state:
                            if st.button(f"Summarize {name}", key=f"sum_search_{lead_id}"):
                                with st.spinner("Summarizing notes..."):
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
                                        + ("\n\n" if base_notes and supplement_notes else "")
                                        + supplement_notes
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
                                            {"role": "system", "content": "You summarize art collectors factually."},
                                            {"role": "user", "content": prompt},
                                        ],
                                        temperature=0.2,
                                        max_tokens=500,
                                    )

                                    st.session_state[summary_key] = (
                                        resp.choices[0].message.content.strip()
                                    )
                                    st.rerun()
                        else:
                            st.markdown("**Summary:**")
                            st.markdown(st.session_state[summary_key], unsafe_allow_html=True)

        prev_col, next_col = st.columns([1, 1])
        with prev_col:
            if st.button("Prev Results", disabled=st.session_state.search_page == 0):
                st.session_state.search_page -= 1
                st.rerun()
        with next_col:
            if st.button(
                "Next Results",
                disabled=st.session_state.search_page >= total_pages - 1,
            ):
                st.session_state.search_page += 1
                st.rerun()

# ============================================================
# CONTACTS TAB
# ============================================================
with tabs[1]:
    st.markdown("## Contacts")

    # --- Create a new contact form ---
    with st.expander("Create a Contact", expanded=False):
        with st.form("create_contact_form"):
            st.markdown("Enter contact details to add a new record to the leads table:")

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

    # --- Spacing before filter + table ---
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    if not supabase:
        st.warning("Database unavailable.")
    else:
        # --------------------------
        # Filter + Export controls
        # --------------------------
        st.markdown("### Filter Contacts")

        filter_text = st.text_input(
            "Search contacts by any visible field",
            placeholder="e.g. Nauman, Minimalism, France, curator…",
            key="contacts_filter_text",
        )

        export_btn = st.button("Export Filtered Results as CSV", key="contacts_export_btn")

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

        # --------------------------
        # Pagination + data fetch
        # --------------------------
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

            # Apply filter
            filtered_df = filter_dataframe(df, filter_text)

            # Show table
            st.dataframe(filtered_df, use_container_width=True)

            # Export CSV for filtered rows
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

            # Pagination controls
            st.markdown("---")
            col_space_left, col_prev, col_next, col_space_right = st.columns(
                [2, 1, 1, 2]
            )
            with col_prev:
                if st.button(
                    "Previous",
                    use_container_width=True,
                    disabled=st.session_state.data_page == 0,
                ):
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

# ============================================================
# SAVED SETS TAB
# ============================================================
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

# ============================================================
# CHAT TAB
# ============================================================
with tabs[3]:
    system_prompt_path = Path("prompts/system_prompt.md")
    if system_prompt_path.exists():
        system_prompt = system_prompt_path.read_text().strip()
    else:
        system_prompt = (
            "You are CollectorGPT — a helpful art-market assistant. "
            "Keep responses factual, concise, and well-reasoned."
        )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    with left:
        st.markdown("### Chats")

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
                    type="secondary",
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

    with right:
        if st.session_state.current_chat_open is not None:
            st.markdown(f"### {st.session_state.current_session_title}")
            st.markdown(st.session_state.current_session_summary)
        else:
            st.markdown("### Current Chat")

            for msg in st.session_state.active_chat:
                if msg["role"] == "user":
                    st.markdown(
                        f"""
                        <div style="
                            background-color:#f5f5f5;
                            padding:10px 14px;
                            border-radius:12px;
                            margin:6px 0;
                            text-align:right;
                            max-width:75%;
                            float:right;
                            clear:both;">
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
                            padding:10px 14px;
                            border-radius:12px;
                            margin:6px 0;
                            text-align:left;
                            max-width:75%;
                            float:left;
                            clear:both;">
                            {msg["content"]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='clear:both;'></div>", unsafe_allow_html=True)

            user_input = st.chat_input("Ask about collectors, regions, or interests...")

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

                with st.spinner("Thinking..."):
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

                if st.button("New Chat", use_container_width=True):
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

