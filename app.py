import streamlit as st
from supabase_client import get_supabase
from components import inject_css

# --- Page setup ---
st.set_page_config(page_title="Dashboard", layout="wide")
inject_css()

# --- Sidebar (minimal) ---
with st.sidebar:
    st.write(" ")

# --- Main content ---
st.markdown("<h1>Dashboard</h1>", unsafe_allow_html=True)

try:
    supabase = get_supabase()
    st.success("Connected to Supabase")

    # === Collector Lookup Section ===
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Collector Lookup")

    # Search filters
    col1, col2, col3, col4, col5 = st.columns([2, 1.2, 1.2, 1.2, 1.2])
    with col1:
        keyword = st.text_input("Keyword", placeholder="Name, email, interests, etc.")
    with col2:
        city = st.text_input("City")
    with col3:
        country = st.text_input("Country")
    with col4:
        tier = st.selectbox("Tier", ["", "1", "2", "3"], index=0)
    with col5:
        role = st.text_input("Primary Role")

    search_button = st.button("Search Leads")

    results = []
    if search_button:
        query = supabase.table("leads").select("*")
        if keyword:
            query = query.ilike("full_name", f"%{keyword}%")
        if city:
            query = query.ilike("city", f"%{city}%")
        if country:
            query = query.ilike("country", f"%{country}%")
        if tier:
            query = query.eq("tier", int(tier))
        if role:
            query = query.ilike("primary_role", f"%{role}%")

        data = query.limit(100).execute()
        results = data.data or []

        if results:
            st.write(f"Found {len(results)} results")
            st.dataframe(results, use_container_width=True, hide_index=True)
        else:
            st.info("No leads found matching your filters.")

    st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Connection failed: {e}")
