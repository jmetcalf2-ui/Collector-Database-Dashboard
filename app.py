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
    st.markdown("<div class='card' style='margin-top: 1.5rem;'>", unsafe_allow_html=True)
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
        tier = st.selectbox("Tier", ["", "A", "B", "C"], index=0)
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
            query = query.eq("tier", tier)
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

    # === Semantic Search Section ===
    st.markdown("<div class='card' style='margin-top: 2rem;'>", unsafe_allow_html=True)
    st.markdown("### Semantic Search (Notes & Content)")

    query_text = st.text_input(
        "Describe the type of collector or interest you’re looking for",
        placeholder="e.g. Minimalism collectors or those following Bruce Nauman"
    )
    top_k = st.slider("Number of results", 5, 100, 25)
    min_similarity = st.slider("Minimum similarity threshold", 0.0, 1.0, 0.15, 0.01)
    run_semantic = st.button("Run Semantic Search")

    if run_semantic and query_text.strip():
        with st.spinner("Embedding query and searching..."):
            from openai import OpenAI
            import os

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

            # Create the query embedding
            emb = client.embeddings.create(model=model, input=query_text).data[0].embedding

            # Call your Supabase function
            res = supabase.rpc(
                "semantic_search_lead_supplements",
                {
                    "query_embedding": emb,
                    "match_count": top_k,
                    "min_similarity": min_similarity
                },
                schema="ai"
            ).execute()

            results = res.data or []
            if results:
                st.write(f"Found {len(results)} semantic matches")
                st.dataframe(results, use_container_width=True, hide_index=True)
            else:
                st.info("No semantic matches found for that query.")

    st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Connection failed: {e}")
