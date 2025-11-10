import streamlit as st
from supabase_client import get_supabase

st.set_page_config(page_title="Dashboard", layout="wide")

# --- Sidebar (blank for now) ---
with st.sidebar:
    st.write(" ")

# --- Main content ---
st.markdown("<h1>Dashboard</h1>", unsafe_allow_html=True)

try:
    supabase = get_supabase()
    data = supabase.table("leads").select("lead_id", "full_name").limit(5).execute().data
    st.success("Connected to Supabase")
    st.write("Sample Leads:")
    st.dataframe(data)
except Exception as e:
    st.error(f"Connection failed: {e}")
