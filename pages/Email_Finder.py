import streamlit as st
import pandas as pd
import re
from datetime import datetime
from supabase_client import get_supabase
from components import inject_css
import os

st.set_page_config(page_title="Email Finder", layout="wide")
inject_css()

# Initialize Supabase
@st.cache_resource(show_spinner=False)
def load_supabase():
    return get_supabase()

try:
    supabase = load_supabase()
except Exception as e:
    st.error(f"Supabase connection failed: {e}")
    supabase = None

# Initialize session state
if 'email_searches' not in st.session_state:
    st.session_state.email_searches = []
if 'verified_emails' not in st.session_state:
    st.session_state.verified_emails = []

# Helper functions
def extract_domain(url):
    """Extract domain from URL"""
    pattern = r'(?:https?://)?(?:www\.)?([^/]+)'
    match = re.search(pattern, url)
    return match.group(1) if match else url

def generate_email_patterns(domain, first_name, last_name):
    """Generate common email patterns"""
    patterns = [
        f"{first_name.lower()}.{last_name.lower()}@{domain}",
        f"{first_name.lower()}{last_name.lower()}@{domain}",
        f"{first_name[0].lower()}{last_name.lower()}@{domain}",
        f"{first_name.lower()}@{domain}",
        f"{last_name.lower()}@{domain}",
        f"{first_name[0].lower()}.{last_name.lower()}@{domain}",
    ]
    return patterns

def mock_email_verification(email):
    """Mock email verification"""
    import random
    statuses = ['Valid', 'Valid', 'Valid', 'Risky', 'Invalid']
    return random.choice(statuses)

def search_domain_in_database(domain):
    """Search for contacts with emails from a specific domain"""
    if not supabase:
        return []
    
    try:
        domain_clean = domain.lower().strip()
        res = supabase.table("leads").select(
            "lead_id, full_name, email, tier, primary_role, city, country"
        ).ilike("email", f"%@{domain_clean}%").limit(100).execute()
        return res.data or []
    except Exception as e:
        st.error(f"Database search error: {e}")
        return []

with st.sidebar:
    st.write(" ")

# Header
st.markdown("<h1>Email Finder</h1>", unsafe_allow_html=True)

# Sidebar stats
with st.sidebar:
    st.markdown("### Dashboard")
    
    total_contacts = 0
    if supabase:
        try:
            res = supabase.table("leads").select("*", count="exact").limit(1).execute()
            total_contacts = getattr(res, "count", 0)
        except:
            pass
    
    st.metric("Total Contacts", total_contacts)
    st.metric("Searches Today", len(st.session_state.email_searches))
    st.metric("Emails Verified", len(st.session_state.verified_emails))
    
    st.markdown("---")
    st.markdown("### Tools")
    tool_option = st.radio(
        "Select Tool",
        ["Domain Search", "Email Finder", "Email Verifier", "Bulk Lookup"],
        label_visibility="collapsed"
    )

# Main content
if tool_option == "Domain Search":
    st.markdown("## Domain Search")
    st.caption("Find all email addresses associated with a domain in your database")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        domain_input = st.text_input("Domain", placeholder="example.com", label_visibility="collapsed")
    with col2:
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    if search_button and domain_input:
        domain = extract_domain(domain_input)
        st.session_state.email_searches.append({
            "domain": domain, 
            "timestamp": datetime.now()
        })
        
        with st.spinner("Searching database..."):
            results = search_domain_in_database(domain)
        
        if results:
            st.write(f"Found {len(results)} contacts with @{domain} emails")
            
            # Display results in expandable cards
            left_col, right_col = st.columns(2)
            
            for i, contact in enumerate(results):
                col = left_col if i % 2 == 0 else right_col
                
                name = contact.get('full_name', 'Unnamed')
                city_val = (contact.get('city') or '').strip()
                label = f"{name} — {city_val}" if city_val else name
                
                with col:
                    with st.expander(label):
                        st.markdown(f"**{name}**")
                        tier_val = contact.get('tier', '—')
                        role_val = contact.get('primary_role', '—')
                        email_val = contact.get('email', '—')
                        country_val = (contact.get('country') or '').strip()
                        
                        if city_val or country_val:
                            st.caption(f"{city_val}, {country_val}".strip(', '))
                        st.caption(f"{role_val} | Tier {tier_val}")
                        st.write(email_val)
        else:
            st.info(f"No contacts found with @{domain} emails in your database")

elif tool_option == "Email Finder":
    st.markdown("## Email Finder")
    st.caption("Find someone's email address using their name and company domain")
    
    col1, col2 = st.columns(2)
    with col1:
        first_name = st.text_input("First Name", placeholder="John")
        domain = st.text_input("Company Domain", placeholder="example.com")
    with col2:
        last_name = st.text_input("Last Name", placeholder="Smith")
        
    find_button = st.button("Find Email", type="primary", use_container_width=True)
    
    if find_button and first_name and last_name and domain:
        domain = extract_domain(domain)
        patterns = generate_email_patterns(domain, first_name, last_name)
        
        st.markdown("### Possible Email Patterns")
        
        left_col, right_col = st.columns(2)
        
        for i, email in enumerate(patterns[:6]):
            col = left_col if i % 2 == 0 else right_col
            status = mock_email_verification(email)
            
            with col:
                with st.expander(f"Pattern {i+1}"):
                    st.code(email, language=None)
                    st.caption(f"Status: {status}")
                    
                    if status == "Valid":
                        st.success("Likely deliverable")
                    elif status == "Risky":
                        st.warning("Verification uncertain")
                    else:
                        st.error("Invalid format")

elif tool_option == "Email Verifier":
    st.markdown("## Email Verifier")
    st.caption("Verify if an email address is valid and deliverable")
    
    email_to_verify = st.text_input("Email Address", placeholder="john.smith@example.com")
    verify_button = st.button("Verify Email", type="primary", use_container_width=True)
    
    if verify_button and email_to_verify:
        with st.spinner("Verifying email..."):
            status = mock_email_verification(email_to_verify)
            st.session_state.verified_emails.append({
                "email": email_to_verify,
                "status": status,
                "timestamp": datetime.now()
            })
        
        st.markdown("### Verification Results")
        
        with st.expander(email_to_verify, expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                if status == "Valid":
                    st.success(f"Status: {status}")
                elif status == "Risky":
                    st.warning(f"Status: {status}")
                else:
                    st.error(f"Status: {status}")
            
            with col2:
                st.caption("Verification Details")
                st.write("Format: Valid")
                st.write("Domain: Active")
                st.write(f"SMTP: {'Verified' if status == 'Valid' else 'Risky'}")
                st.write(f"Deliverable: {'Yes' if status == 'Valid' else 'Maybe'}")

else:  # Bulk Lookup
    st.markdown("## Bulk Lookup")
    st.caption("Upload a CSV file to find or verify multiple emails at once")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], label_visibility="collapsed")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("Process Bulk Task", type="primary"):
            st.success(f"Processing {len(df)} entries...")
            st.info("This feature would process all entries and return results with actual verification APIs.")
    else:
        st.info("Upload a CSV with columns: first_name, last_name, domain OR email")

st.markdown("---")
st.caption("Email verification results are simulated. For production use, integrate with verification APIs like Hunter.io, ZeroBounce, or NeverBounce.")
