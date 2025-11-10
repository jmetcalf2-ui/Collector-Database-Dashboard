import streamlit as st

def inject_css():
    """Inject global styles from styles.css"""
    with open("styles.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
