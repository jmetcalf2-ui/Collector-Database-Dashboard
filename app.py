import streamlit as st
from supabase_client import get_supabase
from components import inject_css
import os
from openai import OpenAI

# --- Page setup ---
st.set_page_config(page_title="Dashboard", layout="wide")
inject_css()

# --- Sidebar (minimal) ---
with st.sidebar:
    st.write(" ")

# --- Initialize session state ---
if "selected_leads" not in st.session_state:
    st.session_state.selected_leads = []

# --- Main content ---
st.markdown("<h1>Dashboard</h1>", unsafe_allow_html=True)

try:
    supabase = get_supabase()
    st.success("Connected to Supabase")

    # === Tabs ===
    tabs = st.tabs(["Search", "Saved Sets", "Chat"])

    # === SEARCH TAB ===
    with tabs[0]:
        st.markdown("## Search")

        # --- Collector Lookup ---
        with st.container():
            st.markdown("### Collector Lookup")

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

                data = query.limit(100).execute().data or []

                if data:
                    st.write(f"Found {len(data)} results")

                    for lead in data:
                        with st.expander(f"{lead.get('full_name', 'Unnamed')} — {lead.get('city', 'Unknown')}"):
                            c1, c2 = st.columns([8, 1])
                            with c1:
                                st.write(f"**Email:** {lead.get('email','—')}")
                                st.write(f"**Tier:** {lead.get('tier','—')}")
                                st.write(f"**Role:** {lead.get('primary_role','—')}")
                                st.write(f"**Notes:** {lead.get('notes','')[:250]}")
                            with c2:
                                chk_key = f"chk_{lead['lead_id']}"
                                checked = st.checkbox("Select", key=chk_key, value=lead["lead_id"] in st.session_state.selected_leads)
                                if checked and lead["lead_id"] not in st.session_state.selected_leads:
                                    st.session_state.selected_leads.append(lead["lead_id"])
                                elif not checked and lead["lead_id"] in st.session_state.selected_leads:
                                    st.session_state.selected_leads.remove(lead["lead_id"])
                else:
                    st.info("No leads found matching your filters.")
            else:
                st.empty()

            # --- Save to Set modal ---
            if st.session_state.selected_leads:
                st.markdown("---")
                st.markdown(f"**{len(st.session_state.selected_leads)} collectors selected.**")

                with st.popover("Save Selected"):
                    st.markdown("### Save Selected Leads")
                    mode = st.radio("Choose option:", ["Add to existing set", "Create new set"])

                    if mode == "Add to existing set":
                        existing_sets = (
                            supabase.table("saved_sets").select("id, name").execute().data or []
                        )
                        set_names = {s["name"]: s["id"] for s in existing_sets}
                        chosen = st.selectbox("Select set", list(set_names.keys()) if set_names else [])
                        if chosen and st.button("Add"):
                            sid = set_names[chosen]
                            for lid in st.session_state.selected_leads:
                                supabase.table("saved_set_items").insert({"set_id": sid, "lead_id": lid}).execute()
                            st.success(f"Added {len(st.session_state.selected_leads)} leads to {chosen}")
                            st.session_state.selected_leads = []

                    elif mode == "Create new set":
                        new_name = st.text_input("New set name")
                        new_desc = st.text_area("Description (optional)")
                        if new_name and st.button("Create and Save"):
                            r = (
                                supabase.table("saved_sets")
                                .insert({"name": new_name, "description": new_desc})
                                .execute()
                            )
                            sid = r.data[0]["id"]
                            for lid in st.session_state.selected_leads:
                                supabase.table("saved_set_items").insert({"set_id": sid, "lead_id": lid}).execute()
                            st.success(f"Created {new_name} and added {len(st.session_state.selected_leads)} leads.")
                            st.session_state.selected_leads = []

        # --- Divider ---
        st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

        # --- Semantic Search ---
        with st.container():
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
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

                    emb = client.embeddings.create(model=model, input=query_text).data[0].embedding
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
            else:
                st.empty()

    # === SAVED SETS TAB ===
    with tabs[1]:
        st.markdown("## Saved Sets")

        sets_data = supabase.table("saved_sets").select("*").order("created_at", desc=True).execute().data or []

        if not sets_data:
            st.info("No saved sets found. Use the Search tab to create one.")
        else:
            for s in sets_data:
                with st.expander(f"{s['name']}"):
                    st.write(f"**Description:** {s.get('description', '—')}")
                    st.write(f"**Created:** {s.get('created_at', '—')}")

                    members = (
                        supabase.table("saved_set_items")
                        .select("lead_id, leads(full_name, city, tier, primary_role)")
                        .eq("set_id", s["id"])
                        .execute()
                        .data
                        or []
                    )

                    if members:
                        member_df = [
                            {
                                "Name": m["leads"]["full_name"],
                                "City": m["leads"]["city"],
                                "Tier": m["leads"]["tier"],
                                "Role": m["leads"]["primary_role"],
                            }
                            for m in members if m.get("leads")
                        ]
                        st.dataframe(member_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("This set is empty.")

                    with st.expander("Manage Set"):
                        new_name = st.text_input(f"Rename '{s['name']}'", value=s["name"], key=f"rename_{s['id']}")
                        new_desc = st.text_area("Edit description", value=s.get("description", ""), key=f"desc_{s['id']}")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Save Changes", key=f"save_{s['id']}"):
                                supabase.table("saved_sets").update(
                                    {"name": new_name, "description": new_desc}
                                ).eq("id", s["id"]).execute()
                                st.success("Saved set updated successfully.")
                        with col2:
                            if st.button("Delete Set", key=f"delete_{s['id']}"):
                                supabase.table("saved_set_items").delete().eq("set_id", s["id"]).execute()
                                supabase.table("saved_sets").delete().eq("id", s["id"]).execute()
                                st.warning("Set deleted.")
                                st.experimental_rerun()

# === CHAT TAB ===
with tabs[2]:
    st.markdown("## Chat with Collector Intelligence")

    from services.rag import answer_with_context
    from pathlib import Path

    # Load system prompt text
    system_prompt_path = Path("prompts/system_prompt.md")
    system_prompt = system_prompt_path.read_text().strip() if system_prompt_path.exists() else (
        "You are a helpful art-market assistant. Answer based only on the provided context."
    )

    # Persistent chat memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render previous exchanges
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box (Streamlit ≥1.31)
    if prompt := st.chat_input("Ask about collectors, regions, or interests..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consulting database and reasoning..."):
                try:
                    res = answer_with_context(
                        supabase,
                        question=prompt,
                        system_prompt=system_prompt,
                        match_count=10,
                        min_similarity=0.15,
                    )
                    st.markdown(res["answer"])
                    if res["sources"]:
                        with st.expander("Sources used"):
                            for s in res["sources"]:
                                name = s.get("full_name", "Unknown")
                                loc = ", ".join(
                                    filter(None, [s.get("city"), s.get("country")])
                                )
                                snippet = (s.get("notes") or "")[:250]
                                st.markdown(f"**{name}** ({loc})  \n{snippet}")
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": res["answer"]}
                    )
                except Exception as e:
                    st.error(f"Chat failed: {e}")


except Exception as e:
    st.error(f"Connection failed: {e}")
