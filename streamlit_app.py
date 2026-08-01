"""
streamlit_app.py
==================
The deployed assistant. Any user can type a personal finance question —
this is general knowledge retrieval, not tied to any individual's data.

Run locally:
    streamlit run streamlit_app.py
(requires ./chroma_db to already exist — run 05_create_chroma_store.py first)
"""

from importlib import import_module

import streamlit as st

retrieve_context = import_module("06_retrieve_context")
prompting = import_module("07_prompting")

# --- API key resolution: environment variable first, then Streamlit secrets ---
try:
    if not prompting.OPENROUTER_API_KEY:
        prompting.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
        prompting.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", prompting.OPENROUTER_MODEL)
except Exception:
    pass

st.set_page_config(page_title="Personal Finance Assistant", page_icon="💰")
st.title("💰 Personal Finance Assistant")
st.caption("Ask a question about budgeting, credit, saving, or debt. Answers are grounded in cited sources.")

query = st.text_input("Your question", placeholder="e.g. How can I lower my grocery bill?")

if st.button("Ask", type="primary") and query.strip():
    with st.spinner("Retrieving relevant sources..."):
        package = retrieve_context.build_context_package(query)

    if package["num_sources"] == 0:
        st.warning("No sufficiently relevant source was found in the knowledge base for this question.")
    else:
        with st.spinner("Generating answer..."):
            answer = prompting.generate_answer(query, package["context_text"])

        st.subheader("Answer")
        st.write(answer)

        with st.expander(f"View {package['num_sources']} cited source(s)"):
            for row in package["selected"]:
                status = "🟢 Current" if row["is_current"] else "🟡 Outdated"
                st.markdown(f"**{row['title']}** — {row['effective_date']} — {status}")
                st.write(row["chunk_text"])
                st.divider()
