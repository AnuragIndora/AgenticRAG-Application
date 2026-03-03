import streamlit as st
import requests
import os

BACKEND_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
st.set_page_config(page_title="Agentic RAG Assistant", layout="wide")

# -- Session state ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -- Page title ---------------------------------------------------------------
st.title("🤖 Agentic RAG Assistant")

# -- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("📂 Upload File")

    uploaded_file = st.file_uploader(
        "Upload document or CSV",
        type=["pdf", "docx", "pptx", "txt", "md", "csv", "xlsx"],
    )

    if uploaded_file:
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        resp = requests.post(f"{BACKEND_URL}/upload", files=files)
        if resp.status_code == 200:
            st.success(resp.json()["message"])
        else:
            st.error(resp.text)

    st.divider()

    if st.button("⚠ Reset Database"):
        resp = requests.delete(f"{BACKEND_URL}/reset")
        if resp.status_code == 200:
            st.success("Database cleared")
        else:
            st.error(resp.text)

# -- Chat history -------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -- Chat input ---------------------------------------------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = requests.post(f"{BACKEND_URL}/query", json={"query": user_input})

            if resp.status_code == 200:
                result = resp.json()
                answer = result["answer"]
                st.markdown(answer)

                # Expandable debug panel — hidden by default to keep the UI clean
                with st.expander("🔍 Details"):
                    st.write("**Intent:**", result.get("intent"))
                    st.write("**Confidence:**", result.get("confidence_score"))
                    st.write("**Reasoning Steps:**", result.get("reasoning_steps"))

                    if result.get("sql_query"):
                        st.code(result["sql_query"], language="sql")

                    if result.get("sources"):
                        st.write("**Sources:**")
                        for s in result["sources"]:
                            st.write(
                                f"- {s.get('file_name')} | page {s.get('page_number')} | score {s.get('score')}"
                            )

                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error(resp.text)