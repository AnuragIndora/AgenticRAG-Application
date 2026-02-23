import streamlit as st
import requests
import json

# ----------------------------
# CONFIG
# ----------------------------
BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Agentic RAG Assistant", layout="wide")

# ----------------------------
# SESSION STATE
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# TITLE
# ----------------------------
st.title("🤖 Agentic RAG Assistant")

# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.header("Controls")

    # Clear chat button
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.subheader("📂 Upload File")

    uploaded_file = st.file_uploader(
        "Upload document or CSV",
        type=["pdf", "docx", "pptx", "txt", "md", "csv", "xlsx"]
    )

    if uploaded_file:
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        response = requests.post(f"{BACKEND_URL}/upload", files=files)

        if response.status_code == 200:
            st.success(response.json()["message"])
        else:
            st.error(response.text)

    st.divider()

    if st.button("⚠ Reset Database"):
        response = requests.delete(f"{BACKEND_URL}/reset")
        if response.status_code == 200:
            st.success("Database cleared")
        else:
            st.error(response.text)

# ----------------------------
# DISPLAY CHAT HISTORY
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# CHAT INPUT
# ----------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Send request to backend
    payload = {"query": user_input}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(f"{BACKEND_URL}/query", json=payload)

            if response.status_code == 200:
                result = response.json()
                answer = result["answer"]

                st.markdown(answer)

                # Optional expandable debug info
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

                # Add assistant message to memory
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            else:
                st.error(response.text)