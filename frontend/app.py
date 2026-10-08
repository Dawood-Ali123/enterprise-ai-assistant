import streamlit as st
import requests
import uuid
# =============================
# Configuration
# =============================

BACKEND_URL = "http://backend:8000"
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# =============================
# Page Configuration
# =============================

st.set_page_config(
    page_title="Enterprise AI Assistant",
    page_icon="🤖",
    layout="wide"
)


# =============================
# Header
# =============================

st.title("🤖 Enterprise AI Assistant")
st.caption(
    "Secure Company Knowledge & Analytics Assistant"
)


# =============================
# Sidebar
# =============================

with st.sidebar:
    st.title("🏢 Enterprise AI Assistant")
    if st.button("🆕 New Chat"):
       st.session_state.session_id = str(uuid.uuid4())
       st.session_state.messages=[]
       st.rerun()

    st.divider()

    st.subheader("💬 Conversation History")

    try:
        memory_response = requests.get(
            f"{BACKEND_URL}/memory/{st.session_state.session_id}",
            params={"limit": 20},
            timeout=30
        )

        if memory_response.status_code == 200:

            memory_data = memory_response.json()

            messages = memory_data.get("messages", [])

            if messages:

                for message in messages:

                    role = message.get("role")
                    content = message.get("content", "")

                    if role == "user":
                        st.markdown(
                            f"👤 **You:** {content}"
                        )

                    elif role == "assistant":
                        st.markdown(
                            f"🤖 **AI:** {content}"
                        )

            else:
                st.caption("No conversation history yet.")

        else:
            st.caption("Unable to load conversation history.")

    except requests.exceptions.RequestException:
        st.caption("Memory service unavailable.")

    st.header("📄 Company Documents")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        st.write(
            f"Selected: **{uploaded_file.name}**"
        )

        if st.button(
            "📤 Upload & Process",
            use_container_width=True
        ):

            try:

                # -----------------------------
                # Step 1: Upload
                # -----------------------------

                with st.spinner("Uploading document..."):

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    }

                    upload_response = requests.post(
                        f"{BACKEND_URL}/documents/upload",
                        files=files,
                        timeout=120
                    )

                if upload_response.status_code != 200:

                    st.error(
                        f"Upload failed: "
                        f"{upload_response.status_code}"
                    )

                else:

                    upload_data = upload_response.json()

                    document_id = upload_data["document_id"]

                    st.success("✅ Document uploaded")

                    # -----------------------------
                    # Step 2: Process
                    # -----------------------------

                    with st.spinner(
                        "Processing PDF, creating chunks "
                        "and generating embeddings..."
                    ):

                        process_response = requests.post(
                            f"{BACKEND_URL}/documents/"
                            f"{document_id}/process",
                            timeout=600
                        )

                    if process_response.status_code != 200:

                        st.error(
                            f"Processing failed: "
                            f"{process_response.status_code}"
                        )

                    else:

                        process_data = (
                            process_response.json()
                        )

                        st.success(
                            "🎉 Document processed "
                            "successfully!"
                        )

                        st.write(
                            f"**Chunks:** "
                            f"{process_data.get('total_chunks', 0)}"
                        )

            except requests.exceptions.RequestException as e:

                st.error(
                    f"Could not connect to backend: {e}"
                )


# =============================
# Retrieval Settings
# =============================

with st.sidebar:

    st.divider()

    st.header("⚙️ Retrieval Settings")

    department = st.selectbox(
        "Department",
        [
            "All",
            "HR",
            "Sales",
            "IT",
            "Finance",
            "Product"
        ]
    )

    document_type = st.selectbox(
        "Document Type",
        [
            "All",
            "Policy",
            "Product",
            "Company Profile",
            "FAQ",
            "Security"
        ]
    )

    top_k = st.slider(
        "Top K Results",
        min_value=1,
        max_value=10,
        value=5
    )

    retrieval_limit = st.slider(
        "Retrieval Limit",
        min_value=5,
        max_value=20,
        value=10
    )


# =============================
# Chat
# =============================

st.subheader("💬 Ask the Company AI")

query = st.text_input(
    "Your question",
    placeholder="e.g. Who is the founder of NexaAI?"
)


# =============================
# Ask AI
# =============================

if st.button(
    "🤖 Ask AI",
    use_container_width=True
):

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        params = {
            "query": query,
            "session_id": st.session_state.session_id,
            "retrieval_limit": retrieval_limit,
            "top_k": top_k,
            
        }

        # Only send actual filters
        if department != "All":
            params["department"] = department

        if document_type != "All":
            params["document_type"] = document_type

        try:

            with st.spinner(
                "🔎 Searching company knowledge..."
            ):

                response = requests.get(
                    f"{BACKEND_URL}/search/hybrid",
                    params=params,
                    timeout=120
                )

            # -----------------------------
            # Backend Error
            # -----------------------------

            if response.status_code != 200:

                st.error(
                    f"Backend error: "
                    f"{response.status_code}"
                )

            else:

                data = response.json()

                # -----------------------------
                # Guardrail Block
                # -----------------------------

                if data.get("status") == "blocked":

                    st.warning(
                        "🛡️ Request blocked by "
                        "enterprise security policy."
                    )

                    st.write(
                        data.get(
                            "message",
                            "Request blocked."
                        )
                    )

                else:

                    # -----------------------------
                    # Answer
                    # -----------------------------

                    st.subheader("💡 Answer")

                    st.write(
                        data.get(
                            "answer",
                            "No answer available."
                        )
                    )

                    # -----------------------------
                    # Confidence
                    # -----------------------------

                    confidence = data.get(
                        "confidence",
                        "unknown"
                    )

                    st.info(
                        f"🎯 Confidence: {confidence}"
                    )

                    # -----------------------------
                    # Sources
                    # -----------------------------

                    st.subheader("📚 Sources")

                    sources = data.get(
                        "sources",
                        []
                    )

                    if sources:

                        for source in sources:

                            st.write(
                                f"📄 "
                                f"**{source['document_name']}** "
                                f"— Chunk "
                                f"{source['chunk_number']}"
                            )

                    else:

                        st.write(
                            "No sources available."
                        )

                    # -----------------------------
                    # Debug Information
                    # -----------------------------

                    with st.expander(
                        "🔍 Retrieval Details"
                    ):

                        st.write(
                            "**Original Query:**"
                        )

                        st.code(query)

                        st.write(
                            "**Rewritten Query:**"
                        )

                        st.code(
                            data.get(
                                "rewritten_query",
                                query
                            )
                        )

                        st.write(
                            "**Retrieved Results:**"
                        )

                        st.json(
                            data.get(
                                "results",
                                []
                            )
                        )

        except requests.exceptions.RequestException as e:

            st.error(
                f"Could not connect to backend: {e}"
            )