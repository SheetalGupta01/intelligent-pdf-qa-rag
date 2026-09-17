import streamlit as st

from rag_pipeline import (
    extract_text_from_pdf,
    split_text,
    create_embeddings,
    create_vector_database,
    retrieve_documents,
    generate_answer
)

st.set_page_config(
    page_title="Intelligent PDF Q&A",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Intelligent PDF Q&A")

st.write(
    "Upload a PDF and ask questions using "
    "Retrieval-Augmented Generation (RAG)."
)

if "vector_db" not in st.session_state:

    st.session_state.vector_db = None


if "processed" not in st.session_state:

    st.session_state.processed = False


if "chunks" not in st.session_state:

    st.session_state.chunks = []

uploaded_file = st.file_uploader(
    "📤 Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )


    if st.button("🔍 Process PDF"):

        with st.spinner(
            "Processing PDF and creating embeddings..."
        ):


            with open("temp.pdf", "wb") as f:

                f.write(
                    uploaded_file.getbuffer()
                )

            text = extract_text_from_pdf(
                "temp.pdf"
            )

            if not text.strip():

                st.error(
                    "Could not extract text from this PDF."
                )

                st.stop()

            chunks = split_text(text)
            embeddings = create_embeddings()

            vector_db = create_vector_database(
                chunks,
                embeddings
            )

            st.session_state.vector_db = vector_db

            st.session_state.processed = True

            st.session_state.chunks = chunks


        st.success(
            "✅ PDF processed successfully!"
        )


if st.session_state.processed:

    st.divider()

    st.subheader("📊 Document Information")

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Total Chunks",
            len(st.session_state.chunks)
        )


    with col2:

        st.metric(
            "Retrieved Chunks",
            "Top 3"
        )

if st.session_state.processed:

    st.divider()

    st.subheader("💬 Ask Questions")

    question = st.text_input(
        "Ask something about the PDF:",
        placeholder="Example: What is the main objective of this paper?"
    )


    if st.button("🤖 Get Answer"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

            st.stop()

        with st.spinner(
            "🔍 Searching the document..."
        ):

            documents = retrieve_documents(
                st.session_state.vector_db,
                question
            )

        with st.spinner(
            "🤖 Generating answer..."
        ):

            answer = generate_answer(
                documents,
                question
            )

        st.subheader("💡 Answer")

        st.write(answer)

        st.subheader("📚 Retrieved Sources")

        for i, document in enumerate(documents):

            with st.expander(
                f"Source {i + 1}"
            ):

                st.write(
                    document.page_content
                )