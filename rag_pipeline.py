import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI


def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    return chunks


def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


def create_vector_database(chunks, embeddings):
    vector_db = FAISS.from_texts(
        chunks,
        embeddings
    )

    return vector_db


def retrieve_documents(vector_db, question):
    documents = vector_db.similarity_search(
        question,
        k=5
    )

    return documents


def generate_answer(documents, question):

    context = "\n\n".join(
        f"CONTEXT {i + 1}:\n{document.page_content}"
        for i, document in enumerate(documents)
    )

    prompt = f"""
You are a question-answering assistant for a research paper.

Answer the user's question based ONLY on the CONTEXT provided below.

Instructions:

1. Carefully read all context sections.
2. Find information that directly answers the question.
3. Combine information from multiple context sections if necessary.
4. Give a clear and concise answer.
5. If the exact answer is not available, say:
"I could not find the answer in the retrieved sections of the document."
6. Do not invent facts.
7. When numbers, percentages, model names, datasets, or results
appear in the context, preserve them accurately.

========================
RETRIEVED CONTEXT
========================

{context}

========================
QUESTION
========================

{question}

========================
ANSWER
========================
"""

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    response = llm.invoke(prompt)

    return response.content