import streamlit as st

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# --------------------------------------------------
# Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Powered RAG Application",
    page_icon="☀️",
    layout="centered"
)

st.title("☀️ AI Based RAG Application")


# --------------------------------------------------
# Hugging Face LLM
# --------------------------------------------------

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    huggingfacehub_api_token = "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",
    max_new_tokens=2500
)

model = ChatHuggingFace(llm=llm)


# --------------------------------------------------
# PDF Upload
# --------------------------------------------------

pdf_list = st.sidebar.file_uploader(
    "📄 Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# --------------------------------------------------
# Website URL
# --------------------------------------------------

url = st.sidebar.text_input(
    "🌐 Enter website URL",
    placeholder="https://example.com"
)


# --------------------------------------------------
# Build Knowledge Base
# --------------------------------------------------

if st.sidebar.button("🔨 Build Knowledge Base", type="primary"):

    docs = []

    # -----------------------------
    # Load PDFs
    # -----------------------------

    if pdf_list:

        for pdf in pdf_list:

            # Save uploaded PDF temporarily
            with open(pdf.name, "wb") as f:
                f.write(pdf.getbuffer())

            pdf_loader = PyPDFLoader(pdf.name)

            docs.extend(pdf_loader.load())


    # -----------------------------
    # Load Website
    # -----------------------------

    if url:

        try:

            web_loader = WebBaseLoader(url)

            docs.extend(web_loader.load())

        except Exception as e:

            st.error(f"Could not load website: {e}")


    # -----------------------------
    # Check documents
    # -----------------------------

    if not docs:

        st.error(
            "❌ No documents found. "
            "Please upload a PDF or enter a valid website URL."
        )

        st.stop()


    st.success(f"📚 Loaded {len(docs)} documents")


    # --------------------------------------------------
    # Text Splitting
    # --------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(docs)

    st.success(f"✂️ Created {len(chunks)} chunks")


    # --------------------------------------------------
    # Embeddings
    # --------------------------------------------------

    # embedding = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2"
    # )
    from langchain_huggingface import HuggingFaceEmbeddings

    embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
    )


    # --------------------------------------------------
    # FAISS Vector Store
    # --------------------------------------------------

    vector_store = FAISS.from_documents(
        chunks,
        embedding
    )

    st.session_state.vector_store = vector_store

    st.success("✅ Knowledge base created successfully!")


# --------------------------------------------------
# Question
# --------------------------------------------------

question = st.text_input(
    "💬 Enter your question",
    placeholder="Example: What is LangChain?"
)


# --------------------------------------------------
# Answer
# --------------------------------------------------

if st.button("🤖 Answer"):

    if not question.strip():

        st.warning("Please enter a question.")
        st.stop()


    # Check whether vector store exists

    if "vector_store" not in st.session_state:

        st.warning(
            "Please build the knowledge base first."
        )

        st.stop()


    vector_store = st.session_state.vector_store


    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    retriever_docs = retriever.invoke(question)


    # --------------------------------------------------
    # Context
    # --------------------------------------------------

    context_text = "\n\n".join(
        [doc.page_content for doc in retriever_docs]
    )


    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------

    prompt = PromptTemplate(
        template="""
You are a helpful RAG assistant.

Answer the user's question using ONLY the provided context.

Question:
{question}

Context:
{context}

Instructions:
- Give a clear answer.
- Use bullet points when appropriate.
- Use paragraphs when appropriate.
- Use tables when they improve understanding.
- If the answer is not available in the context, politely say:
  "I could not find this information in the uploaded documents or website."
""",
        input_variables=["question", "context"]
    )


    final_prompt = prompt.format(
        question=question,
        context=context_text
    )


    # --------------------------------------------------
    # Generate Answer
    # --------------------------------------------------

    with st.spinner("🤖 Searching your knowledge base..."):

        try:

            response = model.invoke(final_prompt)

            st.subheader("📌 Final Answer")

            st.write(response.content)


        except Exception as e:

            st.error(
                f"Something went wrong: {e}"
            )
