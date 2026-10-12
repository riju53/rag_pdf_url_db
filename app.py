import streamlit as st



# Streamlit configuration

st.set_page_config(
    page_title="AI Powered RAG Application",
    page_icon="☀️",
    layout="centered"
)

st.title("☀️ AI Based RAG Application")


# Hugging Face LLM
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    huggingfacehub_api_token = "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",
    max_new_tokens=2500
)

model = ChatHuggingFace(llm=llm)


# PDF Upload
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

pdf_list = st.sidebar.file_uploader(
    "📄 Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# Website URL

url = st.sidebar.text_input(
    "🌐 Enter website URL",
    placeholder="https://example.com"
)


# Build Knowledge Base

if st.sidebar.button("🔨 Build Knowledge Base", type="primary"):

    docs = []

    # Load PDFs

    if pdf_list:

        for pdf in pdf_list:

            # Save uploaded PDF temporarily
            with open(pdf.name, "wb") as f:
                f.write(pdf.getbuffer())

            pdf_loader = PyPDFLoader(pdf.name)

            docs.extend(pdf_loader.load())


    # Load Website

    if url:

        try:

            web_loader = WebBaseLoader(url)

            docs.extend(web_loader.load())

        except Exception as e:

            st.error(f"Could not load website: {e}")


    # Check documents

    if not docs:

        st.error(
            "❌ No documents found. "
            "Please upload a PDF or enter a valid website URL."
        )

        st.stop()


    st.sidebar.success(f"📚 Loaded {len(docs)} documents")


    # Text Splitting
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(docs)

    st.sidebar.success(f"✂️ Created {len(chunks)} chunks")


    # Embeddings
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings

    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # FAISS Vector Store
    from langchain_community.vectorstores import FAISS

    vector_store = FAISS.from_documents(
        chunks,
        embedding
    )

    st.session_state.vector_store = vector_store

    st.success("✅ Knowledge base created successfully!")


# Question

question = st.text_input(
    "💬 Enter your question",
    placeholder="Example: What is LangChain?"
)


# Answer

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


    # Retrieval

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    retriever_docs = retriever.invoke(question)


    # Context

    context_text = "\n\n".join(
        [doc.page_content for doc in retriever_docs]
    )


    # Prompt
    from langchain_core.prompts import PromptTemplate

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


    # Generate Answer

    with st.spinner("🤖 Searching your knowledge base..."):

        try:

            response = model.invoke(final_prompt)

            st.subheader("📌 Final Answer")

            st.write(response.content)


        except Exception as e:

            st.error(
                f"Something went wrong: {e}"
            )
