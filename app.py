# 1. Imports
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
import os

# 2. Streamlit 기본 설정
st.set_page_config(page_title="RAG Chatbot", page_icon="📚")
st.title("📚 Custom RAG Chatbot")

# 📎 GitHub 링크 추가
with st.sidebar:
    st.markdown("🔗 [GitHub Repository](https://github.com/joyfuldev-joy/fullstack-gpt-challenge5)")
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("Upload your document", type=["pdf", "txt", "docx"])
    
    
# 3. 사용자 입력 체크
if not api_key:
    st.warning("Please enter your OpenAI API key to start.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key


# 4. 파일 임베딩
@st.cache_data(show_spinner="Embedding document...")
def embed_file(file):
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.read())

    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=600, chunk_overlap=100)
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    embedding = OpenAIEmbeddings()
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embedding, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    return vectorstore.as_retriever()
  
  
  # 5. 프롬프트 설정
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer only using the context below. If unsure, say 'I don't know'.\n\nContext: {context}"),
    ("human", "{question}")
])

# 6. 대화 시작
if uploaded_file:
    retriever = embed_file(uploaded_file)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["message"])

    query = st.chat_input("Ask your question about the document...")
    if query:
        st.session_state.messages.append({"role": "user", "message": query})
        with st.chat_message("user"):
            st.markdown(query)

        chain = {
            "context": retriever | RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough()
        } | prompt | ChatOpenAI(temperature=0.1, api_key=api_key)

        with st.chat_message("assistant"):
            response = chain.invoke(query)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "message": response})