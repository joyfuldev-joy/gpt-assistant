import streamlit as st
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from bs4 import BeautifulSoup

# -------------------- SETTINGS --------------------
PRODUCTS = {
    "AI Gateway": "ai-gateway",
    "Vectorize": "vectorize",
    "Workers AI": "workers-ai"
}
SITEMAP_URL = "https://developers.cloudflare.com/sitemap-0.xml"

# -------------------- PROMPTS --------------------
answer_prompt = ChatPromptTemplate.from_template("""
Using ONLY the following context, answer the user's question. If you don't know, say you don't know.
Then, give a score between 0 and 5 based on how well the answer matches the question.

Context: {context}

Question: {question}
""")

choose_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Use the highest-scoring and most recent answers to generate a final answer. Cite all sources as-is.\n\nAnswers:\n{answers}"
    ),
    ("human", "{question}")
])

# -------------------- STREAMLIT SETUP --------------------
st.set_page_config(page_title="SiteGPT - Cloudflare", page_icon="🧠")
st.title("📘 SiteGPT for Cloudflare Docs")

with st.sidebar:
    st.markdown("## 🔑 OpenAI API Key")
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")

    st.markdown("## 📂 Select Product")
    product = st.selectbox("Choose a Cloudflare product", list(PRODUCTS.keys()))

    st.markdown("---")
    st.markdown("🔗 [View on GitHub](https://github.com/yourusername/cloudflare-sitegpt)")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key.")
    st.stop()

# -------------------- GPT MODEL --------------------
llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    openai_api_key=openai_api_key,
    model="gpt-3.5-turbo-1106"
)

# -------------------- HTML PARSER --------------------
def parse_page(soup: BeautifulSoup) -> str:
    for tag in ["header", "footer", "nav"]:
        t = soup.find(tag)
        if t: t.decompose()
    return soup.get_text().replace("\n", " ").replace("\xa0", " ").strip()

# -------------------- LOAD + FILTER --------------------
@st.cache_data(show_spinner="🔄 Loading documents...")
def load_product_docs(product_keyword: str):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=150)
    loader = SitemapLoader(SITEMAP_URL, parsing_function=parse_page)
    loader.requests_per_second = 2
    all_docs = loader.load()
    filtered_docs = [doc for doc in all_docs if product_keyword in doc.metadata["source"]]
    split_docs = splitter.split_documents(filtered_docs)
    db = FAISS.from_documents(split_docs, OpenAIEmbeddings(openai_api_key=openai_api_key))
    return db.as_retriever()

retriever = load_product_docs(PRODUCTS[product])

# -------------------- ANSWER CHAIN --------------------
def get_answers(inputs):
    question = inputs["question"]
    docs = inputs["docs"]
    chain = answer_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": chain.invoke({"context": doc.page_content, "question": question}).content,
                "source": doc.metadata.get("source", "unknown"),
                "date": doc.metadata.get("lastmod", "unknown")
            }
            for doc in docs
        ]
    }

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    condensed = "\n\n".join(
        f"{a['answer']}\nSource: {a['source']}\nDate: {a['date']}" for a in answers
    )
    chain = choose_prompt | llm
    return chain.invoke({"question": question, "answers": condensed})

# -------------------- MAIN --------------------
query = st.text_input("💬 Ask your question about Cloudflare docs")

if query:
    chain = (
        {
            "docs": retriever,
            "question": RunnablePassthrough()
        }
        | RunnableLambda(get_answers)
        | RunnableLambda(choose_answer)
    )

    with st.spinner("Thinking..."):
        result = chain.invoke(query)
        st.markdown(result.content.replace("$", "\$"))