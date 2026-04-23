import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

load_dotenv()

st.set_page_config(page_title="DMV Assistant", page_icon="🚗", layout="centered")
st.title("🤖 DMV RAG App")
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stTextInput > div > div > input {
            border-radius: 12px;
            border: 2px solid #4A90D9;
            padding: 10px;
            font-size: 16px;
        }
        .answer-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 25px;
            border-radius: 15px;
            margin: 15px 0;
            font-size: 16px;
            line-height: 1.6;
        }
        .source-box {
            background: white;
            border-left: 4px solid #4A90D9;
            padding: 12px 16px;
            border-radius: 8px;
            margin: 8px 0;
            font-size: 14px;
            color: #333;
        }
        .source-label {
            font-weight: bold;
            color: #4A90D9;
            margin-bottom: 5px;
        }
        .header-sub {
            color: #888;
            font-size: 15px;
            margin-top: -15px;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🚗 DMV Assistant")
st.markdown('<p class="header-sub">Ask anything about the California Driver\'s Handbook</p>', unsafe_allow_html=True)
st.divider()

@st.cache_resource
def load_data():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vectorstore

vectorstore = load_data()

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Answer the user's question using ONLY the context provided below.
Use the context below to answer the question. You may paraphrase and summarize.
If the context contains relevant information even partially, use it to give a helpful answer.
Only if the context is completely unrelated to the question, respond with exactly:
I'm sorry, I am only authorized to talk about the provided document.
Do not use outside knowledge. Do not guess.
Context:
{context}

Question: {question}

Answer:"""
)

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_prompt
    | llm
    | StrOutputParser()
)

question = st.text_input("Ask me about the document:")

if question:
    with st.spinner("Thinking..."):
        response = chain.invoke(question)
        st.markdown("### Answer:")
        st.write(response)

    with st.expander("📄 View Sources"):
        docs = retriever.invoke(question)[:5]
        for i, doc in enumerate(docs):
            st.info(f"**Source {i+1}** (Page {doc.metadata.get('page', 'N/A')}):\n\n{doc.page_content[:300]}...")