import streamlit as st
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_community.llms import ollama 
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# Optional: Gemini integration
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load .env for API key
load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key="AIzaSyCgFPEnXxMgWHIirm9FuQK8c6QEm5FRjGU")

# Streamlit page setup
st.set_page_config(page_title="🎓 Harsha's UPSC Learning Assistant", layout="centered")
st.markdown("""
<h1 style='text-align: center; color: #2B547E;'>🎓 Harsha's UPSC Learning Assistant</h1>
<p style='text-align: center; color: gray;'>Your friendly AI teacher that explains UPSC concepts simply!</p>
<hr>
""", unsafe_allow_html=True)

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are Harsha's UPSC Learning Assistant — a kind, patient teacher who explains UPSC topics 
in the simplest way possible, as if teaching a 10-year-old child.

Use the following context to answer clearly and accurately.

Context:
{context}

Question: {question}

Your reply should:
1. Explain the topic like a story or simple concept (avoid big words unless explained).
2. Use short sentences and examples.
3. End with one small UPSC study tip or strategy related to this topic.
4. Always sound friendly and encouraging.
"""
)

# URLs for content
urls = [
    "https://en.wikipedia.org/wiki/Union_Public_Service_Commission",
    "https://pib.gov.in/",
    "https://www.insightsonindia.com/",
    "https://iasgoogle.com",
    "https://visionias.in/",
    "https://vajiramandravi.com/",
    "https://forumias.com/"
]

# Load and split documents
loader = WebBaseLoader(urls)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Create embeddings and FAISS vector store
#from langchain_community.embeddings import OllamaEmbeddings  # You can switch to any embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(chunks, embeddings)

# Create retriever
retriever = vector_store.as_retriever()

llm = ChatGroq(model="llama-3.1-8b-instant", api_key="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")
from langchain_core.output_parsers import StrOutputParser
llm = llm | StrOutputParser()

# Create a simple RAG QA chain
#question_answer_chain = create_stuff_documents_chain(llm=llm,prompt=prompt_template)
#chain = create_retrieval_chain(retriever, question_answer_chain)

# Function to get AI response using Gemini
def get_gemini_response(prompt_text):
    model = genai.GenerativeModel("models/gemini-2.5-pro")    

    response = model.generate_content(prompt_text)
    return response.text

# Function to fetch images
def fetch_relevant_images(topic):
    search = DDGS()
    results = search.images(topic, max_results=2)
    return [r["image"] for r in results]

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.chat_input("Ask your UPSC question here...")
qa_chain = (
        # Step 1: Prepare inputs dictionary {"context": ..., "question": ...}
        {
            # The context key is created by piping the question (implicitly) to the retriever,
            # and then piping the resulting documents through the formatter.
            "context": retriever, 
            
            # The question key is created by passing the original input straight through.
            "question": RunnablePassthrough() 
        }
        # Step 2: Format the Prompt Template (fills {context} and {question})
        | prompt_template
        
        # Step 3: Call the Language Model (Groq)
        | llm
        
        # Step 4: Parse the output into a clean string
        | StrOutputParser()
    )
if user_input:
    st.session_state.chat_history.append(("You", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get AI response via the LangChain RAG pipeline
    with st.spinner("Harsha's Assistant is preparing the simple explanation..."):
        try:
            # The chain handles ALL steps (retrieval, prompt, LLM call) correctly.
            response = qa_chain.invoke(user_input)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            response = "Sorry! I ran into an issue while generating the response. Please check your setup."
            
    st.session_state.chat_history.append(("Harsha's Assistant", response))
    with st.chat_message("assistant"):
        st.markdown(response)

