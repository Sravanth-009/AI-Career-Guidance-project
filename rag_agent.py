import os
from operator import itemgetter
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Change the model here if needed (run list_models.py to see what your key can use)
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

SYSTEM_PROMPT = """You are an AI Career Guidance Agent for students.

Use ONLY the career knowledge below to give advice. If the answer is not
in the context, say you don't have that information and give general guidance.

Student profile:
{profile}

Career knowledge:
{context}

Conversation so far:
{history}

Rules:
- Recommend careers that match the student's skills and interests.
- Give a clear step-by-step roadmap when asked.
- Be encouraging, practical, and concise.
- Do not invent salaries, companies, or certifications not in the context.
"""


def build_chain():
    # 1. Load the knowledge base
    docs = TextLoader("career_info.txt", encoding="utf-8").load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    # 3. Embed and store in FAISS (local embeddings, no API key needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 4. Groq LLM + prompt
    llm = ChatGroq(model=MODEL_NAME, temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # 5. LCEL chain
    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "profile": itemgetter("profile"),
            "history": itemgetter("history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def format_history(messages, max_turns=6):
    recent = messages[-max_turns * 2:]
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)