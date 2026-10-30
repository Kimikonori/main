import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import uvicorn

app = FastAPI()

# 임베딩 및 LLM 모델 설정 (환경변수 GOOGLE_API_KEY 자동 연동)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)

# 시스템 역할 정의: 서울시 특화 도시 계획 설계자
template = """당신은 도시 계획 설계자입니다. 특히 서울시에 특화된 설계자로 사용자의 질문에 답변하세요.

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

vectorstore = None
retriever = None

@app.get("/")
def read_root():
    return {"status": "running", "message": "Seoul Urban Planner RAG Server is up."}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore, retriever
    file_path = f"./{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(pages)
    
    # 벡터 DB: Chroma DB 사용
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    
    # 다중 질문 검색기(Multi-Query Retriever) 설정
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
    
    return {"status": "success", "message": "PDF processed and Chroma DB built successfully."}

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(req: QueryRequest):
    global retriever
    if not retriever:
        return {
            "answer": "PDF가 아직 업로드되지 않았거나 Chroma DB가 구축되지 않았습니다. 먼저 PDF를 업로드해주세요.",
            "rag_config": {}
        }
    
    # RunnablePassthrough를 활용한 질의 확장 및 RAG 체인 구성
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    answer = rag_chain.invoke(req.question)
    return {
        "answer": answer,
        "rag_config": {
            "vector_db": "Chroma DB",
            "embedding_model": "Google Generative AI Embeddings (embedding-001)",
            "llm_model": "Gemini 1.5 Pro",
            "retriever": "Multi-Query Retriever",
            "query_expansion": "RunnablePassthrough applied"
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
