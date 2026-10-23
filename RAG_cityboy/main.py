import os
import tempfile
import threading
from queue import Queue
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from supabase.client import Client, create_client

# LangChain 관련 모듈 임포트
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks.base import BaseCallbackHandler

# =====================================================================
# 1. 환경 변수 로드 및 외부 클라이언트 초기화
# =====================================================================
# Google AI API 키는 GOOGLE_API_KEY 환경 변수에서 자동으로 참조하거나 명시적으로 로드합니다.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Supabase 클라이언트 생성
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# OpenAI 임베딩 모델 초기화 (1536차원)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# Supabase 벡터 스토어 객체 연결
vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name="pdf_policy_documents",
    query_name="match_policy_documents"
)

# 검색기(Retriever) 생성: 상위 4개의 유사 청크를 반환하도록 설정
retriever = vector_store.as_retriever(search_kwargs={"k": 4})


# =====================================================================
# 2. 실시간 스트리밍 핸들러 (StreamHandler) 정의
# =====================================================================
class StreamHandler(BaseCallbackHandler):
    """
    LLM에서 새로운 토큰이 생성될 때마다 호출되어
    실시간으로 큐(Queue)에 토큰을 담아 클라이언트로 스트리밍하는 핸들러입니다.
    """
    def __init__(self, token_queue: Queue):
        self.token_queue = token_queue

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # 생성된 토큰을 즉시 큐에 삽입
        self.token_queue.put(token)

    def on_llm_end(self, response, **kwargs) -> None:
        # 생성이 완료되면 종료 신호(None)를 큐에 삽입
        self.token_queue.put(None)

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        # 에러 발생 시 에러 메시지를 전달하고 종료 신호 삽입
        self.token_queue.put(f"\n[스트리밍 오류]: {str(error)}")
        self.token_queue.put(None)


# =====================================================================
# 3. 질의 확장 체인 (Query Augmentation Chain) 구성
# =====================================================================
# 3-1. 질의 확장용 LLM 객체 생성 (gemini-3.5-flash-lite 모델 사용)
augmentation_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1
)

# 3-2. 의도 파악 및 대명사 명확화를 위한 시스템 메시지 및 프롬프트 템플릿 설계
augmentation_system_msg = (
    "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 "
    "한문장의 명료한 질문으로 변환하라. "
    "대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라"
)

augmentation_prompt = ChatPromptTemplate.from_messages([
    ("system", augmentation_system_msg),
    # 과거 대화 내역(Chat History)이 동적으로 주입되는 공간 마련
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

# 3-3. 프롬프트 -> LLM -> 문자열 파서(StrOutputParser)를 파이프라인으로 연결
query_augmentation_chain = (
    augmentation_prompt 
    | augmentation_llm 
    | StrOutputParser()
)


# =====================================================================
# 4. RAG 답변 생성 프롬프트 설계
# =====================================================================
# 도시 정책 전문가 역할을 부여하는 시스템 메시지 설정
rag_system_msg = (
    "너는 문서에 기반해 답변하는 도시 정책 전문가야.\n"
    "아래 제공된 [참고 문서]의 내용만을 근거로 삼아 사용자의 질문에 정확하고 체계적으로 답변해.\n"
    "만약 [참고 문서]에서 해답을 찾을 수 없다면 임의로 추측하지 말고 문서에 관련 내용이 없다고 안내해.\n\n"
    "[참고 문서]\n{context}"
)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_system_msg),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])


# =====================================================================
# 5. FastAPI 앱 및 API 엔드포인트 정의
# =====================================================================
app = FastAPI(title="Cloud Run Urban Policy RAG Server")


class ChatMessage(BaseModel):
    role: str      # "user" 또는 "ai"
    content: str   # 메시지 본문


class ChatRequest(BaseModel):
    query: str                        # 현재 사용자의 질문
    chat: List[ChatMessage] = []      # 이전까지의 대화 기록(chat 객체)


@app.post("/upload-pdf")
async def upload_and_index_pdf(file: UploadFile = File(...)):
    """
    업로드된 PDF 파일을 텍스트로 변환하고 청크 단위로 분할한 뒤,
    OpenAI 임베딩을 거쳐 Supabase 벡터 DB에 저장합니다.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    # 임시 파일로 저장하여 PyPDFLoader로 텍스트 추출
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(await file.read())
        tmp_path = tmp_file.name

    try:
        # 1. PDF 파일 로드 및 텍스트 추출
        loader = PyPDFLoader(tmp_path)
        raw_documents = loader.load()

        # 2. 문서를 청크(Chunk) 단위로 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # 각 청크의 최대 글자 수
            chunk_overlap=150     # 문맥 유지를 위한 청크 간 중첩 글자 수
        )
        chunks = text_splitter.split_documents(raw_documents)

        # 원본 파일명을 메타데이터에 기록
        for chunk in chunks:
            chunk.metadata["source_file"] = file.filename

        # 3. 임베딩 생성 및 Supabase 벡터 DB에 저장
        SupabaseVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=supabase_client,
            table_name="pdf_policy_documents",
            query_name="match_policy_documents"
        )

        return {
            "status": "success",
            "filename": file.filename,
            "total_chunks": len(chunks)
        }
    finally:
        # 처리 완료 후 임시 파일 삭제
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/chat")
async def chat_with_policy_rag(request: ChatRequest):
    """
    1) 이전 대화 기록과 현재 질문을 결합해 질의를 확장하고,
    2) 확장된 질의로 Supabase 리트리버에서 문서를 검색한 뒤,
    3) StreamHandler를 통해 답변을 실시간 스트리밍으로 반환합니다.
    """
    # 1. 전달받은 대화 기록(chat 객체)을 LangChain 메시지 객체 리스트로 변환
    formatted_history = []
    for msg in request.chat:
        if msg.role in ("user", "human"):
            formatted_history.append(HumanMessage(content=msg.content))
        else:
            formatted_history.append(AIMessage(content=msg.content))

    # 2. Query Augmentation Chain 실행 (현재 질문 + 대화 기록을 딕셔너리로 주입)
    augmented_query = query_augmentation_chain.invoke({
        "query": request.query,
        "chat_history": formatted_history
    })

    # 3. 증강된 질의(augmented_query)를 Supabase 리트리버에 전달하여 관련 청크 검색
    retrieved_docs = retriever.invoke(augmented_query)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    # 4. 실시간 스트리밍을 위한 큐 및 StreamHandler 설정
    token_queue = Queue()
    stream_handler = StreamHandler(token_queue)

    # 스트리밍이 활성화된 gemini-3.5-flash-lite 모델 생성
    streaming_llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        streaming=True,
        callbacks=[stream_handler],
        temperature=0.2
    )

    # 답변 생성 체인 구성
    rag_answer_chain = rag_prompt | streaming_llm

    # 백그라운드 스레드에서 LLM 체인 실행 (생성되는 토큰은 실시간으로 큐에 쌓임)
    def generate_in_background():
        rag_answer_chain.invoke({
            "context": context_text,
            "chat_history": formatted_history,
            "question": augmented_query
        })

    threading.Thread(target=generate_in_background, daemon=True).start()

    # 큐에서 토큰을 하나씩 꺼내 클라이언트로 즉시 전송하는 제너레이터
    def token_stream_generator():
        while True:
            token = token_queue.get()
            if token is None:
                break
            yield token

    return StreamingResponse(token_stream_generator(), media_type="text/plain; charset=utf-8")
