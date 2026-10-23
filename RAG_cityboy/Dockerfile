# 가벼운 Python 3.11 slim 이미지 사용
FROM python:3.11-slim

# 컨테이너 내부 작업 디렉토리 설정
WORKDIR /app

# 파이썬 출력 버퍼링 비활성화 (실시간 로그 및 스트리밍 반응성 향상)
ENV PYTHONUNBUFFERED=1

# 시스템 필수 패키지 설치 후 캐시 정리
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 코드 복사
COPY . .

# Cloud Run 기본 포트(8080)로 Uvicorn 서버 실행
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
