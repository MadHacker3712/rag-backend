FROM python:3.11-slim

WORKDIR /app

# Install torch CPU-only first (prevents pip from pulling 2GB CUDA packages)
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy data and app code
COPY data/ ./data/
COPY build_index.py .
COPY app.py .

# Pre-build the ChromaDB index at image build time
# This means the container starts fast — no index build on first request
RUN python build_index.py

EXPOSE 8000

ENV CHROMA_DIR=./chroma_db
ENV EMBED_MODEL=all-MiniLM-L6-v2
ENV TOP_K=3

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
