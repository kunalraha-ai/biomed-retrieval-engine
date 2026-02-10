# 1. Base Image: Python 3.9 (Slim version is smaller/faster)
FROM python:3.9-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements first (to cache dependencies)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the code (including server.py and data.csv)
COPY . .

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3', cache_folder='./model_data')"

# 6. Open Port 5000
EXPOSE 5000

# 7. Start the server
CMD ["gunicorn", "--workers=1", "--threads=8", "--bind=0.0.0.0:5000", "--timeout=120", "server:app"]