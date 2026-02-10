"""
Pax Historia Semantic Search Microservice
Model: BAAI/bge-m3 (SOTA MTEB Performance)
Dimensions: 1024
Architecture: Dense Retrieval + In-Memory Index
Target: High-Fidelity Semantic Matching for Game Lore
Author: Kunal Raha
"""

import time
import logging
import sys
import pandas as pd
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util

# --- 1. PROFESSIONAL LOGGING SETUP (CRITICAL FOR GUNICORN) ---
# This forces logs to output immediately to the Docker console
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
# Format: [Time] [Level] Message
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

app = Flask(__name__)

# --- 2. GLOBAL VARIABLES ---
MODEL_NAME = 'BAAI/bge-m3'
DATA_FILE = 'data.csv'
model = None
embeddings = None
raw_data = []

# --- 3. INITIALIZATION (Loads on Startup) ---
def initialize_service():
    global model, embeddings, raw_data
    
    logger.info("🚀 Starting Search Service...")
    
    # Load the SOTA Model
    logger.info(f"🧠 Loading Model: {MODEL_NAME}...")
    try:
        # caching_folder='./model_data' ensures it stays in the Docker container
        model = SentenceTransformer(MODEL_NAME, cache_folder='./model_data')
        logger.info("✅ Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Load Data CSV
    logger.info(f"📂 Loading Data from {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE)
        # Convert to list of dicts for easy JSON response
        raw_data = df.to_dict(orient='records')
        
        # Create a list of text to encode (Title + Content combined gives best results)
        text_corpus = [f"{row['title']}: {row['content']}" for row in raw_data]
        
        logger.info(f"⚡ Encoding {len(text_corpus)} documents (this might take a moment)...")
        embeddings = model.encode(text_corpus, convert_to_tensor=True)
        logger.info(f"✅ Indexed {len(embeddings)} documents.")
        
    except Exception as e:
        logger.error(f"❌ Failed to process data: {e}")
        sys.exit(1)

# Run initialization explicitly
initialize_service()

# --- 4. THE SEARCH ENDPOINT ---
@app.route('/search', methods=['GET'])
def search():
    """
    Endpoint: /search?q=query_string
    """
    # Start the Timer
    start_time = time.time()
    
    query = request.args.get('q')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    if model is None or embeddings is None:
        return jsonify({"error": "Service not initialized"}), 500

    try:
        # 1. Encode the User Query
        query_vec = model.encode(query, convert_to_tensor=True)
        
        # 2. Semantic Search (Cosine Similarity)
        # util.cos_sim returns a tensor of scores
        scores = util.cos_sim(query_vec, embeddings)[0]

        # 3. Format Results
        # Combine score with data, sort descending
        results = []
        for idx, score in enumerate(scores):
            results.append({
                "id": raw_data[idx].get('id'),
                "title": raw_data[idx].get('title'),
                "content": raw_data[idx].get('content'),
                "score": float(score)
            })
        
        # Sort by score (highest first) and take Top 5
        results = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
        
        # End Timer & Calculate Latency
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # --- THE MONEY LOG (Eli sees this) ---
        logger.info(f"🔍 Query: '{query}' | ✅ Found {len(results)} matches | ⏱️ Speed: {latency_ms:.2f}ms")

        return jsonify(results)

    except Exception as e:
        logger.error(f"⚠️ Search Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": MODEL_NAME}), 200

if __name__ == '__main__':
    # This block is for local testing (python server.py)
    # Gunicorn ignores this block and just imports 'app'
    app.run(host='0.0.0.0', port=5000, debug=False)