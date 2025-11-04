# D:\FREEAI\app.py  (cleaned for production/WAI server)
import os
import logging
from flask import Flask, request, jsonify, send_from_directory
import torch
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast, pipeline

# -------------------------
# Config
# -------------------------
MODEL_DIR = r"D:\FREEAI\models\local_distilbert"
STATIC_DIR = os.path.join(os.path.dirname(__file__), "web")
DEVICE = 0 if torch.cuda.is_available() else -1

# -------------------------
# Logging (no prints)
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("local_qa")

logger.info(f"Starting app; model_dir={MODEL_DIR}; device={'cuda' if DEVICE==0 else 'cpu'}")

# -------------------------
# App + model load
# -------------------------
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

# Load model & tokenizer once at startup
logger.info("Loading tokenizer and model (this may take a moment)...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForQuestionAnswering.from_pretrained(MODEL_DIR)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=DEVICE)
logger.info("Model loaded and pipeline ready.")

# Serve index page
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

# Serve other static assets
@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)

# QA endpoint
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    context = data.get("context", "")
    question = data.get("question", "")
    if not context or not question:
        return jsonify({"error": "Provide both 'context' and 'question' in JSON body."}), 400
    try:
        result = qa_pipeline(question=question, context=context)
        return jsonify({
            "answer": result.get("answer", ""),
            "score": float(result.get("score", 0.0)),
            "start": int(result.get("start", -1)),
            "end": int(result.get("end", -1))
        })
    except Exception as e:
        logger.exception("Error answering question")
        return jsonify({"error": str(e)}), 500

# do not run app.run() here when using a WSGI server like waitress
if __name__ == "__main__":
    # fallback for manual runs (still ok for simple testing)
    app.run(host="127.0.0.1", port=5000, debug=False)
