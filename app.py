import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from config import get_config
from rag import RAGEngine

app = Flask(__name__)
config = get_config()
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

engine = RAGEngine(config)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/documents", methods=["GET"])
def documents():
    return jsonify(engine.list_documents())

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "Please select a document."}), 400
    filename = secure_filename(file.filename)
    try:
        result = engine.index_uploaded_file(file, filename)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Please enter a question."}), 400
    try:
        return jsonify(engine.answer(question))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/documents/<document_id>", methods=["DELETE"])
def delete_document(document_id):
    try:
        return jsonify(engine.delete_document(document_id))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
