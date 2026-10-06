```python
import os
import sqlite3

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from config import get_config
from rag import RAGEngine


app = Flask(__name__)

config = get_config()

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

engine = RAGEngine(config)


# =========================================================
# CHAT HISTORY DATABASE
# =========================================================

DATABASE = "chat_history.db"


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():

    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id)
                REFERENCES chats(id)
                ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()


init_db()


# =========================================================
# HOME
# =========================================================

@app.route("/")
def index():
    return render_template("index.html")


# =========================================================
# DOCUMENTS
# =========================================================

@app.route("/documents", methods=["GET"])
def documents():

    return jsonify(
        engine.list_documents()
    )


# =========================================================
# UPLOAD
# =========================================================

@app.route("/upload", methods=["POST"])
def upload():

    file = request.files.get("file")

    if not file or not file.filename:

        return jsonify({
            "error": "Please select a document."
        }), 400

    filename = secure_filename(
        file.filename
    )

    try:

        result = engine.index_uploaded_file(
            file,
            filename
        )

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# CHAT HISTORY LIST
# =========================================================

@app.route("/chats", methods=["GET"])
def get_chats():

    conn = get_db()

    chats = conn.execute("""
        SELECT
            id,
            title,
            created_at,
            updated_at
        FROM chats
        ORDER BY updated_at DESC, id DESC
    """).fetchall()

    conn.close()

    return jsonify([
        {
            "id": chat["id"],
            "title": chat["title"],
            "created_at": chat["created_at"],
            "updated_at": chat["updated_at"]
        }
        for chat in chats
    ])


# =========================================================
# LOAD ONE CHAT
# =========================================================

@app.route("/chats/<int:chat_id>", methods=["GET"])
def get_chat(chat_id):

    conn = get_db()

    chat = conn.execute("""
        SELECT id, title
        FROM chats
        WHERE id = ?
    """, (chat_id,)).fetchone()

    if not chat:

        conn.close()

        return jsonify({
            "error": "Chat not found."
        }), 404


    messages = conn.execute("""
        SELECT
            role,
            content,
            created_at
        FROM messages
        WHERE chat_id = ?
        ORDER BY id ASC
    """, (chat_id,)).fetchall()

    conn.close()


    return jsonify({

        "id": chat["id"],

        "title": chat["title"],

        "messages": [

            {
                "role": message["role"],
                "content": message["content"],
                "created_at": message["created_at"]
            }

            for message in messages

        ]

    })


# =========================================================
# CHAT
# =========================================================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(
        silent=True
    ) or {}


    question = (
        data.get("question") or ""
    ).strip()


    chat_id = data.get(
        "chat_id"
    )


    if not question:

        return jsonify({
            "error": "Please enter a question."
        }), 400


    try:

        conn = get_db()


        # -------------------------------------------------
        # CREATE CHAT WHEN FIRST MESSAGE IS SENT
        # -------------------------------------------------

        if not chat_id:

            title = question

            if len(title) > 45:

                title = (
                    title[:45].rstrip()
                    + "..."
                )


            cursor = conn.execute("""
                INSERT INTO chats (title)
                VALUES (?)
            """, (title,))


            chat_id = cursor.lastrowid


        else:

            existing_chat = conn.execute("""
                SELECT id
                FROM chats
                WHERE id = ?
            """, (chat_id,)).fetchone()


            if not existing_chat:

                conn.close()

                return jsonify({
                    "error": "Chat not found."
                }), 404


        # -------------------------------------------------
        # SAVE USER MESSAGE
        # -------------------------------------------------

        conn.execute("""
            INSERT INTO messages
                (chat_id, role, content)
            VALUES
                (?, ?, ?)
        """, (
            chat_id,
            "user",
            question
        ))


        # -------------------------------------------------
        # RAG ANSWER
        # -------------------------------------------------

        result = engine.answer(
            question
        )


        answer = result.get(
            "answer",
            ""
        )


        # -------------------------------------------------
        # SAVE AI MESSAGE
        # -------------------------------------------------

        conn.execute("""
            INSERT INTO messages
                (chat_id, role, content)
            VALUES
                (?, ?, ?)
        """, (
            chat_id,
            "assistant",
            answer
        ))


        # -------------------------------------------------
        # UPDATE CHAT TIME
        # -------------------------------------------------

        conn.execute("""
            UPDATE chats
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (chat_id,))


        conn.commit()
        conn.close()


        result["chat_id"] = chat_id

        return jsonify(result)


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# DELETE CHAT
# =========================================================

@app.route(
    "/chats/<int:chat_id>",
    methods=["DELETE"]
)
def delete_chat(chat_id):

    conn = get_db()


    chat = conn.execute("""
        SELECT id
        FROM chats
        WHERE id = ?
    """, (chat_id,)).fetchone()


    if not chat:

        conn.close()

        return jsonify({
            "error": "Chat not found."
        }), 404


    conn.execute("""
        DELETE FROM messages
        WHERE chat_id = ?
    """, (chat_id,))


    conn.execute("""
        DELETE FROM chats
        WHERE id = ?
    """, (chat_id,))


    conn.commit()
    conn.close()


    return jsonify({
        "message": "Chat deleted successfully."
    })


# =========================================================
# DELETE DOCUMENT
# =========================================================

@app.route(
    "/documents/<document_id>",
    methods=["DELETE"]
)
def delete_document(document_id):

    try:

        return jsonify(
            engine.delete_document(
                document_id
            )
        )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "5000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
```
