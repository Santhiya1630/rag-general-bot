import io
import re
import fitz

def _clean(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _chunks(text, size, overlap):
    text = _clean(text)
    if not text:
        return []
    chunks = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        end = min(len(text), start + size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start += step
    return chunks

def process_document(data, filename, chunk_size, chunk_overlap):
    suffix = filename.lower()
    if not suffix.endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    doc = fitz.open(stream=data, filetype="pdf")
    chunks = []
    for page_index, page in enumerate(doc):
        text = _clean(page.get_text("text"))
        for piece in _chunks(text, chunk_size, chunk_overlap):
            chunks.append({
                "text": piece,
                "source": filename,
                "page": page_index + 1,
            })
    doc.close()

    return {
        "file_type": "pdf",
        "chunks": chunks,
    }
