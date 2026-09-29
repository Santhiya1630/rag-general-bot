# RAG Bot - Document + General AI

This project combines:
- Flask
- Gemini API
- Qdrant Cloud
- FastEmbed with all-MiniLM-L6-v2 (384 dimensions)
- PyMuPDF PDF parsing
- LlamaIndex is not used
- LangChain is not used

Behavior:
1. If a question matches uploaded PDF content, the retrieved PDF context is used.
2. If the question is unrelated to uploaded documents, Gemini can answer it using general knowledge.
3. The answer does not claim that general knowledge came from an uploaded document.

For deployment, put secrets in the hosting provider's environment variables. Do not upload `.env`.
