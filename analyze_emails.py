import os
from typing import List, Dict
from pathlib import Path
import email.parser
from pydantic import BaseModel
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
# Configuration
EMAIL_DIR = '/c/Users/steve/projects/email-case-rag/emails'
GEN_MODEL_NAME = 'ollama/qwen2.5:14b'  # Ollama model for generation (local)
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'  # Local embedding model
CHROMA_DIR = '/c/Users/steve/projects/email-case-rag/chromadb'
class EmailMeta(BaseModel):
    email_id: str
    subject: str
    from_: str
    to: List[str]
    date: str
    content: str
def parse_email(file_path) -> Dict:
    with open(file_path, 'r') as f:
        e = email.parser.BytesParser().parsebytes(f.read())
        return {
            "email_id": os.path.basename(file_path),
            "subject": e['subject'],
            "from_": e.get('from'),
            "to": e.get_all('to', []),
            "date": e['date'],
            "content": e.get_payload()
        }
def load_emails() -> List[Dict]:
    emails = []
    for file in Path(EMAIL_DIR).glob('*.eml'):
        meta = parse_email(file)
        emails.append(meta)
    return emails
def chunk_text(emails: List[Dict], chunk_size=512):
    chunks = []
    for email in emails:
        content_chunks = [email['content'][i:i+chunk_size] for i in range(0, len(email['content']), chunk_size)]
        for i, chunk in enumerate(content_chunks):
            chunks.append({
                "id": f"{email['email_id']}_{i}",
                "text": chunk,
                "subject": email['subject'],
                "from_": email['from_'],
                "to": ', '.join(email['to']),
                "date": email['date']
            })
    return chunks
def create_embeddings(chunks, model_name=EMBEDDING_MODEL_NAME):
    embeddings = SentenceTransformer(model_name)
    for chunk in chunks:
        text = chunk['text']
        embedding = embeddings.encode(text).tolist()
        chunk['embedding'] = embedding
    return chunks
def init_chroma():
    chroma_client = chromadb.Client(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(name="emails", metadata={"hnsw:space": "cosine"})
    return collection, chroma_client
def index_emails(chunks):
    collection, _ = init_chroma()
    collection.add_embeddings([chunk['embedding'] for chunk in chunks], [chunk['text'] for chunk in chunks])
def query_emails(prompt, collection):
    results = collection.query(
        query_embeddings=[],
        n_results=5
    )
    
    context = "\n".join([result.text for result in results])
    
    question_answering = pipeline("question-answering", model=GEN_MODEL_NAME)
    return question_answering(question=prompt, context=context)
def main():
    # Load and parse emails
    emails = load_emails()
    
    # Chunk texts into pieces
    text_chunks = chunk_text(emails)
    
    # Create embeddings for chunks
    with_embeddings = create_embeddings(text_chunks)
    
    # Initialize ChromaDB client and collection
    init_chroma()
    
    # Index the emails in ChromaDB
    index_emails(with_embeddings)
if __name__ == "__main__":
    main()
    chroma_client = chromadb.Client(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(name="emails", metadata={"hnsw:space": "cosine"})
    
    while True:
        user_input = input("Query (type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        
        prompt = query_emails(user_input, collection)
        print(prompt)