import os
from typing import List, Dict, Optional
from pathlib import Path
import email.parser
from pydantic import BaseModel
import chromadb
import requests

# Configuration
EMAIL_DIR = 'C:/Users/steve/projects/email-case-rag/emails'
GEN_MODEL_NAME = 'qwen2.5:14b'  # Ollama model for generation (local)
EMBEDDING_MODEL_NAME = 'mxbai-embed-large'  # Ollama embedding model (local)
CHROMA_DIR = 'C:/Users/steve/projects/email-case-rag/chromadb'
OLLAMA_BASE_URL = "http://172.22.224.1:11434"

class EmailMeta(BaseModel):
    email_id: str
    subject: str
    from_: str
    to: List[str]
    date: str
    content: str

def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding from local Ollama model"""
    # Truncate text to avoid context length errors (mxbai-embed-large supports ~8192 tokens, use 4000 chars as safe limit)
    max_chars = 4000
    if len(text) > max_chars:
        text = text[:max_chars]
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL_NAME, "prompt": text},
            timeout=30
        )
        if response.status_code != 200:
            return None
        return response.json()["embedding"]
    except:
        return None

def parse_email(file_path) -> Dict:
    with open(file_path, 'rb') as f:
        e = email.parser.BytesParser().parsebytes(f.read())
        content = e.get_payload()
        if not isinstance(content, str):
            # Handle multipart messages
            content = str(content) if content else ""
        return {
            "email_id": os.path.basename(file_path),
            "subject": e['subject'] or "",
            "from_": e.get('from') or "",
            "to": e.get_all('to', []),
            "date": e['date'] or "",
            "content": content
        }

def load_emails() -> List[Dict]:
    emails = []
    for file in Path(EMAIL_DIR).glob('*.eml'):
        meta = parse_email(file)
        emails.append(meta)
    return emails

def chunk_text(emails: List[Dict], chunk_size=1000):
    chunks = []
    for email in emails:
        content = email['content'] or ""
        content_chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
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

def create_embeddings(chunks):
    """Create embeddings using local Ollama model"""
    print(f"Creating embeddings with {EMBEDDING_MODEL_NAME}...")
    for i, chunk in enumerate(chunks):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(chunks)}")
        text = str(chunk['text']) if chunk['text'] else ""
        chunk['embedding'] = get_embedding(text)
    
    # Filter out chunks with failed embeddings (all zeros)
    valid_chunks = [c for c in chunks if c['embedding'] and sum(c['embedding']) != 0]
    skipped = len(chunks) - len(valid_chunks)
    if skipped > 0:
        print(f"  Skipped {skipped} chunks with failed embeddings")
    return valid_chunks

def init_chroma():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(name="emails", metadata={"hnsw:space": "cosine"})
    return collection, chroma_client

def index_emails(chunks):
    collection, _ = init_chroma()
    
    # ChromaDB has a max batch size, so we need to batch
    batch_size = 5000
    total = len(chunks)
    
    print(f"Indexing {total} chunks in batches of {batch_size}...")
    
    for i in range(0, total, batch_size):
        batch = chunks[i:i+batch_size]
        ids = [chunk['id'] for chunk in batch]
        embeddings = [chunk['embedding'] for chunk in batch]
        documents = [chunk['text'] for chunk in batch]
        metadatas = [
            {"subject": chunk['subject'], "from": chunk['from_'], "to": chunk['to'], "date": chunk['date']}
            for chunk in batch
        ]
        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        print(f"  Indexed {min(i+batch_size, total)}/{total}")

def query_emails(prompt, collection):
    # Get embedding for query using local Ollama
    query_embedding = get_embedding(prompt)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    
    context = "\n".join(results.get('documents', [[]])[0])
    
    # Use Ollama via OpenAI-compatible API for generation
    import openai
    client = openai.OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")
    
    response = client.chat.completions.create(
        model=GEN_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an impartial case analyst. Analyze the retrieved email context and answer the user's question. Highlight supporting statements (helpful/strengthening) and opposing statements (detrimental/harmful) with quotes and dates."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
        ]
    )
    return response.choices[0].message.content

def main():
    # Initialize ChromaDB and check if already indexed
    collection, chroma_client = init_chroma()
    
    try:
        existing_count = collection.count()
        if existing_count > 0:
            print(f"Found {existing_count} already indexed chunks. Skipping indexing.")
            return
    except:
        pass
    
    # Load and parse emails
    emails = load_emails()
    print(f"Loaded {len(emails)} emails")
    
    # Chunk texts into pieces
    text_chunks = chunk_text(emails)
    print(f"Created {len(text_chunks)} chunks")
    
    # Create embeddings for chunks (using local Ollama)
    with_embeddings = create_embeddings(text_chunks)
    
    # Index the emails in ChromaDB
    print("Indexing emails...")
    index_emails(with_embeddings)
    print("Indexing complete!")

if __name__ == "__main__":
    main()

    collection, chroma_client = init_chroma()
    
    print("\n=== Email Analysis System Ready ===")
    print("All data is processed locally. No data sent to the internet.\n")
    
    while True:
        user_input = input("Query (type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        
        answer = query_emails(user_input, collection)
        print(f"\n{answer}")