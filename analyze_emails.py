import os
import time
from typing import List, Dict, Optional
from pathlib import Path
import email.parser
from pydantic import BaseModel
import chromadb
import numpy as np
import torch

# Configuration
EMAIL_DIR = 'C:/Users/steve/projects/email-case-rag/emails'
GEN_MODEL_NAME = 'qwen3.5-122b'  # Ollama model for generation (local)
CHROMA_DIR = 'C:/Users/steve/projects/email-case-rag/chromadb'
OLLAMA_BASE_URL = "http://localhost:11434"

# Embedding model - using sentence-transformers with MULTI-GPU
EMBEDDING_MODEL = None

def get_embedding_model():
    """Load sentence-transformers model with multi-GPU"""
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        print("Loading embedding model (multi-GPU)...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('BAAI/bge-large-en-v1.5', device='cuda')
        
        # Use multiple GPUs if available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            from torch.nn import DataParallel
            model = DataParallel(model)
            model = model.cuda()
        
        EMBEDDING_MODEL = model
        print(f"Embedding model loaded on {torch.cuda.device_count()} GPU(s)")
    return EMBEDDING_MODEL

class EmailMeta(BaseModel):
    email_id: str
    subject: str
    from_: str
    to: List[str]
    date: str
    content: str

def parse_email(file_path) -> Dict:
    """Parse email file and extract clean content"""
    with open(file_path, 'rb') as f:
        msg = email.message_from_bytes(f.read())
    
    subject = msg['subject'] or ""
    from_addr = msg['from'] or ""
    to_addrs = msg.get_all('to', [])
    date = msg['date'] or ""
    
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == 'text/plain':
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        body = payload.decode(charset, errors='ignore')
                        break
                except:
                    pass
        if not body:
            for part in msg.walk():
                if part.get_content_type() == 'text/html':
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or 'utf-8'
                            html = payload.decode(charset, errors='ignore')
                            import re
                            body = re.sub(r'<[^>]+>', ' ', html)
                            body = re.sub(r'\s+', ' ', body)
                            break
                    except:
                        pass
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or 'utf-8'
                body = payload.decode(charset, errors='ignore')
        except:
            body = ""
    
    return {
        "email_id": os.path.basename(file_path),
        "subject": subject,
        "from_": from_addr,
        "to": to_addrs,
        "date": date,
        "content": body
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
    """Create embeddings using GPU-accelerated sentence-transformers"""
    model = get_embedding_model()
    
    total = len(chunks)
    print(f"Creating embeddings with BAAI/bge-large-en-v1.5 (GPU)...")
    
    # Extract texts, truncating to 512 chars for efficiency
    texts = [str(chunk['text'])[:512] if chunk['text'] else "" for chunk in chunks]
    
    start_time = time.time()
    
    # Process in batches on GPU
    batch_size = 256
    all_embeddings = []
    
    for i in range(0, total, batch_size):
        batch_texts = texts[i:i+batch_size]
        embeddings = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())
        
        # Progress update
        done = min(i + batch_size, total)
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  Progress: {done}/{total} ({rate:.0f}/sec, ETA: {eta/60:.1f} min)", end='\r')
    
    print()  # New line after progress
    
    # Attach embeddings to chunks
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = all_embeddings[i]
    
    valid_chunks = [c for c in chunks if c['embedding'] and sum(c['embedding']) != 0]
    elapsed = time.time() - start_time
    print(f"  Created {len(valid_chunks)} embeddings in {elapsed:.1f}s ({len(valid_chunks)/elapsed:.0f}/sec)")
    return valid_chunks

def get_query_embedding(text: str) -> List[float]:
    """Get embedding for a query string"""
    model = get_embedding_model()
    embedding = model.encode([text], show_progress_bar=False)
    return embedding.tolist()[0]

def init_chroma():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(name="emails", metadata={"hnsw:space": "cosine"})
    return collection, chroma_client

def index_emails(chunks):
    collection, _ = init_chroma()
    
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
    query_embedding = get_query_embedding(prompt)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    
    context = "\n".join(results.get('documents', [[]])[0])
    
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
    collection, chroma_client = init_chroma()
    
    try:
        existing_count = collection.count()
        if existing_count > 0:
            print(f"Found {existing_count} already indexed chunks. Skipping indexing.")
            return
    except:
        pass
    
    emails = load_emails()
    print(f"Loaded {len(emails)} emails")
    
    text_chunks = chunk_text(emails)
    print(f"Created {len(text_chunks)} chunks")
    
    with_embeddings = create_embeddings(text_chunks)
    
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
