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
GEN_MODEL_NAME = 'qwen2.5:14b'  # Ollama model for generation (local)
CHROMA_DIR = 'C:/Users/steve/projects/email-case-rag/chromadb'
OLLAMA_BASE_URL = "http://localhost:11434"

# Filter by ProtonMail label/folder (set to None to index all emails)
# Set to a list of label IDs to filter by multiple folders
FOLDER_LABEL_IDS = [
    "dj1Oip_9lBDYQgxbFW9C8p9dishNOoOYb7_t8Iyv2RbsvpearIaOhRzKtUTcQaMl9FkcLNBtayJDMtiSz92Hbw==",  # TXAirlots
    "yHZ4f0SyrjgzjY52EshijxzhJLMSQqlGRP2irkIfr4-MjPknNH6rCbWXKGX_ZKoIfkpowrZEeQB6bxc-qwhBmA==",  # Mediated Meeting
    "jSQLfeMYhR2HefOGrfB3CbE6tB14rbkaZZz-CVP5mpAgE8PrnNslSqVkfkQEcyKArCcoE9YwjF0Z-1mHII6xiQ==",  # TXAirlots subfolder
    "r9w4EZaVHzNmWXZCPCcAPX7yZvJkLm33TklAdsQqDUKqEWaJh7-h6-99oyVbscJ1n9GCy243hF91UsimjupOdg==",  # 1008 Group LLC
    "vtaY4A6Gnbe0uUdjxPI0sQ8gZk_8zNmlIKyDS8eIIHT5aP1jxVf1wiDowuDySWfwhTxlE-p7m1qLnn8ulTHSYw==",  # Dorsett Johnson
    "c_p_HlElp9k0zwR2lnuQhZ5wWs7hUMxl55DOKGYRdXsoUxL5v8OebHix4n5caL_4638vMMC5NQAXL-QuoTpW2w==",  # $46,781 deposit
]

# Embedding model - using sentence-transformers with MULTI-GPU
EMBEDDING_MODEL = None

def get_embedding_model():
    """Load sentence-transformers model with GPU"""
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        print("Loading embedding model (GPU)...")
        from sentence_transformers import SentenceTransformer
        EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:0')
        print("Embedding model loaded on GPU")
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
    import json
    emails = []
    email_files = list(Path(EMAIL_DIR).glob('*.eml'))
    total = len(email_files)
    print(f"Loading {total} emails...")
    
    for i, eml_file in enumerate(email_files):
        # Check corresponding metadata file
        metadata_file = eml_file.with_suffix('.metadata.json')
        
        if FOLDER_LABEL_IDS:
            if not metadata_file.exists():
                continue
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                labels = metadata.get('Payload', {}).get('LabelIDs', [])
                # Check if ANY of the folder IDs are in the email's labels
                if not any(fid in labels for fid in FOLDER_LABEL_IDS):
                    continue
            except:
                continue
        
        meta = parse_email(eml_file)
        emails.append(meta)
        
        if (i + 1) % 1000 == 0:
            print(f"  Loaded {i + 1}/{total}")
    
    print(f"Loaded {len(emails)} emails (filtered by {len(FOLDER_LABEL_IDS)} folders)")
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
    print(f"Creating embeddings with all-MiniLM-L6-v2 (GPU)...")
    
    # Extract texts, truncating to 512 chars for efficiency
    texts = [str(chunk['text'])[:512] if chunk['text'] else "" for chunk in chunks]
    
    start_time = time.time()
    
    # Batch size
    batch_size = 512
    all_embeddings = []
    
    for i in range(0, total, batch_size):
        batch_texts = texts[i:i+batch_size]
        embeddings = model.encode(batch_texts, batch_size=len(batch_texts), show_progress_bar=False, convert_to_numpy=True)
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
    embedding = model.encode([text], show_progress_bar=False, convert_to_numpy=True)
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
            {"role": "system", "content": "You are a legal case analyst. Analyze the retrieved email context and answer the user's question directly. Include relevant quotes, dates, and sender/recipient information when available."},
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
        user_input = input("\033[91mQuery (type 'exit' to quit): \033[0m")
        if user_input.lower() == 'exit':
            break
        
        answer = query_emails(user_input, collection)
        print(f"\n{answer}")
