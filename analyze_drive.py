import os
import time
import re
from typing import List, Dict, Optional
from pathlib import Path
import chromadb
import numpy as np
import torch

# Configuration
DRIVE_DIR = 'C:/Users/steve/projects/drive-case-rag/takeout_drive/Takeout/Drive'
GEN_MODEL_NAME = 'qwen2.5:14b'
CHROMA_DIR = 'C:/Users/steve/projects/drive-case-rag/chromadb_drive'
OLLAMA_BASE_URL = "http://localhost:11434"

# Embedding model
EMBEDDING_MODEL = None

def get_embedding_model():
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        print("Loading embedding model (GPU)...")
        from sentence_transformers import SentenceTransformer
        EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:0')
        print("Embedding model loaded on GPU")
    return EMBEDDING_MODEL

def extract_text_from_file(file_path: str) -> str:
    """Extract text from various file types"""
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        elif ext == '.docx':
            try:
                from docx import Document
                doc = Document(file_path)
                return '\n'.join([p.text for p in doc.paragraphs])
            except:
                return ""
        
        elif ext == '.pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    return '\n'.join([page.extract_text() or '' for page in reader.pages])
            except:
                return ""
        
        elif ext == '.html' or ext == '.htm':
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    html = f.read()
                # Simple HTML stripping
                text = re.sub(r'<[^>]+>', ' ', html)
                text = re.sub(r'\s+', ' ', text)
                return text
            except:
                return ""
        
        else:
            # For other types, try plain text
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()[:10000]  # Limit size
            except:
                return ""
    except Exception as e:
        return ""

def load_drive_files() -> List[Dict]:
    """Load relevant files from Google Drive Takeout"""
    docs = []
    drive_path = Path(DRIVE_DIR)
    
    # Get all files recursively
    all_files = list(drive_path.rglob('*'))
    total = len(all_files)
    print(f"Scanning {total} files...")
    
    text_extensions = ['.txt', '.docx', '.pdf', '.html', '.htm', '.json', '.csv']
    
    for i, file_path in enumerate(all_files):
        if not file_path.is_file():
            continue
        
        ext = file_path.suffix.lower()
        if ext not in text_extensions:
            continue
        
        # Extract text
        try:
            content = extract_text_from_file(str(file_path))
            if len(content) < 50:  # Skip empty/very short files
                continue
            
            relative_path = str(file_path.relative_to(drive_path))
            
            docs.append({
                "file_id": str(file_path.name)[:100],
                "path": relative_path,
                "content": content
            })
            
            if len(docs) % 50 == 0:
                print(f"  Loaded {len(docs)} documents...")
        except Exception as e:
            continue
    
    print(f"Loaded {len(docs)} documents")
    return docs

def chunk_text(docs: List[Dict], chunk_size=1000):
    """Chunk documents into smaller pieces"""
    chunks = []
    for doc in docs:
        content = doc['content'] or ""
        content_chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        for i, chunk in enumerate(content_chunks):
            chunks.append({
                "id": f"{doc['file_id']}_{i}",
                "text": chunk[:2000],  # Limit chunk size
                "path": doc['path']
            })
    return chunks

def create_embeddings(chunks):
    """Create embeddings using GPU"""
    model = get_embedding_model()
    total = len(chunks)
    print(f"Creating embeddings with all-MiniLM-L6-v2 (GPU)...")
    
    texts = [str(chunk['text'])[:512] if chunk['text'] else "" for chunk in chunks]
    
    start_time = time.time()
    batch_size = 512
    all_embeddings = []
    
    for i in range(0, total, batch_size):
        batch_texts = texts[i:i+batch_size]
        embeddings = model.encode(batch_texts, batch_size=len(batch_texts), show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.extend(embeddings.tolist())
        
        done = min(i + batch_size, total)
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  Progress: {done}/{total} ({rate:.0f}/sec, ETA: {eta/60:.1f} min)", end='\r')
    
    print()
    
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = all_embeddings[i]
    
    valid_chunks = [c for c in chunks if c['embedding'] and sum(c['embedding']) != 0]
    elapsed = time.time() - start_time
    print(f"  Created {len(valid_chunks)} embeddings in {elapsed:.1f}s ({len(valid_chunks)/elapsed:.0f}/sec)")
    return valid_chunks

def get_query_embedding(text: str) -> List[float]:
    model = get_embedding_model()
    embedding = model.encode([text], show_progress_bar=False, convert_to_numpy=True)
    return embedding.tolist()[0]

def init_chroma():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(name="drive_docs", metadata={"hnsw:space": "cosine"})
    return collection, chroma_client

def index_docs(chunks):
    collection, _ = init_chroma()
    
    batch_size = 1000
    total = len(chunks)
    
    print(f"Indexing {total} chunks in batches of {batch_size}...")
    
    for i in range(0, total, batch_size):
        batch = chunks[i:i+batch_size]
        
        # Clean IDs and make unique
        ids = []
        seen = {}
        for j, chunk in enumerate(batch):
            orig_id = chunk['id'][:50]
            if orig_id in seen:
                seen[orig_id] += 1
                id_clean = f"{orig_id}_{seen[orig_id]}"
            else:
                seen[orig_id] = 0
                id_clean = orig_id
            id_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', id_clean)[:100]
            ids.append(id_clean)
        
        embeddings = [chunk['embedding'] for chunk in batch]
        documents = [chunk['text'] for chunk in batch]
        metadatas = [{"path": chunk['path'][:500]} for chunk in batch]
        
        try:
            collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
            print(f"  Indexed {min(i+batch_size, total)}/{total}")
        except Exception as e:
            print(f"  Error at batch {i}: {e}")
            # Try with sequential IDs
            ids = [f"doc_{i+j}" for j in range(len(batch))]
            collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
            print(f"  Indexed {min(i+batch_size, total)}/{total}")

def query_docs(prompt, collection):
    query_embedding = get_query_embedding(prompt)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    
    context = "\n\n---\n\n".join(results.get('documents', [[]])[0])
    
    # Get sources
    paths = results.get('metadatas', [[{}]])[0]
    sources = "\n".join([f"- {p.get('path', 'Unknown')}" for p in paths])
    
    import openai
    client = openai.OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")
    
    response = client.chat.completions.create(
        model=GEN_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a legal case analyst. Analyze the retrieved document context and answer the user's question directly. Include relevant quotes, dates, and file names when available."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
        ]
    )
    
    answer = response.choices[0].message.content
    return f"{answer}\n\n---\nSources:\n{sources}"

def main():
    collection, chroma_client = init_chroma()
    
    try:
        existing_count = collection.count()
        if existing_count > 0:
            print(f"Found {existing_count} already indexed chunks. Skipping indexing.")
            return
    except:
        pass
    
    docs = load_drive_files()
    text_chunks = chunk_text(docs)
    print(f"Created {len(text_chunks)} chunks")
    
    with_embeddings = create_embeddings(text_chunks)
    
    print("Indexing documents...")
    index_docs(with_embeddings)
    print("Indexing complete!")

if __name__ == "__main__":
    main()
    
    collection, chroma_client = init_chroma()
    
    print("\n=== Drive Document Analysis System Ready ===")
    print("All data is processed locally. No data sent to the internet.\n")
    
    while True:
        user_input = input("\033[91mQuery (type 'exit' to quit): \033[0m")
        if user_input.lower() == 'exit':
            break
        
        answer = query_docs(user_input, collection)
        print(f"\n{answer}")
