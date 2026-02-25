# Email Case Analysis RAG System

A local, offline Retrieval-Augmented Generation (RAG) system for analyzing email archives. All data stays on your machine - no cloud services, no data leaves your computer.

## What It Does

- Parses `.eml` email files from a local directory
- Chunks email content into manageable segments
- Creates embeddings using local Ollama models
- Stores in ChromaDB (local vector database)
- Provides an interactive CLI to query emails using local AI models

## Prerequisites

### 1. Software Requirements

- **Python 3.11+** with pip
- **Ollama** (running on Windows with models installed)
- **PowerShell** (for Windows)

### 2. Python Packages

```bash
pip install pydantic chromadb requests openai
```

### 3. Ollama Models

Pull required models:

```powershell
ollama pull qwen2.5:14b    # For generation/answering
ollama pull mxbai-embed-large  # For embeddings
```

### 4. Firewall Setup (WSL to Windows)

If running Python from WSL but Ollama on Windows:

```powershell
# Run PowerShell as Administrator
netsh advfirewall firewall add rule name="Ollama" dir=in action=allow protocol=tcp localport=11434
```

Set Ollama to listen on all interfaces:

```powershell
# In PowerShell (as Admin), then restart Ollama
$env:OLLAMA_HOST="0.0.0.0"
```

### 5. Find Your Windows Host IP (from WSL)

```bash
ip route show | grep -i default | awk '{print $3}'
```

Update the `OLLAMA_BASE_URL` in `analyze_emails.py` if needed (default: `http://172.22.224.1:11434`).

## Setup

1. **Place emails**: Put your `.eml` files in `emails/` subdirectory

2. **Directory structure**:
   ```
   email-case-rag/
   ├── analyze_emails.py   # Main script
   ├── emails/             # Put .eml files here
   └── chromadb/           # Created automatically (vector DB)
   ```

3. **Configure paths** (in `analyze_emails.py`):
   ```python
   EMAIL_DIR = 'C:/Users/steve/projects/email-case-rag/emails'
   CHROMA_DIR = 'C:/Users/steve/projects/email-case-rag/chromadb'
   OLLAMA_BASE_URL = "http://YOUR_WINDOWS_IP:11434"
   ```

## Usage

### First Run (Indexing)

```bash
python analyze_emails.py
```

This will:
1. Load all `.eml` files from the `emails/` directory
2. Parse and chunk them into smaller segments
3. Generate embeddings using local Ollama (`mxbai-embed-large`)
4. Store in local ChromaDB

**First run takes time** - depends on number of emails and hardware. ~1 hour for 10k emails.

### Subsequent Runs

```bash
python analyze_emails.py
```

Detects existing index and skips re-indexing. Goes straight to query mode.

### Query Mode

Once indexed, type questions like:

```
Summarize key events and timeline
What statements support my position (excellent/strengthening)
What statements weaken my position (detrimental/harmful)
Search for mentions of [person name]
Search for money amounts
Search for dates
```

Type `exit` to quit.

## How It Works

1. **Email Parsing**: Reads `.eml` files, extracts subject, from, to, date, body
2. **Chunking**: Splits email content into ~1000 char segments
3. **Embedding**: Converts text to vectors using `mxbai-embed-large` (local Ollama)
4. **Storage**: ChromaDB stores vectors + text + metadata locally
5. **Query**: 
   - Your question is embedded
   - ChromaDB finds similar email chunks
   - `qwen2.5:14b` generates answer based on retrieved context

## System Requirements

- **RAM**: 16GB+ recommended (for Ollama models)
- **Disk**: ~2x email size for vector DB
- **GPU**: Optional but speeds up embedding/generation

## Privacy

- **100% local** - No data sent to internet
- All processing via Ollama on your machine
- Vector DB stored locally in `chromadb/` folder

## Troubleshooting

### Ollama not accessible from WSL
- Ensure firewall rule is added (see step 4 above)
- Check Windows IP: `ip route show | grep default` in WSL
- Update `OLLAMA_BASE_URL` in script

### Context length errors
- Script automatically truncates to 4000 chars per chunk

### Slow indexing
- Reduce chunk size in `chunk_text()` function
- Current: 1000 chars per chunk
