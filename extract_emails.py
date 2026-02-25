import os
import json
from pathlib import Path
from unstructured.partition.email import partition_email
import ollama

# Paths
email_folder = Path("C:/Emails/CaseFolder")  # ← change to your folder
output_file = "extracted_emails.jsonl"

extracted = []

for eml_file in email_folder.glob("*.eml"):
    try:
        elements = partition_email(filename=str(eml_file))
        text = "\n".join([el.text for el in elements if el.text])
        
        # Basic metadata
        metadata = {
            "filename": eml_file.name,
            "subject": "", "from": "", "to": "", "date": ""
        }
        with open(eml_file, "rb") as f:
            raw = f.read()
            msg = email.message_from_bytes(raw)
            metadata["subject"] = msg["subject"] or ""
            metadata["from"] = msg["from"] or ""
            metadata["to"] = msg["to"] or ""
            metadata["date"] = msg["date"] or ""
        
        extracted.append({
            "metadata": metadata,
            "content": text.strip()
        })
        print(f"Processed: {eml_file.name}")
    except Exception as e:
        print(f"Error on {eml_file.name}: {e}")

# Save as JSONL (one email per line)
with open(output_file, "w", encoding="utf-8") as f:
    for item in extracted:
        f.write(json.dumps(item) + "\n")

print(f"Extracted {len(extracted)} emails to {output_file}")