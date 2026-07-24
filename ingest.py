import os

def load_documents(folder: str) -> list[dict]:
    """Read every .txt file in folder. Returns a list of {source, text} dicts."""
    documents = []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append({"source": filename, "text": text})
    return documents