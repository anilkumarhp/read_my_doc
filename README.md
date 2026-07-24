# Ask My Own Documents

A tool that lets you ask a plain-English question and get a plain-English answer, pulled from your own documents, with the specific source it came from.

Not a chatbot that knows everything - a tool that knows exactly what you told it and nothing else, and says so honestly when your documents don't cover what you asked.

This is project 1 of a RAG learning series. It is deliberately the simplest one: no reranking, no query rewriting, no agent deciding things on its own. It's the baseline every later project builds on.

## The problem this solves

Most people have a folder somewhere - notes app, Google Drive, a pile of PDFs - full of information they wrote down specifically so they wouldn't forget it, and then never actually search, because searching it is worse than just trying to remember. Ctrl-F only finds exact words. Skimming ten documents to answer one question takes longer than the question was worth. So the information sits there, technically saved, practically useless.

This project builds the smallest possible fix for that.

## The tools, and why each one

Every tool was picked for a specific reason, not because it's popular.

| Tool                      | Why                                                                                                                                                                                                                      |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **sentence-transformers** | Turns text into vectors, locally. No API key, no per-call cost. A learner shouldn't need a credit card to try the first project.                                                                                         |
| **ChromaDB**              | Stores those vectors and searches them. Local, free, starts up with one line of code, no server to configure. The simplest vector database that's still a real one.                                                      |
| **Ollama**                | Runs an open-weight language model on your own machine to generate answers. No API key, no rate limit, no data leaving your computer. Slower and less capable than a hosted frontier model - a fair trade for project 1. |
| **Plain Python**          | No LangChain, no LlamaIndex, nothing wrapping the logic in abstractions yet. Seeing the raw logic is the point.                                                                                                          |

**Why not a hosted API from the start?** Because the first project in a learning series should have zero cost and zero external dependency to even try. API keys, rate limits, and hosted model tradeoffs show up starting in project 2, once you already understand what's happening underneath them.

## The three phases

```
Phase 1              Phase 2                     Phase 3
Ingestion    →    Retrieval + Generation    →    Evaluation
```

**Phase 1 - Ingestion** ([ingest.py](ingest.py)): takes your raw text files, breaks them into searchable pieces, and stores them. Answers "why split text at all" and "why split it this specific way."

**Phase 2 - Retrieval and Generation** ([query.py](query.py)): takes a question, finds the pieces of your documents most likely to answer it, and asks a local model to answer using only those pieces, citing which one it used. This is where "don't make things up" turns from a hope into an actual instruction the model follows.

**Phase 3 - Evaluation** ([evaluate.py](evaluate.py)): checks whether the answers were actually honest - grounded in what was retrieved, not invented. The smallest possible version of a discipline that becomes central in every later, production-grade project.

## Setup

Requires Python 3.11+ and [Ollama](https://ollama.com) installed and running.

```bash
pip install -r requirements.txt
ollama pull llama3.2
```

## Smoke test

Confirm the embedding model downloads and ChromaDB round-trips a document before building anything on top of it:

```bash
python -c "
from chromadb.utils import embedding_functions
import chromadb

# This line downloads the embedding model on first run.
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-L6-v2'
)

client = chromadb.Client()
collection = client.create_collection('smoke_test', embedding_function=embedding_fn)
collection.add(documents=['the sky is blue'], ids=['1'])
result = collection.query(query_texts=['what color is the sky'], n_results=1)
print('Retrieved:', result['documents'][0][0])
"
```

Expected output: `Retrieved: the sky is blue`

## Project structure

| File                                     | Purpose                                                        |
| ---------------------------------------- | -------------------------------------------------------------- |
| [main.py](main.py)                       | Smoke test - verifies embeddings + ChromaDB work end to end    |
| [ingest.py](ingest.py)                   | Phase 1 - load documents, split into chunks, store in ChromaDB |
| [query.py](query.py)                     | Phase 2 - retrieve relevant chunks, generate a cited answer    |
| [evaluate.py](evaluate.py)               | Phase 3 - check answers are grounded in retrieved text         |
| [eval_questions.txt](eval_questions.txt) | Questions used by the evaluation phase                         |
