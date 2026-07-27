# Advance-rag-Comparison-of-normal-retrieval-with-Self-query-retrieval

A notebook demonstrating **Self-Query Retrieval** — a technique where an LLM converts a natural-language question into a structured metadata filter, instead of relying on plain semantic similarity alone.

## 🔴 The Core Problem: Normal Retriever vs Self-Query Retriever

**Normal retriever** only compares text meaning. It has no awareness of metadata constraints like year ranges, genre, or rating — it just finds text that sounds similar to your question.

```python
question1 = "What's a movie after 1990 but before 2005 that's all about toys, and preferably is animated"
vectorstore.similarity_search(question1)
```
🔴 **Result:** returns loosely related movies (dinosaurs, romance, comedy) — ignores the year range and "animated" requirement entirely, because it only matches on word/phrase similarity, not actual filterable facts.

**Self-query retriever** uses an LLM to read the question, extract the real constraints (`year > 1990`, `year < 2005`, `genre = animated`), convert them into a structured filter, and only then search.

```python
retriever.invoke(
    "What's a movie after 1990 but before 2005 that's all about toys, and preferably is animated"
)
```
🟢 **Result:** returns only movies that genuinely satisfy all three conditions — correct year range and correct genre, not just similar-sounding text.

## 🆚 Side-by-Side Difference

| Aspect | Normal Retriever | Self-Query Retriever |
|---|---|---|
| How it searches | Pure text similarity | LLM-generated structured filter + similarity |
| Understands "before 2005" | ❌ No — treated as keywords | ✅ Yes — becomes `year < 2005` |
| Understands "animated" | ❌ No — fuzzy word match only | ✅ Yes — becomes `genre = animated` |
| Accuracy on filtered Qs | 🔴 Low — irrelevant results | 🟢 High — only valid matches |
| Needs metadata schema | No | Yes (`AttributeInfo` definitions) |
| Needs LLM call | No | Yes (to build the structured query) |

## 🛠 Tech Stack

- LLM: Groq (`llama-3.1-8b-instant`)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, free)
- Vector Store: Chroma (needed for metadata filtering — FAISS lacks a built-in translator)
- Framework: LangChain 1.x + `langchain-classic` (self-query classes moved here)

## 📦 Setup

```python
!pip install -q langchain langchain_community langchain-huggingface langchain-groq langchain-classic langchain_chroma faiss-cpu sentence-transformers lark
```
```python
import os
os.environ["GROQ_API_KEY"] = "your-groq-api-key-here"
```
🔴 Never hardcode your API key directly in a notebook cell — use an environment variable or Colab secret, especially before sharing or committing the file.

## 🚀 Core Flow

**1. Documents with metadata**
```python
docs = [
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    # ...more documents
]
```

**2. Build vector store**
```python
vectorstore = Chroma.from_documents(docs, embedding)
```

**3. Define metadata schema (what the LLM is allowed to filter on)**
```python
metadata_field_info = [
    AttributeInfo(name="genre", description="Movie genre", type="string"),
    AttributeInfo(name="year", description="Release year", type="integer"),
    AttributeInfo(name="rating", description="1-10 rating", type="float"),
]
```

**4. Build the query constructor (turns language → structured filter)**
```python
query_constructor = prompt | llm | output_parser
```

**5. Assemble the self-query retriever**
```python
retriever = SelfQueryRetriever(
    query_constructor=query_constructor,
    vectorstore=vectorstore,
    structured_query_translator=ChromaTranslator(),
)
```

## ⚠️ Known Issues Fixed in This Notebook

🔴 `SelfQueryRetriever.from_llm(...)` breaks with `ImportError: DatabricksVectorSearch` — fixed by building the retriever manually with an explicit `structured_query_translator`.

🔴 FAISS has no built-in translator for self-query filtering — switched to Chroma.

🔴 `chromadb` conflicts with pre-installed `opentelemetry` packages on Colab — fixed via clean uninstall + reinstall of both.

## 📚 Key Takeaway

Use a normal retriever when your query is purely about semantic meaning. Use a self-query retriever when your query mixes meaning with hard constraints — dates, categories, ratings, names — that should be filtered exactly, not fuzzily matched.
