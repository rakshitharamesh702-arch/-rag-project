from fastapi import FastAPI, HTTPException
from pypdf import PdfReader
import boto3
import json
import io
import math
import logging
import time
from datetime import datetime
from typing import List, Dict

app = FastAPI()

# ---------------- CONFIG ----------------
REGION = "us-east-1"

BUCKET_NAME = "rakshitha-rag-pdf-us-east-1"   # change if needed

# Where PDFs live in S3, e.g. "pdfs/deadlock.pdf"
# You said you don't want prefix, so keep it empty string
PDF_PREFIX = ""

# Where we store RAG index JSON files in S3 (folder in bucket)
# IMPORTANT: previously you had " " (space) which breaks things
INDEX_PREFIX = "indexes/"

# DynamoDB table for document metadata
DOC_TABLE_NAME = "rag_documents"  # create this table in DynamoDB

# Bedrock models
EMBED_MODEL_ID = "amazon.titan-embed-text-v1"
TEXT_MODEL_ID = "amazon.titan-text-lite-v1"   # use premier if larger context needed

# ---------------- AWS CLIENTS ----------------
s3 = boto3.client("s3", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
doc_table = dynamodb.Table(DOC_TABLE_NAME)

# ---------------- LOGGING ----------------
logger = logging.getLogger("rag_app")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def log_metric(operation: str, latency_ms: float, extra: Dict = None) -> None:
    """
    Simple CloudWatch-style metric logger.
    Logs a JSON line that can be picked up by CloudWatch when deployed on AWS.
    """
    metric = {
        "type": "metric",
        "operation": operation,
        "latency_ms": round(latency_ms, 2),
        "timestamp_utc": datetime.utcnow().isoformat()
    }
    if extra:
        metric.update(extra)
    logger.info(f"METRIC: {json.dumps(metric)}")


# ---------------- HELPERS ----------------
def chunk_text(text: str, max_chars: int = 800) -> List[str]:
    """Simple paragraph-aware chunking."""
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 <= max_chars:
            current += ("\n\n" if current else "") + para
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return chunks


def call_bedrock_text(prompt: str, max_tokens: int = 400, temperature: float = 0.3) -> str:
    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": max_tokens,
            "temperature": temperature,
            "stopSequences": []
        }
    })

    resp = bedrock.invoke_model(
        modelId=TEXT_MODEL_ID,
        body=body
    )
    raw = resp["body"].read().decode("utf-8")
    data = json.loads(raw)
    return data["results"][0]["outputText"]


def get_embedding(text: str) -> List[float]:
    body = json.dumps({
        "inputText": text
    })
    resp = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=body
    )
    raw = resp["body"].read().decode("utf-8")
    data = json.loads(raw)
    return data["embedding"]


def cosine_sim(a: List[float], b: List[float]) -> float:
    """Pure python cosine similarity."""
    if len(a) != len(b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def rerank_chunks(question: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    Use the Bedrock text model as a reranker:
    - Input: question + list of candidate chunks
    - Output: same chunks, sorted by LLM relevance score
    """
    rerank_prompt = """
You are a reranking model.
Given a question and a list of chunks, assign each chunk a relevance score between 0 and 1.
Return ONLY a JSON list of objects like:
[
  {"id": 0, "score": 0.91},
  {"id": 3, "score": 0.72}
]

Question:
{question}

Chunks:
{chunks}
"""

    # Send only id + truncated text (to keep prompt small)
    chunk_text_list = [
        {"id": ch["id"], "text": ch["text"][:500]}
        for ch in chunks
    ]

    prompt = rerank_prompt.format(
        question=question,
        chunks=json.dumps(chunk_text_list)
    )

    response = call_bedrock_text(prompt, max_tokens=500, temperature=0.0)

    try:
        rerank_scores = json.loads(response)
    except Exception as e:
        # If LLM doesn't return valid JSON, fall back to original ordering
        logger.warning(f"Reranker JSON parse failed, using cosine order. Error: {e}")
        return chunks[:top_k]

    # Map id -> score
    score_map = {
        str(item["id"]): float(item["score"])
        for item in rerank_scores
        if "id" in item and "score" in item
    }

    # Attach rerank_score to chunks
    for ch in chunks:
        ch["rerank_score"] = score_map.get(str(ch["id"]), 0.0)

    # Sort by rerank_score descending
    sorted_chunks = sorted(chunks, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

    return sorted_chunks[:top_k]


def save_doc_metadata(
    doc_id: str,
    s3_pdf_key: str,
    s3_index_key: str,
    num_chunks: int,
    status: str = "INGESTED",
) -> None:
    """Store/Update document metadata in DynamoDB."""
    now = datetime.utcnow().isoformat()
    item = {
        "doc_id": doc_id,
        "s3_pdf_key": s3_pdf_key,
        "s3_index_key": s3_index_key,
        "num_chunks": num_chunks,
        "status": status,
        "last_updated": now,
    }

    logger.info(f"Saving metadata to DynamoDB: {item}")
    doc_table.put_item(Item=item)


# ---------------- ENDPOINT: list_pdfs ----------------
@app.get("/list_pdfs")
def list_pdfs():
    """List all PDFs under PDF_PREFIX in your bucket."""
    start = time.perf_counter()
    try:
        logger.info(f"Listing PDFs with prefix='{PDF_PREFIX}' in bucket={BUCKET_NAME}")
        resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PDF_PREFIX)
        keys = []
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".pdf"):
                keys.append(key)

        latency = (time.perf_counter() - start) * 1000
        log_metric("list_pdfs", latency, {"pdf_count": len(keys)})

        return {"pdfs": keys}
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        log_metric("list_pdfs_error", latency, {"error": str(e)})
        logger.exception(f"Failed to list PDFs: {e}")
        raise HTTPException(status_code=500, detail="Failed to list PDFs")


# ---------------- ENDPOINT: ingest_from_s3 ----------------
@app.get("/ingest_from_s3")
def ingest_from_s3(key: str):
    """
    Build RAG index for one PDF stored in S3:
    - Download PDF
    - Extract text
    - Chunk
    - Embed each chunk
    - Generate summary + keywords
    - Save index JSON to S3
    - Save metadata to DynamoDB
    """
    start = time.perf_counter()
    logger.info(f"Starting ingestion for key={key}")

    try:
        # 1) Download PDF
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        pdf_bytes = obj["Body"].read()
        logger.info(f"Downloaded PDF from S3: {key}")

        # 2) Extract text
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text_parts: List[str] = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
        full_text = "\n\n".join(text_parts)

        if not full_text.strip():
            logger.warning(f"No text extracted from PDF {key}")
        else:
            logger.info(f"Extracted text length for {key}: {len(full_text)} characters")

        # 3) Chunk text
        chunks = chunk_text(full_text, max_chars=800)
        logger.info(f"Created {len(chunks)} chunks for {key}")

        # 4) Embeddings for each chunk
        indexed_chunks = []
        for idx, ch in enumerate(chunks):
            emb = get_embedding(ch)
            indexed_chunks.append({
                "id": idx,
                "text": ch,
                "embedding": emb
            })

        # 5) Summary + keywords
        summary_prompt = f"Summarise this document in around 10–12 lines:\n\n{full_text}"
        keywords_prompt = f"Extract 10–15 important keywords (comma separated):\n\n{full_text}"

        summary = call_bedrock_text(summary_prompt, max_tokens=400, temperature=0.3)
        keywords_text = call_bedrock_text(keywords_prompt, max_tokens=200, temperature=0.2)

        # 6) Save index JSON to S3
        index = {
            "doc_id": key,
            "chunks": indexed_chunks,
            "summary": summary,
            "keywords": keywords_text
        }

        index_key = f"{INDEX_PREFIX}{key}.json"  # e.g. indexes/Climate Change.pdf.json

        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=index_key,
            Body=json.dumps(index).encode("utf-8"),
            ContentType="application/json"
        )
        logger.info(f"Stored index in S3 at {index_key}")

        # 7) Save metadata to DynamoDB
        save_doc_metadata(
            doc_id=key,
            s3_pdf_key=key,
            s3_index_key=index_key,
            num_chunks=len(indexed_chunks),
            status="INGESTED",
        )

        latency = (time.perf_counter() - start) * 1000
        log_metric("ingest_from_s3", latency, {
            "doc_id": key,
            "num_chunks": len(indexed_chunks)
        })

        return {
            "message": f"Ingested {key} into RAG index",
            "pdf_key": key,
            "index_key": index_key,
            "num_chunks": len(indexed_chunks)
        }

    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        log_metric("ingest_from_s3_error", latency, {
            "doc_id": key,
            "error": str(e)
        })
        logger.exception(f"Ingestion failed for key={key}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ---------------- ENDPOINT: list_docs (DynamoDB) ----------------
@app.get("/list_docs")
def list_docs():
    """
    List all documents tracked in DynamoDB metadata table.
    """
    start = time.perf_counter()
    try:
        logger.info("Scanning DynamoDB for document metadata")
        resp = doc_table.scan()
        items = resp.get("Items", [])

        latency = (time.perf_counter() - start) * 1000
        log_metric("list_docs", latency, {"doc_count": len(items)})

        return {"documents": items}
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        log_metric("list_docs_error", latency, {"error": str(e)})
        logger.exception(f"Failed to list documents from DynamoDB: {e}")
        raise HTTPException(status_code=500, detail="Failed to list document metadata")

# ---------------- LLM RERANKER ----------------

def rerank_chunks(question: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    Simple, safe reranker:
    - DOESN'T assume chunk["id"] exists
    - Uses LLM (call_bedrock_text) to rerank
    - Falls back to cosine order if LLM output is bad or fails
    """
    if not chunks:
        return []

    # Give every chunk a safe ID so we never crash on "id"
    normalized = []
    for i, ch in enumerate(chunks):
        cid = (
            ch.get("id")
            or ch.get("chunk_id")
            or ch.get("index")
            or f"chunk_{i}"
        )
        normalized.append({
            "id": cid,
            "text": ch.get("text", ""),
            "score": ch.get("score", 0.0),
            "raw": ch,   # original dict
            "idx": i,
        })

    # Build a compact prompt for LLM
    chunks_block = ""
    for i, ch in enumerate(normalized):
        chunks_block += f"[{i}] (score={ch['score']:.4f})\n{ch['text']}\n\n"

    prompt = f"""
You are a reranker.

Question:
{question}

You are given several text chunks, each labeled with an index in [brackets].
Choose the {top_k} most relevant chunks to answer the question.

Return ONLY a JSON array of indices. Example:
[1, 0, 3]

Chunks:
{chunks_block}
"""

    try:
        raw = call_bedrock_text(prompt, max_tokens=150, temperature=0.0).strip()

        import json, re
        try:
            indices = json.loads(raw)
        except json.JSONDecodeError:
            # fallback: pull all integers from the string
            indices = list(map(int, re.findall(r"\d+", raw)))

        # keep only valid indices, at most top_k
        indices = [
            i for i in indices
            if isinstance(i, int) and 0 <= i < len(normalized)
        ][:top_k]

        if not indices:
            # fallback to cosine ranking if LLM output is garbage
            return sorted(chunks, key=lambda x: x["score"], reverse=True)[:top_k]

        # map back to original chunks
        return [normalized[i]["raw"] for i in indices]

    except Exception as e:
        logger.warning(f"rerank_chunks failed, falling back to cosine ranking: {e}")
        return sorted(chunks, key=lambda x: x["score"], reverse=True)[:top_k]


# ---------------- ENDPOINT: ask_rag (single-document, with reranking) ----------------

@app.get("/ask_rag")
def ask_rag(doc_id: str, question: str):
    """
    Answer using REAL RAG with two-stage retrieval:
    1) Cosine similarity to get top 10 candidates from one document
    2) LLM-based reranker to pick the best 3 chunks
    """
    start = time.perf_counter()
    logger.info(f"ask_rag called for doc_id={doc_id}, question='{question}'")
    try:
        index_key = f"{INDEX_PREFIX}{doc_id}.json"
        logger.info(f"Loading index from S3: {index_key}")
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=index_key)
        index = json.loads(obj["Body"].read().decode("utf-8"))
        chunks = index["chunks"]

        if not chunks:
            logger.warning(f"No chunks found in index for doc_id={doc_id}")
            raise HTTPException(status_code=404, detail="No chunks found for this document")

        # 1) Question embedding
        q_emb = get_embedding(question)

        # 2) Score each chunk with cosine similarity
        for ch in chunks:
            ch["score"] = cosine_sim(q_emb, ch["embedding"])

        # 3) First stage: take top 10 by cosine
        top_10 = sorted(chunks, key=lambda x: x["score"], reverse=True)[:10]

        # 4) Second stage: rerank using LLM and keep best 3
        top_k_chunks = rerank_chunks(question, top_10, top_k=3)

        # ✅ SAFE: no KeyError on "id"
        used_chunk_ids = [
            ch.get("id")
            or ch.get("chunk_id")
            or ch.get("index")
            or f"{doc_id}_chunk_{i}"
            for i, ch in enumerate(top_k_chunks)
        ]

        # 5) Build final context from reranked chunks
        context = "\n\n---\n\n".join(ch["text"] for ch in top_k_chunks)
        prompt = f"""
You are an assistant answering based ONLY on the context below.
Context:
{context}
Question: {question}
If the answer is not clearly present in the context, reply with:
"I don't know based on the provided context."
"""
        answer = call_bedrock_text(prompt, max_tokens=300, temperature=0.3)

        logger.info(f"ask_rag success for doc_id={doc_id}, used_chunks={used_chunk_ids}")
        latency = (time.perf_counter() - start) * 1000
        log_metric("ask_rag_single", latency, {
            "doc_id": doc_id,
            "used_chunks": used_chunk_ids
        })

        return {
            "answer": answer,
            "used_chunks": used_chunk_ids
        }

    except HTTPException:
        raise
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        log_metric("ask_rag_single_error", latency, {
            "doc_id": doc_id,
            "error": str(e)
        })
        logger.exception(f"ask_rag failed for doc_id={doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

# ---------------- ENDPOINT: ask_rag_multi (multi-document RAG with reranking) ----------------
@app.get("/ask_rag_multi")
def ask_rag_multi(question: str):
    """
    Multi-document RAG:
    - Scan all index JSON files in S3 under INDEX_PREFIX
    - Embed question once
    - Compute cosine similarity against all chunks from all documents
    - Take top N chunks globally, rerank with LLM, and answer
    """
    start = time.perf_counter()
    logger.info(f"ask_rag_multi called with question='{question}'")

    try:
        # List all index JSON files
        resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=INDEX_PREFIX)
        index_keys = [
            obj["Key"] for obj in resp.get("Contents", [])
            if obj["Key"].lower().endswith(".json")
        ]

        if not index_keys:
            logger.warning("No index files found in S3 for multi-document RAG")
            raise HTTPException(status_code=404, detail="No indexed documents available")

        # Embed question once
        q_emb = get_embedding(question)

        all_chunks: List[Dict] = []

        # Load chunks from all index files
        for idx_key in index_keys:
            obj = s3.get_object(Bucket=BUCKET_NAME, Key=idx_key)
            index = json.loads(obj["Body"].read().decode("utf-8"))

            doc_id = index.get("doc_id", idx_key)
            chunks = index.get("chunks", [])

            for ch in chunks:
                # compute cosine similarity
                ch_score = cosine_sim(q_emb, ch["embedding"])
                all_chunks.append({
                    "id": ch["id"],
                    "text": ch["text"],
                    "embedding": ch["embedding"],
                    "doc_id": doc_id,
                    "score": ch_score
                })

        if not all_chunks:
            logger.warning("No chunks found across all documents")
            raise HTTPException(status_code=404, detail="No chunks found in any document")

        # Stage 1: top 20 by cosine globally
        top_20 = sorted(all_chunks, key=lambda x: x["score"], reverse=True)[:20]

        # Stage 2: rerank with LLM and keep best 5
        top_k_chunks = rerank_chunks(question, top_20, top_k=5)
        used_chunk_ids = [ch["id"] for ch in top_k_chunks]
        used_docs = list({ch["doc_id"] for ch in top_k_chunks})

        context = "\n\n---\n\n".join(
            f"[Document: {ch['doc_id']}]\n{ch['text']}"
            for ch in top_k_chunks
        )

        prompt = f"""
You are an assistant answering based ONLY on the context below, which may come from multiple documents.

Context:
{context}

Question: {question}

If the answer is not clearly present in the context, reply with:
"I don't know based on the provided context."
"""

        answer = call_bedrock_text(prompt, max_tokens=350, temperature=0.3)
        logger.info(
            f"ask_rag_multi success, used_docs={used_docs}, used_chunks={used_chunk_ids}"
        )

        latency = (time.perf_counter() - start) * 1000
        log_metric("ask_rag_multi", latency, {
            "used_docs": used_docs,
            "used_chunks": used_chunk_ids,
            "candidate_chunks": len(all_chunks)
        })

        return {
            "answer": answer,
            "used_documents": used_docs,
            "used_chunks": used_chunk_ids
        }

    except HTTPException:
        raise
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        log_metric("ask_rag_multi_error", latency, {"error": str(e)})
        logger.exception(f"ask_rag_multi failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-document RAG query failed: {str(e)}")


# ---------------- ENDPOINT: summary ----------------
@app.get("/summary")
def get_summary(doc_id: str):
    """
    Return the pre-computed summary from the index JSON.
    """
    start = time.perf_counter()
    try:
        index_key = f"{INDEX_PREFIX}{doc_id}.json"
        logger.info(f"Fetching summary from index: {index_key}")

        obj = s3.get_object(Bucket=BUCKET_NAME, Key=index_key)
        index = json.loads(obj["Body"].read().decode("utf-8"))

        latency = (time.perf_counter() - start) * 1000
        log_metric("summary", latency, {"doc_id": doc_id})

        return {
            "doc_id": doc_id,
            "summary": index.get("summary")
        }
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        log_metric("summary_error", latency, {"doc_id": doc_id, "error": str(e)})
        logger.exception(f"Failed to get summary for doc_id={doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch summary")


# ---------------- ENDPOINT: keywords ----------------
@app.get("/keywords")
def get_keywords(doc_id: str):
    """
    Return the pre-computed keywords from the index JSON.
    """
    start = time.perf_counter()
    try:
        index_key = f"{INDEX_PREFIX}{doc_id}.json"
        logger.info(f"Fetching keywords from index: {index_key}")

        obj = s3.get_object(Bucket=BUCKET_NAME, Key=index_key)
        index = json.loads(obj["Body"].read().decode("utf-8"))

        latency = (time.perf_counter() - start) * 1000
        log_metric("keywords", latency, {"doc_id": doc_id})

        return {
            "doc_id": doc_id,
            "keywords": index.get("keywords")
        }
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        log_metric("keywords_error", latency, {"doc_id": doc_id, "error": str(e)})
        logger.exception(f"Failed to get keywords for doc_id={doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch keywords")