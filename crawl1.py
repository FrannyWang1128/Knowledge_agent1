import os
import json
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from supabase import create_client, Client

from sentence_transformers import SentenceTransformer
import torch
import re
import logging
import requests

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

load_dotenv()

# === Supabase configuration ===
supabase_url = os.getenv("SUPABASE_URL1")
supabase_secret = os.getenv("SUPABASE_SECRET1")
supabase: Client = create_client(supabase_url, supabase_secret)

# === BGE-embedding model loading ===
embedding_model_name = "BAAI/bge-base-en-v1.5"
embedding_model = SentenceTransformer(embedding_model_name)

# === DeepSeek LLM API configuration ===
DEEPSEEK_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("OPENAI_API_KEY")

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 4000) -> List[str]:
    """Treat each EDMED page as a single chunk."""
    return [text.strip()]

def extract_title_and_summary_from_edmed(text: str) -> dict:
    """
    Extract the dataset title and summary from an EDMED page in a structured way.

    Returns a dictionary with 'title' and 'summary'.
    """

    m_title = re.search(
        r"Data set name\|[\s\r\n]*#?\s*([^\n\r#]+)", text
    ) or re.search(
        r"Data set name[\s\r\n]+#?\s*([^\n\r#]+)", text
    )
    title = m_title.group(1).strip() if m_title else ""

    m_summary = re.search(r"Summary\|([^\n\r]+)", text)
    summary = m_summary.group(1).strip() if m_summary else ""

    if not title:
        title = "Unknown Title"
    if not summary:
        summary = "No summary found."
    return {"title": title, "summary": summary}

def refine_summary_with_llm(summary: str) -> str:
    """
    用 DeepSeek LLM API 精炼 summary，保证简洁。
    """
    prompt = (
        "Please rewrite the following description as a concise summary, focusing on the key data and facts, within 1-2 sentences:\n"
        f"{summary}\n\nSummary:"
    )
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-r1-distill-llama-8b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 64,
        "temperature": 0.3
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        # 兼容 API 格式
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"].strip()
        else:
            return summary  # fallback
    except Exception as e:
        logging.error(f"DeepSeek LLM API error: {e}")
        return summary  # fallback

def get_embedding(text: str) -> List[float]:
    """
    Get the embedding vector for the input text using the BGE model.
    """
    try:
        return embedding_model.encode(text).tolist()
    except Exception as e:
        logging.error(f"Embedding error: {e}")
        return [0.0]*768

def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """
        Extracts title and summary from the chunk, refines the summary, computes the embedding,
        and returns a structured ProcessedChunk.
        """

    extracted = extract_title_and_summary_from_edmed(chunk)
    refined_summary = refine_summary_with_llm(extracted['summary'])
    embedding = get_embedding(chunk)
    metadata = {
        "source": "edmed_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=refined_summary,
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )

def insert_chunk(chunk: ProcessedChunk):
    """
        Insert a ProcessedChunk into the Supabase table 'site_pages'.
        """
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        result = supabase.table("site_pages").insert(data).execute()
        logging.info(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        logging.error(f"Error inserting chunk: {e}")
        return None

def process_and_store_document(url: str, markdown: str):
    """
        Process the markdown content for a document (URL), split it into chunks,
        process each chunk, and insert into Supabase.
        """
    chunks = chunk_text(markdown)
    for i, chunk in enumerate(chunks):
        processed_chunk = process_chunk(chunk, i, url)
        insert_chunk(processed_chunk)

async def crawl_parallel(urls: List[str], max_concurrent: int = 3):
    """
    Crawl multiple URLs in parallel, extract and process content, and store the results in Supabase.
    """
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.6),
            options={"ignore_links": True}
        ),
        cache_mode=CacheMode.BYPASS
    )
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    try:
        semaphore = asyncio.Semaphore(max_concurrent)
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    logging.info(f"Successfully crawled: {url}")
                    process_and_store_document(url, result.markdown.raw_markdown)
                else:
                    logging.warning(f"Failed: {url} - Error: {result.error_message}")
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_edmed_urls_from_file(file_path="edmed_urls.txt"):
    """
    Read EDMED URLs from a local file, one per line.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls

async def main():
    urls = get_edmed_urls_from_file("edmed_urls.txt")
    if not urls:
        logging.error("No URLs found to crawl")
        return
    logging.info(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())
