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
import logging
import requests
import re


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL1")
supabase_secret = os.getenv("SUPABASE_SECRET1")
supabase: Client = create_client(supabase_url, supabase_secret)

embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

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

def chunk_text(text: str) -> List[str]:
    return [text.strip()]

def extract_title_and_summary_from_icos(text: str) -> dict:
    m_title = re.search(r"# (ICOS .+)", text)
    title = m_title.group(1).strip() if m_title else "Unknown Title"

    m_summary = re.search(r"(?i)^Description\s+(.+?)\n", text, re.MULTILINE)
    summary = m_summary.group(1).strip() if m_summary else "No summary found."

    return {"title": title, "summary": summary}

def refine_summary_with_llm(summary: str) -> str:
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
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"].strip()
        else:
            return summary
    except Exception as e:
        logging.error(f"DeepSeek LLM API error: {e}")
        return summary

def get_embedding(text: str) -> List[float]:
    try:
        return embedding_model.encode(text).tolist()
    except Exception as e:
        logging.error(f"Embedding error: {e}")
        return [0.0] * 768

def extract_main_content_from_icos(markdown: str) -> str:
    """
    Extract the main content from the original ICOS markdown, including key fields such as title, description, time, site, etc., and filter out noise.
    """
    lines = markdown.splitlines()
    keep_lines = []
    capture = False

    for i, line in enumerate(lines):
        if line.strip().startswith("# ICOS"):
            keep_lines.append(line.strip())
            capture = True
            continue

        if "## We use cookies" in line or "### Central Facility websites" in line:
            break

        if capture:
            if re.search(r"\[(Cart|Log in|My Account|Contact|Privacy|Terms)\]", line):
                continue
            if re.search(r"\!\[.*?\]\(.*?\)", line):  # 跳过 logo 图片
                continue
            if re.search(r"^ *[*\-] ", line):  # 跳过纯列表导航
                continue
            if re.search(r"^\s*$", line):  # 跳过空行
                continue

            keep_lines.append(line.strip())

    return "\n".join(keep_lines).strip()


def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    main_content = extract_main_content_from_icos(chunk)
    extracted = extract_title_and_summary_from_icos(main_content)
    refined_summary = refine_summary_with_llm(extracted['summary'])
    embedding = get_embedding(main_content)
    metadata = {
        "source": "icos_docs",
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
    chunks = chunk_text(markdown)
    for i, chunk in enumerate(chunks):
        processed_chunk = process_chunk(chunk, i, url)
        insert_chunk(processed_chunk)

async def crawl_parallel(urls: List[str], max_concurrent: int = 3):
    browser_config = BrowserConfig(headless=True)
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
                result = await crawler.arun(url=url, config=crawl_config, session_id="icos_session")
                if result.success:
                    logging.info(f"Successfully crawled: {url}")
                    process_and_store_document(url, result.markdown.raw_markdown)
                else:
                    logging.warning(f"Failed: {url} - Error: {result.error_message}")
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_icos_urls_from_file(file_path="icos_urls.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

async def main():
    urls = get_icos_urls_from_file("icos_urls.txt")
    if not urls:
        logging.error("No URLs found to crawl")
        return
    logging.info(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())
