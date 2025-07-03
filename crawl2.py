import os
import json
import logging
import time
import requests
import re
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL1")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET1")
DEEPSEEK_API_KEY = os.getenv("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

DEEPSEEK_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def get_urls_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def clean_cdi_url(url):
    return url[:-5] if url.endswith("/json") else url

def cdi_json_to_text(json_obj):
    out = []
    def add_field(label, value):
        if isinstance(value, list):
            value = "; ".join(str(x) for x in value)
        elif isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False)
        out.append(f"{label}: {value}")
    for main_key, sub_dict in json_obj.items():
        if isinstance(sub_dict, dict):
            for k, v in sub_dict.items():
                add_field(f"{main_key} - {k}", v)
        else:
            add_field(main_key, sub_dict)
    return "\n".join(out)

def parse_llm_output(content: str):
    """More robust parsing of Title and Summary output from LLM"""
    title, summary = "", ""
    lines = content.strip().splitlines()
    for line in lines:
        if line.lower().startswith("title:") and not title:
            title = line.split(":", 1)[1].strip()
        elif line.lower().startswith("summary:") and not summary:
            summary = line.split(":", 1)[1].strip()
    if not title and not summary:
        # fallback 正则
        m = re.search(r"Title:\s*(.+?)\s*Summary:\s*(.+)", content, re.DOTALL)
        if m:
            title = m.group(1).strip()
            summary = m.group(2).strip()
    return title or "Unknown Title", summary or content.strip()

def get_llm_title_summary(text):
    prompt = (
        "Given the following dataset metadata, generate:\n"
        "1. A concise, informative title (max 1 sentence).\n"
        "2. A concise summary of the dataset in 1-2 sentences, focusing on key facts (theme, region, time, data type, etc).\n\n"
        "Metadata:\n"
        f"{text}\n\n"
        "Title:\nSummary:"
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
        "max_tokens": 128,
        "temperature": 0.3
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return parse_llm_output(content)
    except Exception as e:
        logging.error(f"DeepSeek LLM API error: {e}")
        return "Unknown Title", "No summary found."

def get_embedding(text):
    try:
        return embedding_model.encode(text).tolist()
    except Exception as e:
        logging.error(f"Embedding error: {e}")
        return [0.0] * 768

def insert_site_page(data):
    try:
        res = supabase.table("site_pages").insert(data).execute()
        logging.info(f"Inserted {data['url']}")
        return res
    except Exception as e:
        logging.error(f"Insert failed for {data['url']}: {e}")
        return None

def extract_metadata_fields(json_obj, url_clean: str):
    """Extract the concise metadata fields required for storage"""
    meta = json_obj.get("metadata", {})
    return {
        "source": "edmed_docs",
        "country": meta.get("Country", ""),
        "organization": meta.get("Originator", ""),
        "url_path": url_clean
    }

def process_and_store_cdi_url(url):
    url_clean = clean_cdi_url(url)
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        if resp.status_code != 200:
            logging.warning(f"Failed {url} status={resp.status_code}")
            return
        json_obj = resp.json()
        text_for_llm = cdi_json_to_text(json_obj)
        title, summary = get_llm_title_summary(text_for_llm)
        embedding = get_embedding(summary)
        row = {
            "url": url_clean,
            "chunk_number": 0,
            "title": title,
            "summary": summary,
            "content": json.dumps(json_obj, ensure_ascii=False, indent=2),
            "metadata": extract_metadata_fields(json_obj, url_clean),
            "embedding": embedding
        }
        insert_site_page(row)
    except Exception as e:
        logging.warning(f"Exception for {url}: {e}")

if __name__ == "__main__":
    urls = get_urls_from_file("cdi_urls.txt")
    logging.info(f"Total: {len(urls)} urls.")
    for url in urls:
        process_and_store_cdi_url(url)
        time.sleep(1.0)
