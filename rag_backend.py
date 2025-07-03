# rag_backend.py

from dataclasses import dataclass
from dotenv import load_dotenv
import os
from typing import List, Any
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from supabase import Client

# ============ 环境变量 ============ #
load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SECRET")
supabase = Client(supabase_url, supabase_key)

# ============ 本地模型全局加载 ============ #
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name, device_map="auto", torch_dtype="auto"
)
hf_pipeline = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, do_sample=False
)

@dataclass
class PydanticAIDeps:
    supabase: Client
    hf_pipeline: Any

deps = PydanticAIDeps(supabase=supabase, hf_pipeline=hf_pipeline)

async def get_embedding(text: str) -> List[float]:
    try:
        return embedding_model.encode(text).tolist()
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0.0] * 768

def local_llama_chat(prompt: str) -> str:
    output = hf_pipeline(prompt)[0]["generated_text"]
    return output[len(prompt):].strip() if output.startswith(prompt) else output.strip()

async def retrieve_relevant_documentation(deps: PydanticAIDeps, user_query: str) -> str:
    query_embedding = await get_embedding(user_query)
    result = deps.supabase.rpc(
        'match_site_pages',
        {
            'query_embedding': query_embedding,
            'match_count': 5,
            'filter': {'source': 'pydantic_ai_docs'}
        }
    ).execute()
    if not result.data:
        return "No relevant documentation found."
    context = "\n\n---\n\n".join(
        [f"# {doc['title']}\n{doc['content']}" for doc in result.data]
    )
    prompt = f"""You are a helpful documentation assistant.
User question: {user_query}
Relevant documentation:
{context}
Based on the above documentation, answer the user's question concisely. If not found, say "not found."
"""
    return local_llama_chat(prompt)

async def list_documentation_pages(deps: PydanticAIDeps) -> List[str]:
    result = deps.supabase.from_('site_pages').select('url').eq('metadata->>source', 'pydantic_ai_docs').execute()
    if not result.data:
        return []
    return sorted(set(doc['url'] for doc in result.data))

async def get_page_content(deps: PydanticAIDeps, url: str) -> str:
    result = deps.supabase.from_('site_pages') \
        .select('title, content, chunk_number') \
        .eq('url', url) \
        .eq('metadata->>source', 'pydantic_ai_docs') \
        .order('chunk_number') \
        .execute()
    if not result.data:
        return f"No content found for URL: {url}"
    page_title = result.data[0]['title'].split(' - ')[0]
    formatted_content = [f"# {page_title}\n"]
    for chunk in result.data:
        formatted_content.append(chunk['content'])
    return "\n\n".join(formatted_content)
