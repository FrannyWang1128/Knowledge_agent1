# streamlit_app.py
import os, asyncio, re
from dotenv import load_dotenv
import streamlit as st
from supabase import Client
from openai import AsyncOpenAI

from pydantic_ai_expert import (
    pydantic_ai_expert,
    PydanticAIDeps,
)

load_dotenv()

# —— Backend dependencies initialization ——
supabase = Client(os.getenv("SUPABASE_URL1"), os.getenv("SUPABASE_SECRET1"))
openai_client = AsyncOpenAI(
    api_key  = os.getenv("OPENAI_API_KEY"),
    base_url = os.getenv("OPENAI_API_BASE"),
)
deps = PydanticAIDeps(supabase=supabase, openai_client=openai_client)

# —— Streamlit page configuration ——
st.set_page_config(page_title="🌍 Env Sci RAG", layout="wide")
st.title("🔎 Environmental Data Retrieval Agent")

# —— Session & dataset state initialization ——
if "history" not in st.session_state:
    st.session_state.history = []    # List of chat history: [{"role":"user"|"assistant","content":...}, ...]
if "docs" not in st.session_state:
    st.session_state.docs = []       # List of picked docs: [{"title","url"}, ...]

left, right = st.columns([1,3])  # Split layout: left for docs, right for chat

# —— Right panel: multi-turn chat & user input ——
with right:
    # 1) Render chat history (all previous messages)
    for msg in st.session_state.history:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # 2) Chat input box for user question
    user_input = st.chat_input("Enter your question…")
    if user_input:
        # 3) Render user message immediately
        st.session_state.history.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        # 4) Show assistant "typing..." bubble and spinner while thinking
        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Thinking…"):
                run_res = asyncio.run(pydantic_ai_expert.run(user_input, deps=deps))
            answer = run_res.output
            # Replace placeholder with real answer
            placeholder.markdown(answer)

        # 5) Save assistant response to history
        st.session_state.history.append({"role": "assistant", "content": answer})

        # 6) From deps.last_docs, pick out docs corresponding to indices in LLM's answer
        raw_docs = getattr(deps, "last_docs", [])
        picked = []
        # Match lines like "1. Dataset Title – …" and capture index and title (title ends before the first dash)
        for num, title in re.findall(r"^\s*(\d+)\.\s*([^-]+)", answer, flags=re.M):
            idx = int(num) - 1
            if 0 <= idx < len(raw_docs):
                doc = raw_docs[idx]
                # Use the title from LLM output (strip any extra spaces)
                picked.append({
                    "title": title.strip(),
                    "url": doc["url"]
                })
        st.session_state.docs = picked

# —— Left panel: Hit datasets list ——
with left:
    st.header("📑 Hit Datasets")
    if not st.session_state.docs:
        st.write("_No datasets yet_")
    else:
        for d in st.session_state.docs:
            st.markdown(f"- [{d['title']}]({d['url']})")
