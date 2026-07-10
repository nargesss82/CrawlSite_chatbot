import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import numpy as np
import pickle
import os
import time
from collections import deque

 
st.set_page_config(
    page_title="وب RAG چت‌بات",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

#css
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;700&display=swap');

* { font-family: 'Vazirmatn', sans-serif; }

.stApp { background: #0f1117; color: #e2e8f0; }

.main-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.main-header h1 { color: #38bdf8; font-size: 2rem; margin: 0; }
.main-header p { color: #94a3b8; margin: 0.5rem 0 0; }

.chat-msg-user {
    background: #1e40af;
    border-radius: 18px 18px 4px 18px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    max-width: 80%;
    float: right;
    clear: both;
    color: #e0f2fe;
    direction: rtl;
}
.chat-msg-bot {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 18px 18px 18px 4px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    max-width: 85%;
    float: left;
    clear: both;
    color: #e2e8f0;
    direction: rtl;
}
.chat-container { overflow: hidden; padding: 1rem 0; }

.source-chip {
    display: inline-block;
    background: #0f172a;
    border: 1px solid #1e40af;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    color: #93c5fd;
    margin: 2px;
}

.stat-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-num { font-size: 1.8rem; font-weight: 700; color: #38bdf8; }
.stat-label { font-size: 0.8rem; color: #64748b; }

.status-ok { color: #34d399; }
.status-err { color: #f87171; }

div[data-testid="stButton"] button {
    border-radius: 10px;
    font-weight: 500;
    transition: all 0.2s;
}
</style>
""", unsafe_allow_html=True)


DB_FILE = "vector_db.pkl"
LM_STUDIO_URL = "http://localhost:1234/v1"
EMBED_MODEL = "nomic-embed-text"  

#helper function
def get_lm_models():
    try:
        r = requests.get(f"{LM_STUDIO_URL}/models", timeout=5)
        if r.status_code == 200:
            return [m["id"] for m in r.json().get("data", [])]
    except:
        pass
    return []

def lm_studio_chat(messages, model, temperature=0.7):
    try:
        r = requests.post(
            f"{LM_STUDIO_URL}/chat/completions",
            json={"model": model, "messages": messages, "temperature": temperature, "stream": False},
            timeout=60
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"خطا در ارتباط با LM Studio: {e}"
    return "پاسخی دریافت نشد."

def lm_studio_embed(text, model=EMBED_MODEL):
    try:
        r = requests.post(
            f"{LM_STUDIO_URL}/embeddings",
            json={"model": model, "input": text},
            timeout=30
        )
        if r.status_code == 200:
            return r.json()["data"][0]["embedding"]
    except:
        pass
    return None

def simple_embed(text):
    import hashlib
    words = text.lower().split()
    vec = np.zeros(512)
    for w in words:
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        idx = h % 512
        vec[idx] += 1
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm > 0 else vec.tolist()

def get_embedding(text):
    emb = lm_studio_embed(text)
    if emb:
        return emb
    return simple_embed(text)

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

#crawler
def crawl_website(base_url, max_pages=20, progress_bar=None, status_text=None):
    visited = set()
    to_visit = deque([base_url])
    pages = []
    base_domain = urlparse(base_url).netloc

    headers = {"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"}

    while to_visit and len(visited) < max_pages:
        url = to_visit.popleft()
        if url in visited:
            continue
        try:
            if status_text:
                status_text.text(f"در حال خواندن: {url[:70]}...")
            r = requests.get(url, headers=headers, timeout=10)
            if "text/html" not in r.headers.get("content-type", ""):
                continue
            soup = BeautifulSoup(r.text, "html.parser")

            #extract text
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            text = " ".join(text.split())

            if len(text) > 100:
                title = soup.title.string.strip() if soup.title else url
                pages.append({"url": url, "title": title, "text": text})

            visited.add(url)
            if progress_bar:
                progress_bar.progress(len(visited) / max_pages)

            #find link
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"])
                parsed = urlparse(href)
                if parsed.netloc == base_domain and href not in visited:
                    href_clean = parsed.scheme + "://" + parsed.netloc + parsed.path
                    to_visit.append(href_clean)

            time.sleep(0.3)
        except Exception as e:
            visited.add(url)
            continue

    return pages

#chunk
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

#database (vector)
def build_vector_db(pages, progress_bar=None, status_text=None):
    db = []
    total_chunks = sum(len(chunk_text(p["text"])) for p in pages)
    done = 0
    for page in pages:
        chunks = chunk_text(page["text"])
        for chunk in chunks:
            if status_text:
                status_text.text(f"امبد کردن: {page['title'][:50]}...")
            emb = get_embedding(chunk)
            db.append({
                "url": page["url"],
                "title": page["title"],
                "text": chunk,
                "embedding": emb
            })
            done += 1
            if progress_bar:
                progress_bar.progress(min(done / max(total_chunks, 1), 1.0))
    return db

def save_db(db):
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return []

def search_db(query, db, top_k=3):
    q_emb = get_embedding(query)
    scored = []
    for item in db:
        score = cosine_similarity(q_emb, item["embedding"])
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

#session state
if "vector_db" not in st.session_state:
    st.session_state.vector_db = load_db()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "crawled_site" not in st.session_state:
    st.session_state.crawled_site = ""

#ui: streamlit
st.markdown("""
<div class="main-header">
    <h1>وب RAG چت‌بات</h1>
    <p>سایت مورد نظرت رو کرال کن، امبد کن، بعد باهاش چت کن</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("###تنظیمات")

    # LM Studio status
    models = get_lm_models()
    if models:
        st.markdown(f'<span class="status-ok">LM Studio متصله</span>', unsafe_allow_html=True)
        selected_model = st.selectbox("مدل چت:", models)
    else:
        st.markdown('<span class="status-err">LM Studio آفلاینه</span>', unsafe_allow_html=True)
        st.caption("LM Studio رو روی پورت 1234 بالا بیار")
        selected_model = st.text_input("اسم مدل:", "local-model")

    lm_url = st.text_input("آدرس LM Studio:", LM_STUDIO_URL)

    st.divider()
    st.markdown("###آمار دیتابیس")
    db = st.session_state.vector_db
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{len(db)}</div><div class="stat-label">تکه متن</div></div>', unsafe_allow_html=True)
    with col2:
        unique_urls = len(set(d["url"] for d in db)) if db else 0
        st.markdown(f'<div class="stat-card"><div class="stat-num">{unique_urls}</div><div class="stat-label">صفحه</div></div>', unsafe_allow_html=True)

    if st.session_state.crawled_site:
        st.caption(f"سایت: {st.session_state.crawled_site}")

    st.divider()
    if st.button("پاک کردن دیتابیس", use_container_width=True):
        st.session_state.vector_db = []
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
        st.success("دیتابیس پاک شد!")

    if st.button("پاک کردن تاریخچه چت", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

tab1, tab2 = st.tabs(["کرال و امبد کردن", "چت‌بات"])

#tab1: crawl
with tab1:
    st.markdown("###کرال کردن سایت")

    col1, col2 = st.columns([3, 1])
    with col1:
        site_url = st.text_input("آدرس سایت:", placeholder="https://example.com", label_visibility="collapsed")
    with col2:
        max_pages = st.number_input("حداکثر صفحه:", min_value=1, max_value=100, value=10)

    if st.button("شروع کرال و امبد کردن", use_container_width=True, type="primary"):
        if not site_url:
            st.error("آدرس سایت رو وارد کن!")
        else:
            if not site_url.startswith("http"):
                site_url = "https://" + site_url

            with st.container():
                st.markdown("**مرحله ۱: کرال کردن صفحات**")
                p1 = st.progress(0)
                s1 = st.empty()

                pages = crawl_website(site_url, max_pages=max_pages, progress_bar=p1, status_text=s1)
                s1.text(f"{len(pages)} صفحه پیدا شد!")
                p1.progress(1.0)

                st.markdown("**مرحله ۲: امبد کردن متن‌ها**")
                p2 = st.progress(0)
                s2 = st.empty()

                new_db = build_vector_db(pages, progress_bar=p2, status_text=s2)
                s2.text(f"{len(new_db)} تکه متن امبد شد!")
                p2.progress(1.0)

                st.session_state.vector_db = new_db
                st.session_state.crawled_site = site_url
                save_db(new_db)

                st.success(f"تموم شد! {len(pages)} صفحه با {len(new_db)} تکه متن ذخیره شد.")

    if st.session_state.vector_db:
        with st.expander("صفحات ذخیره‌شده"):
            unique_pages = {}
            for item in st.session_state.vector_db:
                if item["url"] not in unique_pages:
                    unique_pages[item["url"]] = item["title"]
            for url, title in unique_pages.items():
                st.markdown(f"- **{title}** — [{url}]({url})")

#tab2: chatbot
with tab2:
    if not st.session_state.vector_db:
        st.info("ابتدا یه سایت رو از تب اول کرال کن!")
    else:
        # Chat history display
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-container"><div class="chat-msg-user">{msg["content"]}</div></div>', unsafe_allow_html=True)
                else:
                    content_html = msg["content"].replace("\n", "<br>")
                    sources_html = ""
                    if msg.get("sources"):
                        for s in msg["sources"]:
                            sources_html += f'<span class="source-chip">{s}</span>'
                    st.markdown(
                        f'<div class="chat-container"><div class="chat-msg-bot">{content_html}'
                        f'{"<br><br>" + sources_html if sources_html else ""}'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        # Input
        col1, col2 = st.columns([5, 1])
        with col1:
            user_q = st.text_input("سوالت رو بنویس:", placeholder="از محتوای سایت بپرس...", label_visibility="collapsed", key="user_input")
        with col2:
            send = st.button("ارسال", use_container_width=True, type="primary")

        top_k = st.slider("تعداد نتایج مرتبط:", 1, 5, 3)

        if send and user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})

            with st.spinner("در حال جستجو و تولید پاسخ."):
                # Search
                results = search_db(user_q, st.session_state.vector_db, top_k=top_k)

                # Build context
                context_parts = []
                sources = []
                for score, item in results:
                    context_parts.append(f"[منبع: {item['title']} | {item['url']}]\n{item['text']}")
                    short = item['url'].replace("https://", "").replace("http://", "")[:40]
                    if short not in sources:
                        sources.append(short)

                context = "\n\n---\n\n".join(context_parts)

                system_prompt = """تو یک دستیار هوشمند هستی که بر اساس محتوای یک وب‌سایت به سوالات پاسخ می‌دهی.
اطلاعات زیر از صفحات وب استخراج شده‌اند. بر اساس این اطلاعات، یک پاسخ جامع، دقیق و با جمله‌بندی خوب به فارسی بده.
اگر جواب در متن‌های داده شده نبود، صادقانه بگو که اطلاعاتی پیدا نکردی.
از اطلاعات خارج از متن‌های داده شده استفاده نکن."""

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"متن‌های مرتبط:\n\n{context}\n\n---\n\nسوال: {user_q}"}
                ]

                answer = lm_studio_chat(messages, selected_model)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
            st.rerun()
