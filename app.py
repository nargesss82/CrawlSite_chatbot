import html
import os
import pickle
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup


DB_FILE = "vector_db.pkl"
DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1"
EMBED_MODEL = "rodion-m/text-embedding-multilingual-e5-small"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"}

CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

st.set_page_config(
    page_title="وب RAG چت‌بات",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
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
    direction: ltr;   /* جواب‌ها انگلیسی‌ان */
    text-align: left;
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

div[data-testid="stButton"] button,
div[data-testid="stFormSubmitButton"] button {
    border-radius: 10px;
    font-weight: 500;
    transition: all 0.2s;
}
</style>
""", unsafe_allow_html=True)


# LM Studio
def get_lm_models(base_url):
    try:
        r = requests.get(f"{base_url}/models", timeout=5)
        r.raise_for_status()  # اگه استاتوس کد خطا باشه خودش خطارو برمیگردونه
        return [m["id"] for m in r.json().get("data", [])]
    except requests.RequestException:
        return []


def lm_studio_chat(base_url, messages, model, temperature=0.3, max_tokens=500):
    # temperature پایین‌تر برای RAG بهتره؛ جواب پایبندتر به متن میشه
    r = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def lm_studio_embed(base_url, text, model=EMBED_MODEL):
    try:
        r = requests.post(
            f"{base_url}/embeddings",
            json={"model": model, "input": text},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]
    except (requests.RequestException, KeyError, IndexError):
        return None


def get_embedding(base_url, text, is_query=False):
    prefix = "query: " if is_query else "passage: "
    return lm_studio_embed(base_url, prefix + text)


COMMENT_MARKERS = ("دیدگاه ها", "دیدگاه‌ها", "نظرات کاربران", "leave a reply", "comments")


def normalize_link(parsed):
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}"


# Crawl
def crawl_website(base_url, max_pages=20, progress_bar=None, status_text=None):  # BFS: breadth-first search
    visited = set()              # صفحاتی که پردازش شدن
    queued = {base_url}          # صفحاتی که دیده شدن و قبلا به صف اضافه شدن
    to_visit = deque([base_url])  # یه صف دوطرفه از صفحاتی که باید پردازش شن
    pages = []                   # لیست صفحات با عنوان صفحه و متنش
    base_domain = urlparse(base_url).netloc  # دامنه اصلی سایت

    while to_visit and len(visited) < max_pages:
        url = to_visit.popleft()
        if url in visited:
            continue
        try:
            if status_text:
                status_text.text(f"در حال خواندن: {url[:70]}...")
            r = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
            if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
                visited.add(url)  # صفحات خطادار (404 و...) پردازش نمیشن
                continue

            soup = BeautifulSoup(r.content, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            blocks = [
                " ".join(line.split())
                for line in soup.get_text(separator="\n", strip=True).split("\n")
                if line.strip()
            ]

            for i, b in enumerate(blocks):
                if b.lower() in COMMENT_MARKERS:
                    blocks = blocks[:i]
                    break

            text = " ".join(blocks)
            if len(text) > 100:
                title = soup.title.string.strip() if soup.title and soup.title.string else url
                pages.append({"url": url, "title": title, "blocks": blocks})

            visited.add(url)
            if progress_bar:
                progress_bar.progress(min(len(visited) / max_pages, 1.0))

            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"])
                parsed = urlparse(href)
                if parsed.netloc != base_domain:
                    continue
                if parsed.scheme not in ("http", "https"):
                    continue
                href_clean = normalize_link(parsed)
                if href_clean not in visited and href_clean not in queued:
                    queued.add(href_clean)
                    to_visit.append(href_clean)

            time.sleep(0.3)
        except requests.RequestException:
            visited.add(url)
            continue

    return pages


def separate_boilerplate(pages, threshold=0.5):

    if not pages:
        return []
    if len(pages) < 3:
        return [
            {"url": p["url"], "title": p["title"], "text": " ".join(p["blocks"])}
            for p in pages
        ]

    block_page_count = {}
    for p in pages:
        for b in set(p["blocks"]):
            block_page_count[b] = block_page_count.get(b, 0) + 1

    cutoff = max(2, int(len(pages) * threshold))
    boilerplate = {b for b, n in block_page_count.items() if n >= cutoff}

    result = []
    for p in pages:
        content = " ".join(b for b in p["blocks"] if b not in boilerplate)
        if len(content) > 100:
            result.append({"url": p["url"], "title": p["title"], "text": content})

    if boilerplate:
        first = pages[0]
        bp_text = " ".join(b for b in first["blocks"] if b in boilerplate)
        if len(bp_text) > 50:
            result.append({
                "url": first["url"],
                "title": "Site menu / header / footer (shared across pages)",
                "text": bp_text,
            })

    return result


# Chunking + Vector DB
def split_sentences(text):
    """متن رو به جمله تقسیم می‌کنه (علائم انگلیسی و فارسی).
    جمله‌های خیلی بلند (مثل لیست‌های بدون نقطه) به قطعات کوچیک‌تر شکسته میشن."""
    parts = re.split(r"(?<=[.!?؟۔…])\s+", text)
    sentences = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        words = p.split()
        if len(words) <= CHUNK_SIZE:
            sentences.append(p)
        else:
            for i in range(0, len(words), CHUNK_SIZE):
                sentences.append(" ".join(words[i:i + CHUNK_SIZE]))
    return sentences


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if chunk_size <= 0:
        return []

    sentences = split_sentences(text)
    chunks = []
    current, current_words = [], 0

    for sent in sentences:
        n = len(sent.split())
        if current and current_words + n > chunk_size:
            chunks.append(" ".join(current))
            kept, kept_words = [], 0
            for prev in reversed(current):
                pw = len(prev.split())
                if kept_words + pw > overlap:
                    break
                kept.insert(0, prev)
                kept_words += pw
            current, current_words = kept, kept_words
        current.append(sent)
        current_words += n

    if current and current_words >= 8:  
        chunks.append(" ".join(current))

    return chunks


def build_vector_db(base_url, pages, progress_bar=None, status_text=None):
    db = []
    all_chunks = []
    for p in pages:
        for c in chunk_text(p["text"]):
            all_chunks.append((p, c))
    total = max(len(all_chunks), 1)

    for done, (page, chunk) in enumerate(all_chunks, start=1):
        if status_text:
            status_text.text(f"امبد کردن: {page['title'][:50]}...")
        emb = get_embedding(base_url, chunk, is_query=False)
        if emb is None:
            continue  # این تکه‌متن امبد نشد؛ رد می‌شویم
        db.append({
            "url": page["url"],
            "title": page["title"],
            "text": chunk,
            "embedding": emb,
        })
        if progress_bar:
            progress_bar.progress(done / total)

    return db, len(all_chunks)


def save_db(db):
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)


def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return []


def search_db(base_url, query, db, top_k=3, min_score=0.0):
    """None برمی‌گردونه اگه embedding سوال ساخته نشد؛
    لیست خالی اگه هیچ چانکی از آستانه‌ی شباهت رد نشد."""
    q_emb = get_embedding(base_url, query, is_query=True)
    if q_emb is None:
        return None
    if not db:
        return []

    q = np.array(q_emb, dtype=np.float32)
    mat = np.array([item["embedding"] for item in db], dtype=np.float32)

    q_norm = np.linalg.norm(q)
    mat_norms = np.linalg.norm(mat, axis=1)
    denom = mat_norms * q_norm
    denom[denom == 0] = 1e-9
    scores = (mat @ q) / denom

    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), db[i]) for i in top_idx if scores[i] >= min_score]


# توابع کمکی برای UI
def render_user_message(content):
    safe = html.escape(content)
    st.markdown(
        f'<div class="chat-container"><div class="chat-msg-user">{safe}</div></div>',
        unsafe_allow_html=True,
    )


def render_bot_message(content, sources=None):
    content_html = html.escape(content).replace("\n", "<br>")
    sources_html = "".join(
        f'<span class="source-chip">{html.escape(s)}</span>' for s in (sources or [])
    )
    extra = f"<br><br>{sources_html}" if sources_html else ""
    st.markdown(
        f'<div class="chat-container"><div class="chat-msg-bot">{content_html}{extra}</div></div>',
        unsafe_allow_html=True,
    )


# مقداردهی اولیه‌ی session_state
if "vector_db" not in st.session_state:
    st.session_state.vector_db = load_db()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "crawled_site" not in st.session_state:
    st.session_state.crawled_site = ""
if "lm_studio_url" not in st.session_state:
    st.session_state.lm_studio_url = DEFAULT_LM_STUDIO_URL


# header
st.markdown("""
<div class="main-header">
    <h1>وب RAG چت‌بات</h1>
    <p>سایت مورد نظرت رو کرال کن، امبد کن، بعد باهاش چت کن</p>
</div>
""", unsafe_allow_html=True)


# sidebar
with st.sidebar:
    st.markdown("### تنظیمات")

    lm_url = st.text_input("آدرس LM Studio:", value=st.session_state.lm_studio_url)
    st.session_state.lm_studio_url = lm_url

    models = get_lm_models(lm_url)
    if models:
        st.markdown('<span class="status-ok">LM Studio متصله</span>', unsafe_allow_html=True)
        chat_models = [m for m in models if "embed" not in m.lower()] or models
        selected_model = st.selectbox("مدل چت:", chat_models)
        if EMBED_MODEL not in models:
            st.warning(f"مدل embedding لود نشده:\n{EMBED_MODEL}")
    else:
        st.markdown('<span class="status-err">LM Studio آفلاینه</span>', unsafe_allow_html=True)
        st.caption("LM Studio رو روی پورت 1234 بالا بیار")
        selected_model = st.text_input("اسم مدل:", "local-model")

    st.divider()
    st.markdown("### آمار دیتابیس")
    db = st.session_state.vector_db
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{len(db)}</div>'
            f'<div class="stat-label">تکه متن</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        unique_urls = len({d["url"] for d in db}) if db else 0
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{unique_urls}</div>'
            f'<div class="stat-label">صفحه</div></div>',
            unsafe_allow_html=True,
        )

    if st.session_state.crawled_site:
        st.caption(f"سایت: {st.session_state.crawled_site}")

    st.divider()
    if st.button("پاک کردن دیتابیس", use_container_width=True):
        st.session_state.vector_db = []
        st.session_state.crawled_site = ""
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
        st.success("دیتابیس پاک شد!")

    if st.button("پاک کردن تاریخچه چت", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# Tabs
tab1, tab2 = st.tabs(["کرال و امبد کردن", "چت‌بات"])

# tab 1: کرال و امبد کردن
with tab1:
    st.markdown("### کرال کردن سایت")

    col1, col2 = st.columns([3, 1])
    with col1:
        site_url = st.text_input(
            "آدرس سایت:", placeholder="https://example.com", label_visibility="collapsed"
        )
    with col2:
        max_pages = st.number_input("حداکثر صفحه:", min_value=1, max_value=100, value=10)

    if st.button("شروع کرال و امبد کردن", use_container_width=True, type="primary"):
        if not site_url:
            st.error("آدرس سایت رو وارد کن!")
        else:
            if not site_url.startswith("http"):
                site_url = "https://" + site_url

            st.markdown("**مرحله ۱: کرال کردن صفحات**")
            p1, s1 = st.progress(0), st.empty()
            pages = crawl_website(site_url, max_pages=max_pages, progress_bar=p1, status_text=s1)
            s1.text(f"{len(pages)} صفحه پیدا شد!")
            p1.progress(1.0)

            if not pages:
                st.warning("هیچ صفحه‌ای پیدا نشد. آدرس یا دسترسی به سایت رو چک کن.")
            else:
                pages = separate_boilerplate(pages)

                st.markdown("**مرحله ۲: امبد کردن متن‌ها**")
                p2, s2 = st.progress(0), st.empty()
                new_db, expected_chunks = build_vector_db(
                    st.session_state.lm_studio_url, pages, progress_bar=p2, status_text=s2
                )
                if len(new_db) < expected_chunks:
                    st.warning(
                        "بعضی از تکه‌متن‌ها امبد نشدند. مطمئن شو مدل embedding روی LM Studio لود شده."
                    )
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
                unique_pages.setdefault(item["url"], item["title"])
            for url, title in unique_pages.items():
                st.markdown(f"- **{title}** — [{url}]({url})")

# tab 2: چت‌بات
with tab2:
    if not st.session_state.vector_db:
        st.info("ابتدا یه سایت رو از تب اول کرال کن!")
    else:
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    render_user_message(msg["content"])
                else:
                    render_bot_message(msg["content"], msg.get("sources"))

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            top_k = st.slider("تعداد نتایج مرتبط:", 1, 5, 3)
        with col_b:
            min_sim = st.slider(
                "حداقل شباهت:", 0.50, 0.90, 0.75, 0.01,
                help="چانک‌هایی با شباهت کمتر از این مقدار به مدل داده نمیشن",
            )

        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                user_q = st.text_input(
                    "سوالت رو بنویس:",
                    placeholder="Ask about the site content...",
                    label_visibility="collapsed",
                )
            with col2:
                send = st.form_submit_button("ارسال", use_container_width=True, type="primary")

        if send and user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})

            with st.spinner("در حال جستجو و تولید پاسخ..."):
                results = search_db(
                    st.session_state.lm_studio_url,
                    user_q,
                    st.session_state.vector_db,
                    top_k=top_k,
                    min_score=min_sim,
                )

                if results is None:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": (
                            "Could not create an embedding for this question. "
                            f"Make sure LM Studio is running and the embedding model "
                            f"({EMBED_MODEL}) is loaded."
                        ),
                        "sources": [],
                    })
                elif not results:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": (
                            "I couldn't find anything relevant to your question on this "
                            "website. Try rephrasing, or lower the similarity threshold."
                        ),
                        "sources": [],
                    })
                else:
                    context_parts, sources = [], []
                    for score, item in results:
                        context_parts.append(
                            f"[Source: {item['title']} | {item['url']}]\n{item['text']}"
                        )
                        short = item["url"].replace("https://", "").replace("http://", "")[:40]
                        if short not in sources:
                            sources.append(short)

                    context = "\n\n---\n\n".join(context_parts)

                    system_prompt = (
                        "You are an assistant that answers questions ONLY based on the text "
                        "excerpts below, which were extracted from a website. "
                        "All excerpts and questions are in English.\n"
                        "Important rules:\n"
                        "1. Only use text that is directly relevant to the question. "
                        "Ignore irrelevant excerpts completely.\n"
                        "2. If none of the excerpts answer the question, say clearly: "
                        "'The answer to this question was not found in the available text.' "
                        "Do not guess, infer, or construct an answer from loosely related "
                        "information.\n"
                        "3. Do not add any information from general knowledge, training data, "
                        "or your own assumptions — every claim in your answer must be traceable "
                        "to the provided excerpts.\n"
                        "4. Do not pad the answer with unrelated context just to sound "
                        "comprehensive. If the excerpts only partially answer the question, "
                        "say so explicitly and state what is missing.\n"
                        "5. LANGUAGE RULE (strict, mandatory): Your answer MUST be written "
                        "entirely in English, no matter what language appears anywhere in the "
                        "conversation."
                    )
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"Relevant excerpts:\n\n{context}\n\n---\n\nQuestion: {user_q}",
                        },
                    ]

                    try:
                        answer = lm_studio_chat(
                            st.session_state.lm_studio_url, messages, selected_model
                        )
                    except (requests.RequestException, KeyError, IndexError):
                        best_score, best = results[0]
                        answer = (
                            "(Chat model unavailable — showing the most relevant excerpt instead)\n\n"
                            f"{best['text']}\n\n"
                            f"Source: {best['title']} ({best['url']})"
                        )

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "debug": [
                            {"score": round(score, 3), "url": item["url"], "preview": item["text"][:250]}
                            for score, item in results
                        ],
                    })
            st.rerun()

        if st.session_state.chat_history:
            last = st.session_state.chat_history[-1]
            if last["role"] == "assistant" and last.get("debug"):
                with st.expander("چانک‌های بازیابی‌شده (دیباگ)"):
                    for d in last["debug"]:
                        st.markdown(f"**score: {d['score']}** — {d['url']}")
                        st.text(d["preview"])


# run: pip install -r requirements.txt , streamlit run app.py