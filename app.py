import streamlit as st
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import markdown
import os
from typing import List, Tuple
import tempfile
import torch

# تنظیم کلید API برای OpenAI و Aval AI
openai.api_key = "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"  # کلید API برای چت
try:
    avalai_client = openai.OpenAI(
        base_url="https://api.avalai.ir/v1",
        api_key="aa-p1CZqsHEt317KY2ZLBZTN0atr5MlMUNh7yytaOEAzbvtHEfS"  # کلید API برای Aval AI
    )
except AttributeError:
    st.error("کتابخانه openai از ساختار OpenAI پشتیبانی نمی‌کند. لطفاً نسخه openai را به‌روزرسانی کنید یا مستندات Aval AI را بررسی کنید.")
    st.stop()

# بارگذاری مدل embedding چندزبانه روی CPU
try:
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device='cpu')
except Exception as e:
    st.error(f"خطا در بارگذاری مدل SentenceTransformer: {str(e)}")
    st.stop()

# خواندن فایل مارک‌داون
def read_markdown_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# تبدیل مارک‌داون به متن ساده
def markdown_to_text(md_content: str) -> str:
    return markdown.markdown(md_content)

# تقسیم متن به بخش‌های کوچک‌تر
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_length += len(word) + 1
        current_chunk.append(word)
        if current_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# محاسبه بردارهای متنی
def get_embeddings(texts: List[str]) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True)

# جست‌وجوی مرتبط‌ترین بخش‌ها
def retrieve_relevant_chunks(query: str, chunks: List[str], embeddings: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
    query_embedding = model.encode([query])[0]
    similarities = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(chunks[i], similarities[i]) for i in top_indices]

# تولید پاسخ با API چت جی‌پی‌تی
def generate_response(query: str, relevant_chunks: List[Tuple[str, float]]) -> str:
    context = "\n\n".join([chunk for chunk, _ in relevant_chunks])
    prompt = f"""
    شما یک دستیار پزشکی تخصصی هستید که فقط بر اساس پروتکل‌های اورژانس موجود در فایلの下 پاسخ می‌دهید. پاسخ‌ها باید به زبان فارسی، دقیق، مختصر و کاملاً منطبق با محتوای پروتکل‌ها باشند. از افزودن اطلاعات خارجی یا فرضیات خودداری کنید. اگر سوال کاربر مبهم است، مرتبط‌ترین بخش‌های پروتکل را شناسایی کرده و پاسخ دهید. در صورت عدم وجود اطلاعات مرتبط در پروتکل‌ها، به صراحت اعلام کنید که اطلاعات کافی در فایل وجود ندارد.

    **متن پروتکل‌ها:**
    {context}

    **سوال کاربر:**
    {query}

    **دستورالعمل:**
    1. پاسخ را به زبان فارسی و به‌صورت ساختاریافته ارائه دهید.
    2. بخش‌های مرتبط پروتکل را به‌طور خلاصه ذکر کنید.
    3. اگر پروتکل خاصی مستقیماً به سوال اشاره دارد، آن را اولویت دهید.
    4. پاسخ باید حداکثر ۵۰۰ کلمه باشد، مگر اینکه سوال نیاز به جزئیات بیشتری داشته باشد.
    5. در صورت نیاز به اقدامات دارویی، دوزها و شرایط تجویز را دقیقاً از پروتکل‌ها استخراج کنید.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "شما یک دستیار پزشکی هستید که فقط بر اساس متن ارائه‌شده پاسخ می‌دهد."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"خطا در تولید پاسخ: {str(e)}")
        return "خطا در تولید پاسخ. لطفاً دوباره تلاش کنید."

# تولید فایل صوتی از پاسخ
def generate_speech(text: str) -> str:
    try:
        # ایجاد فایل موقت برای ذخیره صوت
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            speech_file_path = temp_file.name
        
        # تولید صوت با API Aval AI
        response = avalai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        response.stream_to_file(speech_file_path)
        return speech_file_path
    except Exception as e:
        st.error(f"خطا در تولید فایل صوتی: {str(e)}")
        return None

# رابط کاربری Streamlit
st.title("سیستم پرس‌وجو پروتکل‌های اورژانس")
st.markdown("سوال خود را درباره پروتکل‌های اورژانس وارد کنید تا پاسخ دقیق دریافت کنید.")

# خواندن و پردازش فایل مارک‌داون
markdown_file = r"C:\Users\techp\OneDrive\Desktop\New folder\RAG_sys\protocol.md"
if not os.path.exists(markdown_file):
    st.error("فایل protocol.md یافت نشد. لطفاً مسیر فایل را بررسی کنید.")
else:
    markdown_content = read_markdown_file(markdown_file)
    text_content = markdown_to_text(markdown_content)
    chunks = chunk_text(text_content)
    embeddings = get_embeddings(chunks)

    # دریافت سوال کاربر
    query = st.text_input("سوال خود را وارد کنید:", placeholder="مثال: در صورت ایست قلبی چه باید کرد؟")
    if query:
        # بازیابی بخش‌های مرتبط
        relevant_chunks = retrieve_relevant_chunks(query, chunks, embeddings)
        # تولید پاسخ
        response = generate_response(query, relevant_chunks)
        st.markdown("### پاسخ:")
        st.write(response)

 
        if st.button("پخش صوتی پاسخ"):
            speech_file = generate_speech(response)
            if speech_file:
                st.audio(speech_file, format="audio/mp3")
                # حذف فایل موقت پس از پخش
                try:
                    os.remove(speech_file)
                except:
                    pass
        
        # نمایش بخش‌های بازیابی‌شده (اختیاری برای دیباگ)
        with st.expander("بخش‌های مرتبط از پروتکل‌ها"):
            for chunk, score in relevant_chunks:
                st.markdown(f"**امتیاز مشابهت: {score:.2f}**\n\n{chunk}\n\n---")