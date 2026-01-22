import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Sentiment Analysis Dashboard | Amazon Reviews",
    page_icon="🛒",
    layout="wide"
)

# --- 2. FUNGSI LOAD DATA ---
@st.cache_data
def load_data():
    # Pastikan fail ini ada di dalam GitHub repository anda
    df = pd.read_csv("amazon_cleaned_with_sentiment.csv")
    return df

# Memuatkan data
try:
    df = load_data()
except FileNotFoundError:
    st.error("Ralat: Fail 'amazon_cleaned_with_sentiment.csv' tidak dijumpai. Sila muat naik fail ke GitHub.")
    st.stop()

# --- 3. SIDEBAR (NAVIGASI & UJIAN LIVE) ---
st.sidebar.image("https://www.vectorlogo.zone/logos/amazon/amazon-icon.svg", width=100)
st.sidebar.title("NLP Project Dashboard")
st.sidebar.markdown("---")

st.sidebar.subheader("🧪 Uji Sentimen Baru")
user_input = st.sidebar.text_area("Masukkan ulasan pelanggan di sini:")

if st.sidebar.button("Analisis Teks"):
    if user_input:
        analysis = TextBlob(user_input)
        polarity = analysis.sentiment.polarity
        if polarity > 0:
            st.sidebar.success(f"Sentimen: Positif (Skor: {polarity:.2f})")
        elif polarity == 0:
            st.sidebar.info(f"Sentimen: Neutral (Skor: {polarity:.2f})")
        else:
            st.sidebar.error(f"Sentimen: Negatif (Skor: {polarity:.2f})")
    else:
        st.sidebar.warning("Sila masukkan teks untuk dianalisis.")

st.sidebar.markdown("---")
st.sidebar.write("Dibuat oleh: [Nama Anda]")

# --- 4. BAHAGIAN UTAMA DASHBOARD ---
st.title("📊 Papan Pemuka Analisis Sentimen Amazon")
st.write("Projek ini menggunakan **Natural Language Processing (NLP)** untuk menganalisis emosi pelanggan terhadap produk Amazon.")

# Metrik Ringkasan
st.markdown("### Ringkasan Data")
col1, col2, col3, col4 = st.columns(4)

total_reviews = len(df)
pos_count = len(df[df['sentiment_result'] == 'Positif'])
neg_count = len(df[df['sentiment_result'] == 'Negatif'])
neu_count = len(df[df['sentiment_result'] == 'Neutral'])

col1.metric("Jumlah Ulasan", total_reviews)
col2.metric("Sentimen Positif", pos_count, f"{(pos_count/total_reviews)*100:.1f}%")
col3.metric("Sentimen Negatif", neg_count, f"-{(neg_count/total_reviews)*100:.1f}%", delta_color="inverse")
col4.metric("Sentimen Neutral", neu_count, f"{(neu_count/total_reviews)*100:.1f}%")

st.markdown("---")

# --- 5. VISUALISASI DATA ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 Taburan Sentimen")
    fig, ax = plt.subplots(figsize=(8, 6))
    sentiment_data = df['sentiment_result'].value_counts()
    sns.barplot(x=sentiment_data.index, y=sentiment_data.values, palette=['green', 'grey', 'red'], ax=ax)
    ax.set_xlabel("Kategori Sentimen")
    ax.set_ylabel("Bilangan Ulasan")
    st.pyplot(fig)

with col_right:
    st.subheader("☁️ Awan Kata (Word Cloud)")
    target_sentiment = st.selectbox("Pilih Sentimen:", ["Positif", "Negatif"])
    
    # Jana Word Cloud
    text_subset = " ".join(df[df['sentiment_result'] == target_sentiment]['cleaned_review'].astype(str))
    
    if text_subset:
        wc = WordCloud(width=800, height=500, background_color='white', colormap='viridis').generate(text_subset)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 6))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
    else:
        st.write("Tiada data untuk dipaparkan.")

st.markdown("---")

# --- 6. JADUAL DATA ---
st.subheader("📂 Paparan Data Ulasan")
st.write("Berikut adalah 50 baris pertama data yang telah diproses:")
st.dataframe(df[['product_name', 'review_content', 'sentiment_result']].head(50), use_container_width=True)

# --- 7. FOOTER ---
st.caption("Projek JIE43303: Natural Language Processing - Sentiment Analysis Dashboard")
