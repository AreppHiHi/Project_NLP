import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sentiment Analysis Dashboard | Amazon Reviews",
    page_icon="🛒",
    layout="wide"
)

# --- 2. DATA LOADING FUNCTION ---
@st.cache_data
def load_data():
    # Ensure this file is present in your GitHub repository
    df = pd.read_csv("amazon_cleaned_with_sentiment.csv")
    return df

# Load the data
try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: The file 'amazon_cleaned_with_sentiment.csv' was not found. Please upload the file to GitHub.")
    st.stop()

# --- 3. SIDEBAR (NAVIGATION & LIVE TEST) ---
st.sidebar.image("https://www.vectorlogo.zone/logos/amazon/amazon-icon.svg", width=100)
st.sidebar.title("NLP Project Dashboard")
st.sidebar.markdown("---")

st.sidebar.subheader("🧪 Live Sentiment Test")
user_input = st.sidebar.text_area("Enter a customer review here:")

if st.sidebar.button("Analyze Text"):
    if user_input:
        analysis = TextBlob(user_input)
        polarity = analysis.sentiment.polarity
        if polarity > 0:
            st.sidebar.success(f"Sentiment: Positive (Score: {polarity:.2f})")
        elif polarity == 0:
            st.sidebar.info(f"Sentiment: Neutral (Score: {polarity:.2f})")
        else:
            st.sidebar.error(f"Sentiment: Negative (Score: {polarity:.2f})")
    else:
        st.sidebar.warning("Please enter some text to analyze.")

st.sidebar.markdown("---")
st.sidebar.write("Developed by: [Your Name]")

# --- 4. MAIN DASHBOARD SECTION ---
st.title("📊 Amazon Sentiment Analysis Dashboard")
st.write("This project utilizes **Natural Language Processing (NLP)** to analyze customer emotions regarding Amazon products.")

# Summary Metrics
st.markdown("### Data Summary")
col1, col2, col3, col4 = st.columns(4)

total_reviews = len(df)

# IMPORTANT: Ensure these labels match exactly with your CSV file labels
# If your CSV uses 'Positif', change 'Positive' to 'Positif' below.
pos_count = len(df[df['sentiment_result'].str.contains('Posit', case=False)])
neg_count = len(df[df['sentiment_result'].str.contains('Negat', case=False)])
neu_count = len(df[df['sentiment_result'].str.contains('Neutr', case=False)])

col1.metric("Total Reviews", total_reviews)
col2.metric("Positive Sentiment", pos_count, f"{(pos_count/total_reviews)*100:.1f}%")
col3.metric("Negative Sentiment", neg_count, f"-{(neg_count/total_reviews)*100:.1f}%", delta_color="inverse")
col4.metric("Neutral Sentiment", neu_count, f"{(neu_count/total_reviews)*100:.1f}%")

st.markdown("---")

# --- 5. DATA VISUALIZATION ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 Sentiment Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sentiment_data = df['sentiment_result'].value_counts()
    
    # Updated Seaborn barplot to prevent ValueError in newer versions
    sns.barplot(
        x=sentiment_data.index, 
        y=sentiment_data.values, 
        hue=sentiment_data.index, 
        palette=['green', 'grey', 'red'], 
        legend=False, 
        ax=ax
    )
    
    ax.set_xlabel("Sentiment Category")
    ax.set_ylabel("Number of Reviews")
    st.pyplot(fig)

with col_right:
    st.subheader("☁️ Word Cloud")
    # Change "Positive" to "Positif" if that is what is in your CSV
    target_sentiment = st.selectbox("Select Sentiment:", ["Positive", "Negative"])
    
    # Generate Word Cloud
    text_subset = " ".join(df[df['sentiment_result'].str.contains(target_sentiment, case=False)]['cleaned_review'].astype(str))
    
    if text_subset:
        wc = WordCloud(width=800, height=500, background_color='white', colormap='viridis').generate(text_subset)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 6))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
    else:
        st.write("No data available to display.")

st.markdown("---")

# --- 6. DATA TABLE ---
st.subheader("📂 Review Data View")
st.write("Displaying the first 50 rows of processed data:")
st.dataframe(df[['product_name', 'review_content', 'sentiment_result']].head(50), use_container_width=True)

# --- 7. FOOTER ---
st.caption("Project JIE43303: Natural Language Processing - Sentiment Analysis Dashboard")
