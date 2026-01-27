import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Amazon Sentiment Analysis Dashboard",
    page_icon="🛒",
    layout="wide"
)

# --- 2. DATA LOADING FUNCTION ---
@st.cache_data
def load_data():
    # Make sure this file exists in your folder/GitHub
    try:
        df = pd.read_csv("amazon_cleaned_with_sentiment.csv")
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

# --- 3. SIDEBAR CONFIGURATION ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=120)
st.sidebar.title(" Dashboard Controls")
st.sidebar.markdown("---")

st.sidebar.header("Live Sentiment Test")
st.sidebar.write("Test the model with your own text:")
user_input = st.sidebar.text_area("Enter review text here:", height=100)

if st.sidebar.button("Analyze Text"):
    if user_input:
        blob = TextBlob(user_input)
        polarity = blob.sentiment.polarity
        
        st.sidebar.markdown("### Result:")
        if polarity > 0:
            st.sidebar.success(f"😊 Positive (Score: {polarity:.2f})")
        elif polarity == 0:
            st.sidebar.info(f"😐 Neutral (Score: {polarity:.2f})")
        else:
            st.sidebar.error(f"😡 Negative (Score: {polarity:.2f})")
    else:
        st.sidebar.warning("Please enter some text first.")

st.sidebar.markdown("---")
st.sidebar.info("Project: JIE43303 NLP\nCreated by: [Your Name]")

# --- 4. MAIN DASHBOARD UI ---

if not df.empty:
    st.title("📊 Amazon Product Sentiment Analysis")
    st.markdown("This dashboard analyzes customer reviews using **Natural Language Processing (NLP)** to determine overall sentiment trends.")
    st.markdown("---")

    # --- A. KEY METRICS ---
    st.subheader(" Key Performance Indicators (KPIs)")
    
    total_reviews = len(df)
    pos_reviews = len(df[df['sentiment_result'] == 'Positive'])
    neg_reviews = len(df[df['sentiment_result'] == 'Negative'])
    neu_reviews = len(df[df['sentiment_result'] == 'Neutral'])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", f"{total_reviews}")
    col2.metric("Positive Reviews", f"{pos_reviews}", f"{(pos_reviews/total_reviews)*100:.1f}%")
    col3.metric("Negative Reviews", f"{neg_reviews}", f"-{(neg_reviews/total_reviews)*100:.1f}%", delta_color="inverse")
    col4.metric("Neutral Reviews", f"{neu_reviews}", f"{(neu_reviews/total_reviews)*100:.1f}%")

    st.markdown("---")

    # --- B. VISUALIZATIONS ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Sentiment Distribution")
        
        # Prepare data for chart
        sentiment_counts = df['sentiment_result'].value_counts()
        
        # Create Bar Chart using Matplotlib/Seaborn
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=colors, ax=ax)
        
        ax.set_title("Number of Reviews by Sentiment Category")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col_right:
        st.subheader("☁️ Word Cloud Analysis")
        sentiment_filter = st.selectbox("Select Sentiment to Visualize:", ["Positive", "Negative"])
        
        # Filter text based on selection
        text_data = " ".join(df[df['sentiment_result'] == sentiment_filter]['cleaned_review'].astype(str))
        
        if text_data:
            # Generate WordCloud
            wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text_data)
            
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)
        else:
            st.warning("No reviews available for this category.")

    st.markdown("---")

    # --- C. DATA TABLE ---
    st.subheader(" Processed Data Review")
    st.write("Explore the cleaned data and sentiment labels:")
    
    # Display interactive dataframe
    st.dataframe(df[['product_name', 'review_content', 'sentiment_result']].head(50), use_container_width=True)

else:
    # Error handling if file is missing
    st.error(" Data file not found! Please ensure 'amazon_cleaned_with_sentiment.csv' is in the repository.")
