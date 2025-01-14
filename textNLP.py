import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from textblob import TextBlob
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import string

def preprocess_text(text):
    """Cleans and tokenizes the text."""
    # Ensure necessary data is downloaded
    nltk.download('stopwords')
    nltk.download('punkt', quiet=True)
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word not in punctuation]
    return tokens

def generate_wordcloud(text):
    """Generates a word cloud from the text."""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

def extract_keywords(text):
    """Extracts keywords using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_scores[:10]]

def analyze_sentiment(text):
    """Analyzes sentiment using TextBlob."""
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# Streamlit App
st.title("NLP 기반 문학 텍스트 분석")
st.write("이 애플리케이션은 자연어 처리를 통해 텍스트 전처리, 주제어 추출, 감정 분석, 워드 클라우드 생성을 수행합니다.")

# Text input
user_text = st.text_area("분석할 텍스트를 입력하세요:", height=200)

# Stopwords and punctuation for preprocessing
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

if st.button("분석 실행"):
    if user_text.strip():
        # Text Preprocessing
        tokens = preprocess_text(user_text)
        processed_text = " ".join(tokens)

        # Word Cloud
        st.subheader("워드 클라우드")
        wordcloud = generate_wordcloud(processed_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

        # Keyword Extraction
        st.subheader("주요 키워드")
        keywords = extract_keywords(user_text)
        st.write(", ".join(keywords))

        # Sentiment Analysis
        st.subheader("감정 분석")
        polarity, subjectivity = analyze_sentiment(user_text)
        st.write(f"**Polarity** (긍정/부정 지표): {polarity:.2f}")
        st.write(f"**Subjectivity** (주관성 지표): {subjectivity:.2f}")

        # Tokenized Text
        st.subheader("전처리된 텍스트")
        st.write(processed_text)
    else:
        st.warning("텍스트를 입력하세요.")
