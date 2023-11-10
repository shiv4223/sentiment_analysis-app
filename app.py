import streamlit as st
from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')

# Function to display the introduction and acknowledgments
def display_intro():
    st.title("Sentiment Analysis App")
    st.image('sentimentanalysishotelgeneric-2048x803-1.jpg')
    st.subheader("Submitted by:")
    st.markdown("Shivansh Galav (Roll No: 20104097 ")
    st.markdown("Manjeet (Roll No: 20104060)")
    st.subheader("Submitted to:")
    st.markdown("Dr. Bodile Roshan Kumar Mukindrao")
    st.image('Logo_of_NIT_Jalandhar.png')


# Function to display the main sentiment analysis page
def display_sentiment_page():
    st.header("Sentiment Analysis")
    st.image('bigstock-Market-Sentiment-Fear-And-Gre-451706057-2880x1800.jpg')
    # Input text box
    user_input = st.text_area("Enter a text for sentiment analysis:")

    # Button to perform sentiment analysis
    if st.button("Analyze Sentiment"):
        if user_input:
            # Perform sentiment analysis
            result = sentiment_pipeline(user_input)

            # Display result
            st.write(f"Sentiment: {result[0]['label']} (confidence: {result[0]['score']:.2f})")

# Function to display the "About the Project" page with additional information
def display_about():
    # Add title and description
    st.title("Sentiment Analysis Project")
    st.write("This Streamlit app provides information about sentiment analysis and various techniques used in the project.")
    st.image('61727sentiment-fig-1-689.jpeg')
    # Add information about sentiment analysis
    st.header("Sentiment Analysis Overview")
    st.write("""
    Sentiment analysis involves the process of analyzing text to determine the sentiment or emotional tone expressed, commonly categorized as positive, negative, or neutral. It plays a crucial role in Natural Language Processing (NLP) by helping to understand opinions, attitudes, or emotions conveyed in textual data.
    """)

    # Add information about NLP
    st.header("Natural Language Processing (NLP)")
    st.write("""
    Natural Language Processing is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. In this project, NLP techniques are employed to analyze and understand the sentiment conveyed in textual data, particularly tweets related to US Airlines.
    """)

    # Add overview of the project
    st.header("Project Overview")
    st.write("""
    **Data Description:**
    The dataset used in this project consists of tweets related to US Airlines, comprising approximately 10,000-15,000 tweets. The CSV file includes various columns such as tweets and their associated sentiments (Positive, Negative, Neutral).

    **Project Steps:**
    1. **Reading and Exploratory Data Analysis (EDA):** The initial steps involve reading the data and performing Exploratory Data Analysis (EDA) to gain insights.
    2. **Data Cleaning:** The data is cleaned to ensure its quality and reliability.
    3. **Training Model (Bag of Words):** A model is trained using the Bag of Words approach.
    4. **Training Model (Word2Vec):** A model is trained using Word2Vec, representing words as vectors in a continuous vector space.
    5. **Training Model (Transformer - BERT):** The project utilizes pre-trained models like BERT for context-aware feature extraction, enhancing sentiment analysis accuracy.

    For detailed code and steps, refer to the [Colab Notebook](https://colab.research.google.com/drive/1Rc0cg-xn_ZM_mzLwQizWMM0Q27QWF4AX?authuser=1#scrollTo=hqNJRp5ol3FM).
    """)

    # Add information about different approaches
    st.header("Sentiment Analysis Approaches")
    st.write("""
    - **Bag of Words (BoW):** This approach represents text as an unordered set of words, disregarding grammar and word order. It is a foundational technique in sentiment analysis.
    - **TF-IDF (Term Frequency-Inverse Document Frequency):** Measures the importance of each word in a document relative to a collection of documents, contributing to more nuanced sentiment analysis.
    - **Word2Vec:** In this approach, words are represented as vectors in a continuous vector space, capturing semantic relationships and enhancing sentiment understanding.
    - **Transformers (BERT):** Utilizing pre-trained models like BERT allows for context-aware feature extraction, improving the accuracy of sentiment analysis in diverse contexts.
    """)

    # Add a summary of the code
    st.header("Code Summary")
    st.write("""
    The provided code implements sentiment analysis on US Airlines tweets using various techniques. It includes data reading, cleaning, and training models using Bag of Words, Word2Vec, and BERT transformers. The code is available in the [Colab Notebook](https://colab.research.google.com/drive/1Rc0cg-xn_ZM_mzLwQizWMM0Q27QWF4AX?authuser=1#scrollTo=hqNJRp5ol3FM).
    """)




# Streamlit app
def main():
    # Page selection
    page = st.sidebar.selectbox("Select Page", ["Sentiment Analysis APP" , "Introduction to Team" , "About the Project"])

    # Display selected page
    if page == "Introduction to Team":
        display_intro()
    elif page == "Sentiment Analysis APP":
        display_sentiment_page()
    elif page == "About the Project":
        display_about()

if __name__ == "__main__":
    main()


