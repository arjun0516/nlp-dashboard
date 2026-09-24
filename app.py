import streamlit as st
import spacy
import pandas as pd
from collections import Counter
from nltk.stem import PorterStemmer

# Page configuration
st.set_page_config(
    page_title="NLP Dashboard",
    layout="wide"
)

# Load spaCy model
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()
stemmer = PorterStemmer()

# Title
st.title("Natural Language Processing Dashboard")

st.write(
    "Analyze text using Tokenization, Stopword Removal, "
    "Lemmatization, Named Entity Recognition, POS Tagging, "
    "POS Distribution and Stemming."
)

# Text input
st.subheader("Enter Your Text")

text = st.text_area(
    "Input Text",
    value=(
        "Apple was founded by Steve Jobs in California in 1976. "
        "The company develops innovative products and employs "
        "thousands of people around the world."
    ),
    height=150
)

if text.strip():

    doc = nlp(text)

    # Statistics
    tokens = [token for token in doc if not token.is_space]

    stopwords = [
        token for token in doc
        if token.is_stop
    ]

    punctuation = [
        token for token in doc
        if token.is_punct
    ]

    entities = list(doc.ents)

    # Statistics dashboard
    st.subheader("Text Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Number of Tokens", len(tokens))

    with col2:
        st.metric("Number of Stopwords", len(stopwords))

    with col3:
        st.metric("Number of Punctuation", len(punctuation))

    with col4:
        st.metric("Number of Entities", len(entities))

    st.divider()

    # Sidebar
    st.sidebar.title("NLP Operations")

    operation = st.sidebar.selectbox(
        "Select an Operation",
        [
            "Tokenization",
            "Stopword Removal",
            "Lemmatization",
            "Named Entity Recognition",
            "POS Tagging",
            "POS Distribution",
            "Stemming"
        ]
    )

    # Tokenization
    if operation == "Tokenization":

        st.header("Tokenization")

        st.write(
            "Tokenization divides the input text into individual "
            "words, numbers and punctuation tokens."
        )

        data = []

        for token in doc:
            if not token.is_space:
                data.append({
                    "Token": token.text,
                    "Type": (
                        "Punctuation"
                        if token.is_punct
                        else "Word"
                    )
                })

        df = pd.DataFrame(data)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    # Stopword Removal
    elif operation == "Stopword Removal":

        st.header("Stopword Removal")

        st.write(
            "Stopword removal identifies and removes common "
            "words that generally carry less semantic meaning."
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Stopwords Found")

            stopword_data = [
                token.text
                for token in doc
                if token.is_stop
            ]

            if stopword_data:
                st.write(", ".join(stopword_data))
            else:
                st.info("No stopwords found.")

        with col2:

            st.subheader("Text After Stopword Removal")

            filtered_text = " ".join(
                token.text
                for token in doc
                if not token.is_stop
                and not token.is_punct
            )

            st.write(filtered_text)

        st.subheader("Stopword Details")

        data = []

        for token in doc:
            if token.is_stop:
                data.append({
                    "Stopword": token.text,
                    "Lemma": token.lemma_
                })

        if data:
            df = pd.DataFrame(data)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

    # Lemmatization
    elif operation == "Lemmatization":

        st.header("Lemmatization")

        st.write(
            "Lemmatization converts words into their base "
            "or dictionary form."
        )

        data = []

        for token in doc:
            if not token.is_space:
                data.append({
                    "Word": token.text,
                    "Lemma": token.lemma_,
                    "POS": token.pos_
                })

        df = pd.DataFrame(data)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    # Named Entity Recognition
    elif operation == "Named Entity Recognition":

        st.header("Named Entity Recognition")

        st.write(
            "Named Entity Recognition identifies people, "
            "organizations, locations, dates and other entities."
        )

        if entities:

            data = []

            for entity in entities:
                data.append({
                    "Entity": entity.text,
                    "Label": entity.label_,
                    "Description": spacy.explain(entity.label_)
                })

            df = pd.DataFrame(data)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("No named entities were detected.")

    # POS Tagging
    elif operation == "POS Tagging":

        st.header("Part-of-Speech Tagging")

        st.write(
            "POS tagging identifies the grammatical category "
            "of each word."
        )

        data = []

        for token in doc:
            if not token.is_space:
                data.append({
                    "Word": token.text,
                    "POS": token.pos_,
                    "Detailed Tag": token.tag_,
                    "Description": spacy.explain(token.tag_)
                })

        df = pd.DataFrame(data)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    # POS Distribution
    elif operation == "POS Distribution":

        st.header("POS Distribution")

        st.write(
            "This section displays the frequency of each "
            "Part-of-Speech category."
        )

        pos_counts = Counter(
            token.pos_
            for token in doc
            if not token.is_space
            and not token.is_punct
        )

        df = pd.DataFrame(
            pos_counts.items(),
            columns=["POS", "Count"]
        ).sort_values(
            by="Count",
            ascending=False
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("POS Frequency")

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

        with col2:

            st.subheader("POS Distribution Chart")

            chart_data = df.set_index("POS")

            st.bar_chart(chart_data)

    # Stemming
    elif operation == "Stemming":

        st.header("Stemming")

        st.write(
            "Stemming reduces words to their stem using "
            "the Porter Stemmer algorithm."
        )

        data = []

        for token in doc:
            if token.is_alpha:
                data.append({
                    "Word": token.text,
                    "Stem": stemmer.stem(token.text)
                })

        df = pd.DataFrame(data)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

else:

    st.info(
        "Please enter some text above to begin analysis."
    )