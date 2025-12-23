import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re

st.set_page_config(page_title="TED Talk Recommender", layout="wide")


@st.cache_data
def load_data():
    df_main = pd.read_csv("data/ted_main.csv")
    df_transcripts = pd.read_csv("data/transcripts.csv")
    df = pd.merge(left=df_main, right=df_transcripts, on="url", how="inner")
    return df


@st.cache_data
def clean_data(df):
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return text

    df["clean_transcript"] = df["transcript"].apply(clean_text)
    return df


@st.cache_data
def build_model(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["clean_transcript"])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
    return cosine_sim, indices


st.title("TED Talk Recommender Engine")
st.markdown(
    "Select a talk you enjoyed, and this AI will recommend 5 similar talks based on their content (transcript)."
)

with st.spinner("Loading data and training AI model..."):
    df = load_data()
    df = clean_data(df)
    cosine_sim, title_to_index = build_model(df)

selected_talk = st.selectbox("Select a TED Talk:", df["title"].values)


def get_recommendations(title):
    if title not in title_to_index:
        return []

    idx = title_to_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]

    talk_indices = [i[0] for i in sim_scores]
    return df.iloc[talk_indices]


if st.button("Get Recommendations"):
    st.subheader(f"Because you liked '{selected_talk}':")
    recommendations = get_recommendations(selected_talk)

    for _, row in recommendations.iterrows():
        with st.expander(f"{row['title']} (by {row['main_speaker']})"):
            st.write(f"**Description:** {row['description']}")
            st.caption(f"Tags: {row['tags']}")
            st.caption(f"Link: {row['url']}")
