import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import scipy

st.set_page_config(page_title="TED Talk Recommender", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_pickle("data/ted_lite.pkl")
    indices = pd.read_pickle("data/indices.pkl")
    tfidf_matrix = scipy.sparse.load_npz("data/tfidf_matrix.npz")
    return df, indices, tfidf_matrix


st.title("TED Talk Recommender Engine")
st.markdown(
    "Select a talk you enjoyed, and this AI will recommend 5 similar talks based on their content (transcript)."
)

with st.spinner("Loading data and training AI model..."):
    df, title_to_index, tfidf_matrix = load_data()

selected_talk = st.selectbox("Select a TED Talk:", df["title"].values)


def get_recommendations(title):
    if title not in title_to_index:
        return []

    idx = title_to_index[title]

    query_vector = tfidf_matrix[idx]

    sim_scores = linear_kernel(query_vector, tfidf_matrix).flatten()

    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5

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
