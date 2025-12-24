import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import scipy.sparse


def load_data():
    df_main = pd.read_csv("data/ted_main.csv")
    df_transcripts = pd.read_csv("data/transcripts.csv")
    df = pd.merge(left=df_main, right=df_transcripts, on="url", how="inner")

    return df


def clean_text(text):
    return re.sub(r"[^a-z\s]", " ", text.lower())


def save_optimized_files(df_lite, tfidf_matrix):
    df_lite.to_pickle("data/ted_lite.pkl")

    scipy.sparse.save_npz("data/tfidf_matrix.npz", tfidf_matrix)

    indices = pd.Series(df_lite.index, index=df_lite["title"]).drop_duplicates()
    indices.to_pickle("data/indices.pkl")


def main():
    print("Loading CSVs...")
    df = load_data()

    print("Cleaning text...")
    df["clean_transcript"] = df["transcript"].apply(clean_text)

    print("Generating TF-IDF Matrix...")
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["clean_transcript"])

    df_lite = df[["title", "main_speaker", "description", "tags", "url"]].copy()

    print("Saving optimized files...")
    save_optimized_files(df_lite, tfidf_matrix)

    print("Done! Files ready for Cloud.")


if __name__ == "__main__":
    main()
