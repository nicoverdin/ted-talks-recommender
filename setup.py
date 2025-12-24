import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import scipy.sparse


# Load data
print("Loading CSVs...")
df_main = pd.read_csv("data/ted_main.csv")
df_transcripts = pd.read_csv("data/transcripts.csv")
df = pd.merge(left=df_main, right=df_transcripts, on="url", how="inner")

# Clean text
print("Cleaning text...")


def clean_text(text):
    return re.sub(r"[^a-z\s]", " ", text.lower())


df["clean_transcript"] = df["transcript"].apply(clean_text)

print("Generating TF-IDF Matrix...")
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["clean_transcript"])

df_lite = df[["title", "main_speaker", "description", "tags", "url"]].copy()

print("Saving optimized files...")

df_lite.to_pickle("data/ted_lite.pkl")

scipy.sparse.save_npz("data/tfidf_matrix.npz", tfidf_matrix)

indices = pd.Series(df_lite.index, index=df_lite["title"]).drop_duplicates()
indices.to_pickle("data/indices.pkl")

print("Done! Files ready for Cloud.")
