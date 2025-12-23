# 🎬 TED Talk Recommender System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![Status](https://img.shields.io/badge/Status-Completed-success)

A content-based recommendation engine that suggests TED Talks based on the semantic similarity of their transcripts.

This project was built to apply **Natural Language Processing (NLP)** techniques and mathematical vectorization to solve a real-world information retrieval problem.

## 🚀 Overview

Unlike collaborative filtering (which relies on user ratings), this system uses a **Content-Based Filtering** approach. It analyzes the actual words spoken in each talk to find patterns and similarities.

**Key Features:**
* **NLP Pipeline:** Text cleaning, tokenization, and stop-word removal.
* **TF-IDF Vectorization:** Converts unstructured text into a mathematical vector space.
* **Cosine Similarity:** Calculates the angular distance between vectors to rank the most similar talks.
* **Interactive UI:** Built with **Streamlit** for real-time interaction.

## 🧠 How it Works

The core logic follows these engineering steps:

1.  **Data Ingestion:** Merges metadata (titles, authors) with transcripts.
2.  **Preprocessing:** Cleans the raw text (lowercasing, removing punctuation/special characters, and filtering English stop words).
3.  **Vectorization (TF-IDF):** * Uses *Term Frequency-Inverse Document Frequency* to evaluate the importance of a word in a document relative to the collection.
    * Reduces the weight of common words and highlights unique keywords.
4.  **Similarity Matrix:** Computes the **Cosine Similarity** between all pairs of talks ($O(n^2)$ complexity) to determine closeness in the vector space.

## 🛠️ Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning / NLP:** Scikit-learn (TfidfVectorizer, linear_kernel), NLTK
* **Frontend:** Streamlit

## 📂 Project Structure

```bash
ted-talks-recommender/
├── data/                  # CSV datasets (excluded from repo)
├── venv/                  # Virtual Environment
├── app.py                 # Main application logic & UI
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation