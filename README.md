# 🎬 TED Talk Recommender System

[![Docker Build CI](https://github.com/nicoverdin/ted-talks-recommender/actions/workflows/docker-build.yml/badge.svg)](https://github.com/nicoverdin/ted-talks-recommender/actions/workflows/docker-build.yml)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![Status](https://img.shields.io/badge/Status-Completed-success)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ted-talks-recommender.onrender.com/)

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

## 🧠 Architecture & Optimization

To ensure high performance and low memory usage in production environments (like Render Free Tier), the project follows a **Split-Architecture** approach:

1.  **Offline Batch Processing (`setup.py`):**
    * Loads raw CSV data.
    * Cleans and tokenizes text.
    * **Pre-computes** the TF-IDF matrix locally.
    * Exports lightweight artifacts (`.pkl`, `.npz`) containing only the mathematical vectors and necessary metadata.

2.  **Online Inference Engine (`app.py`):**
    * Loads the pre-computed sparse matrix (memory efficient).
    * Performs **Cosine Similarity** calculations on-demand ($O(1 \times N)$ complexity) instead of calculating the full matrix ($O(N^2)$).
    * Delivers sub-second recommendations without reprocessing text.

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
```

## ⚡ How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/ted-talks-recommender.git](https://github.com/YOUR_USERNAME/ted-talks-recommender.git)
    cd ted-talks-recommender
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Data & Pre-compute Vectors:**
    * Download `ted_main.csv` and `transcripts.csv` from [Kaggle](https://www.kaggle.com/datasets/rounakbanik/ted-talks) into the `data/` folder.
    * **Run the setup script to generate the models:**
        ```bash
        python setup.py
        ```
    * *This will create `ted_lite.pkl` and `tfidf_matrix.npz`.*

5.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

### 🐳 Run with Docker (Recommended)

You can run the application without installing Python or dependencies manually.
*Note: The Docker image will automatically bundle the pre-computed vector models present in your repository.*

1.  **Build the image:**
    ```bash
    docker build -t ted-recommender .
    ```

2.  **Run the container:**
    ```bash
    docker run -p 8501:8501 ted-recommender
    ```

## 🔮 Future Improvements

* Implement a hybrid approach combining content-based filtering with collaborative filtering.
* Deploy the application to Streamlit Cloud or AWS.
* Optimize the similarity search using approximate nearest neighbors (like Faiss) for larger datasets.

---
*Built with ❤️ by Nicolás Verdín Domínguez*