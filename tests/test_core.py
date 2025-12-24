import os
from app import get_recommendations, load_data

df, title_to_index, tfidf_matrix = load_data()


# 1. TEST DE INTEGRIDAD DE DATOS
def test_data_files_exist():
    required_files = ["data/ted_lite.pkl", "data/indices.pkl", "data/tfidf_matrix.npz"]
    for f in required_files:
        assert os.path.exists(f), f"El archivo crítico {f} no existe."


# 2. TEST DE DIMENSIONES
def test_matrix_and_dataframe_match():
    assert (
        df.shape[0] == tfidf_matrix.shape[0]
    ), "Mismatch entre DataFrame y Matriz TF-IDF"


# 3. TEST DE LÓGICA DE RECOMENDACIÓN (Happy Path)
def test_recommendation_system_returns_5_talks():
    test_talk = "Do schools kill creativity?"  # Asegúrate de que este título exista en tu CSV original

    recs = get_recommendations(test_talk, title_to_index, tfidf_matrix, df)

    assert not recs.empty
    assert len(recs) == 5
    assert test_talk not in recs["title"].values


# 4. TEST DE CASOS BORDE (Edge Cases)
def test_recommendation_handles_unknown_talk():
    fake_talk = "Esta charla no existe 12345"

    recs = get_recommendations(fake_talk, title_to_index, tfidf_matrix, df)

    assert len(recs) == 0
