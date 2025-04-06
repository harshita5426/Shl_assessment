import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Title and description
st.title("🔍 SHL Assessment Recommendation System")
st.write("This app helps you find the most relevant SHL assessments based on your role or keyword.")

# Load model and data
@st.cache_data
def load_data():
    df = pd.read_csv("shl_assessments_enriched.csv")
    with open("shl_embeddings.pkl", "rb") as f:
        df_embeds, embeddings = pickle.load(f)
    return df, embeddings

df, embeddings = load_data()
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Search box ---
query = st.text_input("Enter role/skill/assessment query:", "Software Developer Assessment")

# --- Filter options ---
selected_types = st.multiselect(
    "Filter by Test Type (optional):",
    options=["C", "P", "A", "B", "S", "K"],
    default=["C", "P"]
)

top_n = st.slider("How many results to show?", min_value=1, max_value=20, value=5)

# --- Recommendation Function ---
def find_similar_assessments(query, filter_by_types=None, top_n=5):
    query_embedding = model.encode(query).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    df["Similarity"] = similarities

    if filter_by_types:
        def test_type_match(row):
            return any(t in row for t in filter_by_types)
        filtered_df = df[df["Test Type"].apply(test_type_match)]
    else:
        filtered_df = df

    top_matches = filtered_df.sort_values(by="Similarity", ascending=False).head(top_n)
    return top_matches

# --- Run search ---
if st.button("🔎 Recommend"):
    results = find_similar_assessments(query, filter_by_types=selected_types, top_n=top_n)

    if results.empty:
        st.warning("No matching assessments found. Try a broader query or different filters.")
    else:
        st.success("Top matching assessments found!")
        st.dataframe(results[["Assessment Name", "Test Type", "Similarity", "URL"]], use_container_width=True, height=400)

        # Show explanation
        with st.expander("💡 How are results ranked?"):
            st.write("""
                - We embed the query and each assessment using SentenceTransformer (`all-MiniLM-L6-v2`).
                - We compute cosine similarity between the query and each assessment.
                - Higher similarity means more relevance.
                - Filters (like Test Type) are applied **after** computing similarity.
            """)

        # Option to download results
        csv = results.to_csv(index=False)
        st.download_button("⬇️ Download Results as CSV", csv, file_name="shl_recommendations.csv", mime="text/csv")
