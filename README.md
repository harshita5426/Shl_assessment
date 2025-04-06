 SHL Assessment Recommendation System

This Streamlit web app recommends the most relevant SHL assessments based on your input query like job roles, skills, or topics. You can also filter by test types (C, P, A, B, S, K).


 Features

- Enter a query like `"Java Developer"` or `"Logical Reasoning"`.
- Filter assessments by **Test Type** (C = Cognitive, P = Personality, etc.).
- Get top-N SHL assessments ranked by **semantic similarity**.
- Explanation of how results are ranked.
- Download recommendations as a CSV.

 How It Works

1. SHL assessment descriptions are embedded using `SentenceTransformer (all-MiniLM-L6-v2)`.
2. Input query is also embedded and compared via **cosine similarity**.
3. Optionally filtered by test type.
4. Results are ranked and displayed with download support.

---
