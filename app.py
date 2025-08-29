import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# Load Dataset
# ======================
df = pd.read_csv("product_recs.csv")

# ======================
# Collaborative Filtering Setup
# ======================
user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)


def predict_rating(user_id, product_id):
    if product_id not in user_item_matrix.columns:
        return 0
    sim_scores = user_similarity_df[user_id]
    product_ratings = user_item_matrix[product_id]
    return np.dot(sim_scores, product_ratings) / sim_scores.sum()


def cf_recommendations(user_id, top_n=5):
    unrated_products = user_item_matrix.columns[user_item_matrix.loc[user_id] == 0]
    scores = [(pid, predict_rating(user_id, pid)) for pid in unrated_products]
    scores.sort(key=lambda x: x[1], reverse=True)
    top_products = [(pid, df[df['product_id'] == pid]['product_name'].values[0]) for pid, _ in scores[:top_n]]
    return top_products


# ======================
# Content-Based Filtering Setup
# ======================
df['product_info'] = df['product_name'] + " " + df['category']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['product_info'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def cb_recommend(product_id, top_n=5):
    idx = df.index[df['product_id'] == product_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    product_indices = [i[0] for i in sim_scores]
    return df.iloc[product_indices][['product_id', 'product_name', 'category']]


# ======================
# Hybrid Recommendation
# ======================
def hybrid_recommend(user_id, top_n=5, alpha=0.7):
    cf_scores = {pid: predict_rating(user_id, pid) for pid in user_item_matrix.columns}
    content_scores = {}
    for pid in user_item_matrix.columns:
        sim_products = cb_recommend(pid, top_n=5)
        sim_score = np.mean([cf_scores[row['product_id']] for _, row in sim_products.iterrows()])
        content_scores[pid] = sim_score
    final_scores = {pid: alpha * cf_scores[pid] + (1 - alpha) * content_scores[pid] for pid in cf_scores.keys()}
    top_products = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(pid, df[df['product_id'] == pid]['product_name'].values[0]) for pid, _ in top_products]


# ======================
# Streamlit App
# ======================
st.set_page_config(page_title="🛒 Product Recommendation System", layout="wide")
st.title("🛒 Product Recommendation System")

menu = st.sidebar.radio("📌 Navigation", ["🏠 Home", "🔮 Single User", "📂 Batch Prediction", "📊 About Author"])

# Home Page
if menu == "🏠 Home":
    st.markdown("""
    Welcome to the **Product Recommendation System**! 🚀  
    This tool helps suggest products using **Collaborative Filtering**, **Content-Based Filtering**, and **Hybrid Recommendations**.

    ### Features:
    - 🔮 **Single User**: Enter a user ID and get recommendations.
    - 📂 **Batch Prediction**: Upload CSV to recommend products for multiple users.
    - 📊 **About Author**: Learn more about the creator.
    """)

# Single User Prediction
elif menu == "🔮 Single User":
    st.header("🔮 Single User Recommendations")
    user_id = st.number_input("Enter User ID:", min_value=int(df['user_id'].min()), max_value=int(df['user_id'].max()),
                              value=int(df['user_id'].min()))
    top_n = st.slider("Number of Recommendations:", min_value=1, max_value=10, value=5)

    if st.button("Get Recommendations"):
        st.subheader("Collaborative Filtering Recommendations")
        cf_recs = cf_recommendations(user_id, top_n)
        st.table(pd.DataFrame(cf_recs, columns=["Product ID", "Product Name"]))

        st.subheader("Hybrid Recommendations")
        hybrid_recs = hybrid_recommend(user_id, top_n)
        st.table(pd.DataFrame(hybrid_recs, columns=["Product ID", "Product Name"]))

# Batch Prediction
elif menu == "📂 Batch Prediction":
    st.header("📂 Batch Recommendations")
    uploaded_file = st.file_uploader("Upload CSV File with User IDs", type=["csv"])
    top_n = st.slider("Number of Recommendations per User:", min_value=1, max_value=10, value=5)

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:", batch_df.head())
        results = []

        for uid in batch_df['user_id']:
            recs = hybrid_recommend(uid, top_n)
            for pid, pname in recs:
                results.append({"user_id": uid, "product_id": pid, "product_name": pname})

        results_df = pd.DataFrame(results)
        st.dataframe(results_df.head(20))
        results_df.to_csv("batch_recommendations.csv", index=False)
        st.success("💾 Batch Recommendations saved as batch_recommendations.csv")

        # Visualization
        fig, ax = plt.subplots()
        sns.countplot(x=results_df['product_name'], order=results_df['product_name'].value_counts().index,
                      palette="coolwarm", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

# About Author
elif menu == "📊 About Author":
    st.sidebar.markdown("""
    **Author:**  
    Sherriff Abdul-Hamid  

    **Email:**  
    [sherriffhamid001@gmail.com](mailto:sherriffhamid001@gmail.com)  

    **LinkedIn:**  
    [LinkedIn](https://www.linkedin.com/in/sherriffhamid)
    """)
    st.markdown("""
    **Author:** Sherriff Abdul-Hamid  
    Creator of this Product Recommendation System.  
    Uses real-time recommendations with **Python, Pandas, Scikit-learn**.
    """)
