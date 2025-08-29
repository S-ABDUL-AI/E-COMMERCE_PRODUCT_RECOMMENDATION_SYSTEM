import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ======================
# Create or Load Dataset
# ======================
df = pd.read_csv("product_recs.csv")
print(df.head())

# ======================
# Collaborative Filtering (User-Item Matrix)
# ======================
user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Function to predict rating for a user-product pair
def predict_rating(user_id, product_id):
    if product_id not in user_item_matrix.columns:
        return 0
    # Similarity scores for current user with other users
    sim_scores = user_similarity_df[user_id]
    # Ratings from other users for the product
    product_ratings = user_item_matrix[product_id]
    # Weighted sum
    weighted_avg = np.dot(sim_scores, product_ratings) / sim_scores.sum()
    return weighted_avg

# Top-N recommendations based on CF
def cf_recommendations(user_id, top_n=5):
    unrated_products = user_item_matrix.columns[user_item_matrix.loc[user_id]==0]
    scores = [(pid, predict_rating(user_id, pid)) for pid in unrated_products]
    scores.sort(key=lambda x: x[1], reverse=True)
    top_products = [(pid, df[df['product_id']==pid]['product_name'].values[0]) for pid, _ in scores[:top_n]]
    return top_products

# Example
print("CF Recommendations for User 1:")
print(cf_recommendations(1))

# ======================
# Content-Based Filtering
# ======================
df['product_info'] = df['product_name'] + " " + df['category']
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['product_info'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def cb_recommend(product_id, top_n=5):
    idx = df.index[df['product_id'] == product_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    product_indices = [i[0] for i in sim_scores]
    return df.iloc[product_indices][['product_id','product_name','category']]

# Example
print("CB Recommendations similar to Product 1:")
print(cb_recommend(1))

# ======================
# Hybrid Recommendation
# ======================
def hybrid_recommend(user_id, top_n=5, alpha=0.7):
    cf_scores = {pid: predict_rating(user_id, pid) for pid in user_item_matrix.columns}
    # Content scores: average similarity of each product with others
    content_scores = {}
    for pid in user_item_matrix.columns:
        sim_products = cb_recommend(pid, top_n=5)
        sim_score = np.mean([cf_scores[row['product_id']] for _, row in sim_products.iterrows()])
        content_scores[pid] = sim_score
    # Combine
    final_scores = {pid: alpha*cf_scores[pid] + (1-alpha)*content_scores[pid] for pid in cf_scores.keys()}
    top_products = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(pid, df[df['product_id']==pid]['product_name'].values[0]) for pid,_ in top_products]

# Example
print("Hybrid Recommendations for User 1:")
print(hybrid_recommend(1))
