import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PopularityRecommender:
    def __init__(self, df):
        self.df = df
        self.sorted_df = self.df.sort_values(by='course_rating', ascending=False)

    def recommend(self, n=5):
        return self.sorted_df.head(n)[['course_title', 'course_rating', 'course_difficulty']]

class ContentBasedRecommender:
    def __init__(self, df):
        self.df = df
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['combined_text'].fillna(''))

    def recommend(self, title, n=5):
        # Check if title exists
        if title not in self.df['course_title'].values:
            return pd.DataFrame()
            
        idx = self.df[self.df['course_title'] == title].index[0]
        cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]
        course_indices = [i[0] for i in sim_scores]
        return self.df.iloc[course_indices][['course_title', 'course_rating', 'course_difficulty']]

class CollaborativeRecommender:
    def __init__(self, interactions_df, courses_df):
        self.interactions_df = interactions_df
        self.courses_df = courses_df
        self.user_item_matrix = self.interactions_df.pivot(index='user_id', columns='course_title', values='rating').fillna(0)
        # Simple SVD
        self.U, self.sigma, self.Vt = np.linalg.svd(self.user_item_matrix, full_matrices=False)
        
    def predict_rating(self, user_id, course_title):
        if user_id not in self.user_item_matrix.index or course_title not in self.user_item_matrix.columns:
            return 0
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        course_idx = self.user_item_matrix.columns.get_loc(course_title)
        return np.dot(np.dot(self.U[user_idx, :], np.diag(self.sigma)), self.Vt[:, course_idx])

    def recommend(self, user_id, n=5):
        if user_id not in self.user_item_matrix.index:
            return pd.DataFrame()
            
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        # Reconstruct prediction
        pred_ratings = np.dot(np.dot(self.U[user_idx, :], np.diag(self.sigma)), self.Vt)
        
        # Get top rated items that user hasn't seen
        user_ratings = self.user_item_matrix.iloc[user_idx]
        seen_courses = user_ratings[user_ratings > 0].index.tolist()
        
        pred_df = pd.DataFrame({'course_title': self.user_item_matrix.columns, 'predicted_rating': pred_ratings})
        pred_df = pred_df[~pred_df['course_title'].isin(seen_courses)]
        
        top_courses = pred_df.sort_values(by='predicted_rating', ascending=False).head(n)
        return top_courses.merge(self.courses_df, on='course_title')[['course_title', 'predicted_rating', 'course_difficulty']]

class PedagogicalHybridRecommender:
    """
    A unified hybrid recommender that applies Lev Vygotsky's Zone of Proximal 
    Development (ZPD) penalty to algorithmic collaborative filtering predictions.
    """
    def __init__(self, collab_model, content_model, interactions_df, courses_df, lambda_weight=0.5):
        """
        Initializes the Hybrid Recommender.

        Args:
            collab_model: The Collaborative filter model (SVD).
            content_model: The Content-based filter model (TF-IDF).
            interactions_df (pd.DataFrame): User interaction history mapping.
            courses_df (pd.DataFrame): Educational course metadata.
            lambda_weight (float): Hyperparameter controlling the severity of the ZPD penalty.
        """
        self.collab_model = collab_model
        self.content_model = content_model
        self.interactions_df = interactions_df
        self.courses_df = courses_df
        self.lambda_weight = lambda_weight
        
        # Map difficulty to numeric
        self.diff_map = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3, 'Mixed': 1.5}
        self.courses_df['difficulty_score'] = self.courses_df['course_difficulty'].map(self.diff_map).fillna(1.5)

    def infer_user_skill(self, user_id):
        """
        Dynamically infers a user's current cognitive capability based on successful course interactions.

        Args:
            user_id (int): The unique identifier for the user.
            
        Returns:
            float: A scalar value representing the user's inferred skill capability.
        """
        # Get courses user rated highly (>3)
        user_history = self.interactions_df[
            (self.interactions_df['user_id'] == user_id) & 
            (self.interactions_df['rating'] > 3)
        ]
        
        if len(user_history) == 0:
            return 1.0 # Default to Beginner
            
        # Join with course difficulty
        user_history = user_history.merge(self.courses_df, on='course_title')
        avg_difficulty = user_history['difficulty_score'].median()
        return avg_difficulty

    def recommend(self, user_id, n=5):
        """
        Generates personalized pedagogical recommendations by penalizing collaborative 
        predictions mathematically outside the user's inferred capability zone.

        Args:
            user_id (int): The unique identifier for the user.
            n (int): Number of top recommendations to return.

        Returns:
            pd.DataFrame: Top pedagogical recommendations formatted for frontend consumption.
        """
        # 1. Get Collaborative Candidates (Top 20)
        collab_recs = self.collab_model.recommend(user_id, n=20)
        if collab_recs.empty:
            return pd.DataFrame()
            
        # 2. Infer User Skill
        user_skill = self.infer_user_skill(user_id)
        
        # 3. Apply Pedagogical Penalty (Zone of Proximal Development)
        # Penalty increases as distance between user_skill and course_difficulty increases
        collab_recs['difficulty_score'] = collab_recs['course_difficulty'].map(self.diff_map).fillna(1.5)
        collab_recs['diff_penalty'] = abs(collab_recs['difficulty_score'] - user_skill)
        
        # New Score = Predicted Rating - (Penalty * Weight)
        # Weight controls the strictness of the pedagogical matching
        collab_recs['pedagogical_score'] = collab_recs['predicted_rating'] - (collab_recs['diff_penalty'] * self.lambda_weight)
        
        # 4. Sort and Return
        final_recs = collab_recs.sort_values(by='pedagogical_score', ascending=False).head(n)
        return final_recs[['course_title', 'predicted_rating', 'course_difficulty', 'pedagogical_score']]

def train_models(data_path = 'data/processed_courses.csv', interactions_path = 'data/user_interactions.csv'):
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    logger.info("Training models...")
    df = pd.read_csv(data_path)
    
    pop_model = PopularityRecommender(df)
    content_model = ContentBasedRecommender(df)
    
    if os.path.exists(interactions_path):
        interactions_df = pd.read_csv(interactions_path)
        collab_model = CollaborativeRecommender(interactions_df, df)
        hybrid_model = PedagogicalHybridRecommender(collab_model, content_model, interactions_df, df)
        
        with open("models/collab_model.pkl", "wb") as f:
            pickle.dump(collab_model, f)
            
        with open("models/hybrid_model.pkl", "wb") as f:
            pickle.dump(hybrid_model, f)
    
    os.makedirs("models", exist_ok=True)
    with open("models/popularity_model.pkl", "wb") as f:
        pickle.dump(pop_model, f)
        
    with open("models/content_model.pkl", "wb") as f:
        pickle.dump(content_model, f)
        
    logger.info("Models saved to models/")

if __name__ == "__main__":
    train_models()
