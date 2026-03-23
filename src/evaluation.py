import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, ndcg_score
from sklearn.model_selection import KFold
from scipy import stats
from math import sqrt
import sys
import os
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath("src"))
from models import PopularityRecommender, ContentBasedRecommender, CollaborativeRecommender, PedagogicalHybridRecommender

def evaluate_fold(train_df, test_df, courses_df, lambdas=[0.1, 0.3, 0.5, 0.7, 1.0]):
    collab_model = CollaborativeRecommender(train_df, courses_df)
    content_model = ContentBasedRecommender(courses_df)
    
    # We will test multiple lambdas for the hybrid model simultaneously
    hybrid_models = {l: PedagogicalHybridRecommender(collab_model, content_model, train_df, courses_df, lambda_weight=l) for l in lambdas}
    
    svd_rmse = []
    svd_ndcg = []
    
    hybrid_metrics = {l: {'rmse': [], 'ndcg': []} for l in lambdas}
    
    cb_ndcg = []
    
    test_users = test_df['user_id'].unique()
    
    for user_id in test_users:
        user_test_data = test_df[test_df['user_id'] == user_id]
        if len(user_test_data) < 2: continue
        
        # Ground Truth
        true_ratings = dict(zip(user_test_data['course_title'], user_test_data['rating']))
        
        # We need identically ordered arrays for NDCG
        course_titles = list(true_ratings.keys())
        y_true = np.array([true_ratings[t] for t in course_titles])
        
        svd_preds = []
        cb_preds = []
        
        # Evaluate Content Based
        try:
            # Simple fallback for CB: we use a history item to get similar items
            # Just predict arbitrary scores based on TF-IDF
            history = train_df[train_df['user_id'] == user_id]
            if not history.empty:
                fav_course = history.iloc[0]['course_title']
                cb_recs = content_model.recommend(fav_course, n=50) # get plenty
                for title in course_titles:
                    if title in cb_recs.values:
                        cb_preds.append(4.0) # crude approximation for CB score
                    else:
                        cb_preds.append(1.0)
            else:
                cb_preds = [1.0]*len(course_titles)
        except:
            cb_preds = [1.0]*len(course_titles)

            
        for title in course_titles:
            svd_preds.append(collab_model.predict_rating(user_id, title))
            
        svd_preds = np.array(svd_preds)
        
        # Calculate Hybrid manually for speed
        user_skill = hybrid_models[0.1].infer_user_skill(user_id)
        hybrid_preds = {l: [] for l in lambdas}
        
        for i, title in enumerate(course_titles):
            course_row = courses_df[courses_df['course_title'] == title].iloc[0]
            diff_score = hybrid_models[0.1].diff_map.get(course_row['course_difficulty'], 1.5)
            penalty = abs(diff_score - user_skill)
            
            for l in lambdas:
                hybrid_preds[l].append(svd_preds[i] - (penalty * l))
                
        # Calculate Metrics
        if len(y_true) > 1:
            svd_rmse.append(sqrt(mean_squared_error(y_true, svd_preds)))
            try:
                # Sklearn ndcg_score expects shape (1, n_items)
                svd_ndcg_val = ndcg_score([y_true], [svd_preds], k=10)
                svd_ndcg.append(svd_ndcg_val)
                cb_ndcg.append(ndcg_score([y_true], [cb_preds], k=10))
            except Exception as e:
                pass
                
            for l in lambdas:
                hybrid_metrics[l]['rmse'].append(sqrt(mean_squared_error(y_true, hybrid_preds[l])))
                try:
                    hybrid_metrics[l]['ndcg'].append(ndcg_score([y_true], [hybrid_preds[l]], k=10))
                except Exception as e:
                    pass

    return {
        'svd_rmse': np.mean(svd_rmse) if svd_rmse else 0,
        'svd_ndcg': np.mean(svd_ndcg) if svd_ndcg else 0,
        'cb_ndcg': np.mean(cb_ndcg) if cb_ndcg else 0,
        'hybrid_metrics': {l: {
            'rmse': np.mean(hybrid_metrics[l]['rmse']) if hybrid_metrics[l]['rmse'] else 0,
            'ndcg': np.mean(hybrid_metrics[l]['ndcg']) if hybrid_metrics[l]['ndcg'] else 0
        } for l in lambdas}
    }

def rigorous_evaluation():
    logger.info("Starting Rigorous Evaluation (5-Fold CV with Lambda Grid Search)...")
    
    interactions_df = pd.read_csv("data/user_interactions.csv")
    courses_df = pd.read_csv("data/processed_courses.csv")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    results = []
    
    fold = 1
    for train_idx, test_idx in kf.split(interactions_df):
        logger.info(f"Evaluating Fold {fold}/5...")
        train_df = interactions_df.iloc[train_idx]
        test_df = interactions_df.iloc[test_idx]
        
        metrics = evaluate_fold(train_df, test_df, courses_df, lambdas)
        results.append(metrics)
        fold += 1
        
    avg_svd_rmse = np.mean([r['svd_rmse'] for r in results])
    avg_svd_ndcg = np.mean([r['svd_ndcg'] for r in results])
    avg_cb_ndcg = np.mean([r['cb_ndcg'] for r in results])

    logger.info("--- Final Results ---")
    logger.info(f"SVD RMSE: {avg_svd_rmse:.4f}")
    logger.info(f"SVD NDCG: {avg_svd_ndcg:.4f}")
    logger.info(f"CBF NDCG: {avg_cb_ndcg:.4f}")
    
    # Aggregate Grid Search
    lambda_ndcgs = []
    logger.info("--- Lambda Grid Search ---")
    for l in lambdas:
        l_rmse = np.mean([r['hybrid_metrics'][l]['rmse'] for r in results])
        l_ndcg = np.mean([r['hybrid_metrics'][l]['ndcg'] for r in results])
        lambda_ndcgs.append(l_ndcg)
        logger.info(f"Lambda={l:.1f} | RMSE: {l_rmse:.4f} | NDCG: {l_ndcg:.4f}")

    # Plot
    os.makedirs('output', exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.plot(lambdas, lambda_ndcgs, marker='o', linestyle='-', color='b')
    plt.axhline(y=avg_svd_ndcg, color='r', linestyle='--', label=f'Base SVD ({avg_svd_ndcg:.4f})')
    plt.title('Hybrid Model NDCG@10 Optimization via Lambda Grid Search')
    plt.xlabel('ZPD Penalty Weight (Lambda)')
    plt.ylabel('Average NDCG@10')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/lambda_tuning.png')
    logger.info("Saved output/lambda_tuning.png")

if __name__ == "__main__":
    rigorous_evaluation()
