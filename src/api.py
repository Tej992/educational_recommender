from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path to import classes
sys.path.append(os.path.abspath("src"))
from models import PopularityRecommender, ContentBasedRecommender, CollaborativeRecommender, PedagogicalHybridRecommender

app = FastAPI(title = 'Educational Content Recommender API')

# Load models
logger.info("Loading models...")
try:
    # Fix for models pickled directly via `python src/models.py`
    import sys
    sys.modules['__main__'].PopularityRecommender = PopularityRecommender
    sys.modules['__main__'].ContentBasedRecommender = ContentBasedRecommender
    sys.modules['__main__'].CollaborativeRecommender = CollaborativeRecommender
    sys.modules['__main__'].PedagogicalHybridRecommender = PedagogicalHybridRecommender

    with open("models/popularity_model.pkl", "rb") as f:
        pop_model = pickle.load(f)
    with open("models/content_model.pkl", "rb") as f:
        content_model = pickle.load(f)
    with open("models/collab_model.pkl", "rb") as f:
        collab_model = pickle.load(f)
    with open("models/hybrid_model.pkl", "rb") as f:
        hybrid_model = pickle.load(f)
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    pop_model = None
    content_model = None
    collab_model = None
    hybrid_model = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the Educational Content Recommender API"}

@app.get("/health")
def health_check():
    """Returns the operational status of the API and loaded models."""
    models_loaded = all([pop_model, content_model, collab_model, hybrid_model])
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded
    }

@app.get("/recommend/popularity")
def get_popularity_recommendations(n: int = 5):
    if not pop_model:
        raise HTTPException(status_code=500, detail = 'Model not loaded')
    recs = pop_model.recommend(n)
    return recs.to_dict(orient = 'records')

@app.get("/recommend/content/{course_title}")
def get_content_recommendations(course_title: str, n: int = 5):
    if not content_model:
        raise HTTPException(status_code=500, detail = 'Model not loaded')
    recs = content_model.recommend(course_title, n)
    if recs.empty:
        raise HTTPException(status_code=404, detail = 'Course not found')
    return recs.to_dict(orient = 'records')

@app.get("/recommend/collaborative/{user_id}")
def get_collaborative_recommendations(user_id: int, n: int = 5):
    if not collab_model:
        raise HTTPException(status_code=500, detail = 'Model not loaded')
    if user_id not in collab_model.interactions_df['user_id'].unique():
        raise HTTPException(status_code=404, detail=f"User ID {user_id} not found in database.")
    
    recs = collab_model.recommend(user_id, n)
    if recs.empty:
         # Fallback to popularity if user not found or no recs
        return get_popularity_recommendations(n)
    return recs.to_dict(orient = 'records')

@app.get("/recommend/hybrid/{user_id}")
def get_hybrid_recommendations(user_id: int, n: int = 5):
    if not hybrid_model:
        raise HTTPException(status_code=500, detail = 'Model not loaded')
    if user_id not in hybrid_model.interactions_df['user_id'].unique():
        raise HTTPException(status_code=404, detail=f"User ID {user_id} not found in database.")
        
    recs = hybrid_model.recommend(user_id, n)
    if recs.empty:
         # Fallback to popularity
        return get_popularity_recommendations(n)
    return recs.to_dict(orient = 'records')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = '0.0.0.0', port=8000)
