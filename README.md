# CM3070 Final Project: Educational Recommender System

## Overview
This repository contains the source code for the CM3070 Final Project (Project Idea 1.1), focused on Educational Content Recommendation. The project implements a novel Hybrid Pedagogical Recommender engine that modifies standard Singular Value Decomposition (SVD) with a penalty mathematically linked to Lev Vygotsky's Zone of Proximal Development (ZPD). This shifts the algorithmic paradigm from engagement-based 'click-optimization' to actual pedagogical outcomes.

## Repository Contents
- `src/` (or root): Contains the primary source code modules for the recommender system.
  - `api.py`: The FastAPI server containing the routing and fallback logic for live recommendations.
  - `preprocessing.py`: Handles data ingestion from raw Coursera datasets, imputes nulls, and calculates TF-IDF embeddings.
  - `generate_interactions.py`: Stochastically simulates user ratings anchoring to structured personas to evaluate ranking capability.
- `models/`: Saved `scikit-learn` and SVD object artifacts.
- `data/`: Ingested and synthetically generated interaction arrays.
- `output/`: Auto-generated visualization plots and charts utilized within the report.

## Installation & Usage

To ensure reproducibility, trained model files (`.pkl`) are explicitly excluded from version control. Follow these steps to train the algorithms from the raw `.csv` data and launch the recommendation server.

### Step 1: Clone Repository & Create Virtual Environment
First, clone the repository to your local machine:
```bash
git clone https://github.com/Tej992/educational_recommender.git
cd educational_recommender
```

Next, ensure Python 3.9+ is installed. Isolate the environment using `venv` and install all required dependent packages:

**Windows (PowerShell/CMD):**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**Mac/Linux (Bash):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Train Models & Run Evaluation
Run the training script to independently compile the SVD Collaborative and TF-IDF Content models from the raw data. This will automatically generate the required `.pkl` files into the `models/` directory. Then, launch the rigorous evaluation to execute the 5-Fold Cross Validation and prove the $NDCG@10$ performance uplift.
```bash
python src/models.py
python src/evaluation.py
```

### Step 3: Launch the Core API Server
Once the models are generated, launch the FastAPI microservice ("The Curator"):
```bash
uvicorn src.api:app --reload
```
Navigate to `http://127.0.0.1:8000/docs` in your browser. You can now dynamically test the engine using the interactive Swagger UI (e.g., querying the `/recommend/hybrid/{user_id}` or `/recommend/popularity` routes).
