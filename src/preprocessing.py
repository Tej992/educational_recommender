import pandas as pd
import numpy as np
import os
import re

def clean_enrolled(x):
    if isinstance(x, str):
        x = x.replace(',', '')
        if 'k' in x.lower():
            return float(x.lower().replace('k', '')) * 1000
        if 'm' in x.lower():
            return float(x.lower().replace('m', '')) * 1000000
    return float(x)

def preprocess_data(input_path, output_path):
    print(f"Preprocessing {input_path}...")
    df = pd.read_csv(input_path)
    
    # Clean numerical columns
    if 'course_students_enrolled' in df.columns:
        df['course_students_enrolled'] = df['course_students_enrolled'].apply(clean_enrolled)
        
    # Fill missing values
    df['course_rating'] = df['course_rating'].fillna(df['course_rating'].mean())
    df['course_difficulty'] = df['course_difficulty'].fillna('Unknown')
    
    # Combine text features for content-based filtering
    df['combined_text'] = (
        df['course_title'].fillna('') + " " + 
        df['course_skills'].fillna('') + " " + 
        df['course_description'].fillna('')
    )
    
    # Basic text cleaning
    df['combined_text'] = df['combined_text'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    preprocess_data("data/coursera_courses.csv", "data/processed_courses.csv")
