import pandas as pd
import numpy as np
import os
import random

class PersonaGenerator:
    def __init__(self, courses_df):
        self.courses_df = courses_df
        self.personas = [
            {
                "name": "Beginner Web Dev",
                "skills": ["HTML", "CSS", "JavaScript", "Web Development"],
                "difficulty_pref": ["Beginner"],
                "n_users": 30
            },
            {
                "name": "Data Science Aspirant",
                "skills": ["Python", "Data Analysis", "Machine Learning", "Statistics"],
                "difficulty_pref": ["Beginner", "Intermediate"],
                "n_users": 40
            },
            {
                "name": "Advanced AI Researcher",
                "skills": ["Deep Learning", "Neural Networks", "Computer Vision", "NLP"],
                "difficulty_pref": ["Advanced", "Intermediate"],
                "n_users": 20
            },
            {
                "name": "Business Manager",
                "skills": ["Management", "Leadership", "Strategy", "Finance"],
                "difficulty_pref": ["Beginner", "Intermediate", "Advanced", "Mixed"],
                "n_users": 20
            }
        ]
        
    def generate_rating(self, user_skills, user_diff_pref, course_row):
        # Base rating
        rating = 3.0
        
        # Skill Match Bonus
        course_text = str(course_row.get('combined_text', '')).lower()
        skill_match = sum(1 for skill in user_skills if skill.lower() in course_text)
        if skill_match > 0:
            rating += min(2.0, skill_match * 0.5)
            
        # Difficulty Match Bonus
        if course_row['course_difficulty'] in user_diff_pref:
            rating += 0.5
        else:
            rating -= 0.5
            
        # Decouple difficulty signal by adding significant organic noise
        rating += np.random.normal(0, 1.5)
        
        # Clip
        return int(max(1, min(5, round(rating))))

    def generate_dataset(self, n_interactions_per_user=10):
        all_interactions = []
        user_id_counter = 0
        
        print("Generating persona-based interactions...")
        
        for persona in self.personas:
            print(f"  - Simulating {persona['n_users']} users for persona: {persona['name']}")
            for _ in range(persona['n_users']):
                user_id = user_id_counter
                user_id_counter += 1
                
                # Select courses that are somewhat relevant to this persona to rate
                # (Real users don't pick random courses, they pick things they are interested in)
                # We simulate this by weighting selection towards their skills
                
                # Simple approach: Mix of random courses and keyword-matched courses
                relevant_courses = self.courses_df[
                    self.courses_df['combined_text'].str.contains('|'.join(persona['skills']), case=False, na=False)
                ]
                
                if len(relevant_courses) > 0:
                    # 70% chance to pick a relevant course, 30% random exploration
                    n_relevant = int(n_interactions_per_user * 0.7)
                    n_random = n_interactions_per_user - n_relevant
                    
                    selected_relevant = relevant_courses.sample(min(len(relevant_courses), n_relevant), replace=True)
                    selected_random = self.courses_df.sample(n_random, replace=True)
                    user_courses = pd.concat([selected_relevant, selected_random])
                else:
                    user_courses = self.courses_df.sample(n_interactions_per_user)
                
                for _, course in user_courses.iterrows():
                    rating = self.generate_rating(persona['skills'], persona['difficulty_pref'], course)
                    all_interactions.append({
                        'user_id': user_id,
                        'course_title': course['course_title'],
                        'rating': rating,
                        'persona': persona['name'] # Keep for analysis
                    })
                    
        return pd.DataFrame(all_interactions)

def generate_interactions(courses_path, output_path):
    print(f"Loading courses from {courses_path}...")
    courses_df = pd.read_csv(courses_path)
    
    generator = PersonaGenerator(courses_df)
    interactions_df = generator.generate_dataset()
    
    # Remove duplicates
    interactions_df = interactions_df.drop_duplicates(subset=['user_id', 'course_title'])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    interactions_df.to_csv(output_path, index=False)
    print(f"Saved {len(interactions_df)} interactions to {output_path}")

if __name__ == "__main__":
    generate_interactions("data/processed_courses.csv", "data/user_interactions.csv")
