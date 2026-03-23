import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure data directory exists
os.makedirs("data", exist_ok=True)
os.makedirs("output", exist_ok=True)

def perform_eda(file_path):
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    print("\n--- Dataset Info ---")
    print(df.info())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Basic Statistics ---")
    print(df.describe())

    print("\n--- Sample Data ---")
    print(df.head())

    # Visualizations
    plt.figure(figsize=(10, 6))
    sns.histplot(df['course_rating'], bins=20, kde=True)
    plt.title('Distribution of Course Ratings')
    plt.savefig('output/rating_distribution.png')
    print("\nSaved rating_distribution.png")

    plt.figure(figsize=(10, 6))
    sns.countplot(y='course_difficulty', data=df, order=df['course_difficulty'].value_counts().index)
    plt.title('Count of Courses by Difficulty')
    plt.savefig('output/difficulty_counts.png')
    print("Saved difficulty_counts.png")

if __name__ == "__main__":
    perform_eda("data/coursera_courses.csv")
