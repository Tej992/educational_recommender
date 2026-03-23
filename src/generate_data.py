import pandas as pd
import os

data = [
    {
        "course_title": "(ISC)² Systems Security Certified Practitioner (SSCP)",
        "course_organization": "ISC2",
        "course_certificate_type": "Specialization",
        "course_time": "3 - 6 Months",
        "course_rating": 4.7,
        "course_reviews_num": 492,
        "course_difficulty": "Beginner",
        "course_url": "https://www.coursera.org/specializations/sscp-training",
        "course_students_enrolled": "6,958",
        "course_skills": "['Risk Management', 'Access Control', 'Asset', 'Incident Detection and Response', 'Cloud Computing Security', 'Wireless Security', 'Security Software']",
        "course_summary": "[]",
        "course_description": "Pursue better IT security job opportunities and prove knowledge with confidence. The SSCP Professional Training Certificate shows employers you have the IT security foundation to defend against cyber attacks – and puts you on a clear path to earning SSCP certification."
    },
    {
        "course_title": ".NET FullStack Developer",
        "course_organization": "Board Infinity",
        "course_certificate_type": "Specialization",
        "course_time": "1 - 3 Months",
        "course_rating": 4.3,
        "course_reviews_num": 51,
        "course_difficulty": "Intermediate",
        "course_url": "https://www.coursera.org/specializations/dot-net-fullstack",
        "course_students_enrolled": "2,531",
        "course_skills": "['Web API', 'Web Development', 'Cascading Style Sheets (CSS)', 'HTML', 'React (Web Framework)', 'RESTful APIs', 'C# programming', 'API Integration', 'Model–View–Controller (MVC)', 'asp.net']",
        "course_summary": "['Master .NET full stack web dev', 'Dive into React frontend development']",
        "course_description": "Develop the proficiency required to design and develop comprehensive, scalable, and high-performing applications with the .NET framework via this in-depth specialization."
    },
    {
        "course_title": "21st Century Energy Transition: how do we make it work?",
        "course_organization": "University of Alberta",
        "course_certificate_type": "Course",
        "course_time": "1 - 3 Months",
        "course_rating": 4.8,
        "course_reviews_num": 62,
        "course_difficulty": "Beginner",
        "course_url": "https://www.coursera.org/learn/21st-century-energy-transition",
        "course_students_enrolled": "4,377",
        "course_skills": "[]",
        "course_summary": "['Understand the complexity of systems supplying energy', 'Evaluate the merits and the costs associated with each major available energy source']",
        "course_description": "Affordable, abundant and reliable energy is fundamental to human well-being and prosperity. For the past 150 years, more and more people have gained access to energy, primarily in the form of fossil fuels – coal, petroleum and natural gas."
    },
    {
        "course_title": "A Crash Course in Causality: Inferring Causal Effects from Observational Data",
        "course_organization": "University of Pennsylvania",
        "course_certificate_type": "Course",
        "course_time": "1 - 3 Months",
        "course_rating": 4.7,
        "course_reviews_num": 517,
        "course_difficulty": "Intermediate",
        "course_url": "https://www.coursera.org/learn/crash-course-in-causality",
        "course_students_enrolled": "39,004",
        "course_skills": "['Instrumental Variable', 'Propensity Score Matching', 'Causal Inference', 'Causality']",
        "course_summary": "[]",
        "course_description": "We have all heard the phrase “correlation does not equal causation.” What, then, does equal causation? This course aims to answer that question and more!"
    },
    {
        "course_title": "Machine Learning",
        "course_organization": "Stanford University",
        "course_certificate_type": "Course",
        "course_time": "2 - 3 Months",
        "course_rating": 4.9,
        "course_reviews_num": 160000,
        "course_difficulty": "Beginner",
        "course_url": "https://www.coursera.org/learn/machine-learning",
        "course_students_enrolled": "4,000,000",
        "course_skills": "['Machine Learning', 'Logistic Regression', 'Neural Networks']",
        "course_summary": "[]",
        "course_description": "This course provides a broad introduction to machine learning, datamining, and statistical pattern recognition."
    },
    {
        "course_title": "Python for Everybody",
        "course_organization": "University of Michigan",
        "course_certificate_type": "Specialization",
        "course_time": "3 - 6 Months",
        "course_rating": 4.8,
        "course_reviews_num": 200000,
        "course_difficulty": "Beginner",
        "course_url": "https://www.coursera.org/specializations/python",
        "course_students_enrolled": "1,500,000",
        "course_skills": "['Python Programming', 'Data Structures', 'Web Scraping']",
        "course_summary": "[]",
        "course_description": "This Specialization builds on the success of the Python for Everybody course and will introduce fundamental programming concepts including data structures, networked application program interfaces, and databases, using the Python programming language."
    },
    {
        "course_title": "Data Science Methodology",
        "course_organization": "IBM",
        "course_certificate_type": "Course",
        "course_time": "1 - 3 Months",
        "course_rating": 4.6,
        "course_reviews_num": 15000,
        "course_difficulty": "Beginner",
        "course_url": "https://www.coursera.org/learn/data-science-methodology",
        "course_students_enrolled": "80,000",
        "course_skills": "['Data Science', 'Methodology', 'CRISP-DM']",
        "course_summary": "[]",
        "course_description": "Learn the steps involved in a data science project, from problem definition to deployment."
    },
    {
        "course_title": "Deep Learning Specialization",
        "course_organization": "DeepLearning.AI",
        "course_certificate_type": "Specialization",
        "course_time": "3 - 6 Months",
        "course_rating": 4.9,
        "course_reviews_num": 120000,
        "course_difficulty": "Intermediate",
        "course_url": "https://www.coursera.org/specializations/deep-learning",
        "course_students_enrolled": "800,000",
        "course_skills": "['Deep Learning', 'Neural Networks', 'Convolutional Networks', 'RNN']",
        "course_summary": "[]",
        "course_description": "Master Deep Learning, and break into AI. In this Specialization, you will build and train neural network architectures such as Convolutional Neural Networks, Recurrent Neural Networks, LSTMs, Transformers, and learn how to make them better with strategies such as Dropout, BatchNorm, Xavier/He initialization, and more."
    }
]

df = pd.DataFrame(data)
os.makedirs("data", exist_ok=True)
df.to_csv("data/coursera_courses.csv", index=False)
print("Successfully created data/coursera_courses.csv")
