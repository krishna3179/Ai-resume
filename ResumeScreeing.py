import os
import spacy
import PyPDF2
import docx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from collections import Counter

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Predefined skillset (can be expanded)
skills_list = ['python', 'machine learning', 'data science', 'nlp', 'tensorflow', 'keras', 'pandas', 'numpy', 'scikit-learn']

# 1. Resume Parsing Functions
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ''

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        return ''

def parse_resume(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        print(f"Unsupported file format: {file_path}")
        return ''

# 2. Preprocessing and Similarity
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

def get_similarity(resume_text, job_desc):
    processed_resume = preprocess(resume_text)
    processed_jd = preprocess(job_desc)
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([processed_resume, processed_jd])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)

# 3. Skill Extraction
def extract_skills(resume_text):
    doc = nlp(resume_text.lower())
    resume_skills = [token.text for token in doc if token.text in skills_list]
    return list(set(resume_skills))

# 4. Named Entity Recognition (Experience, Education, Job Titles)
def extract_experience_and_education(resume_text):
    doc = nlp(resume_text)
    job_titles = []
    organizations = []
    
    for ent in doc.ents:
        if ent.label_ == 'ORG':  # Organization
            organizations.append(ent.text)
        if ent.label_ == 'WORK_OF_ART':  # Job Titles (simplified example)
            job_titles.append(ent.text)
    
    return job_titles, organizations

# 5. Sentiment Analysis for Cover Letter (if available)
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Range: -1 (negative) to 1 (positive)

# 6. Scoring and Ranking Resumes
def score_resumes(resume_folder, job_description):
    scores = []
    
    for filename in os.listdir(resume_folder):
        if not (filename.endswith('.pdf') or filename.endswith('.docx')):
            continue

        file_path = os.path.join(resume_folder, filename)
        resume_text = parse_resume(file_path)
        
        if resume_text.strip():
            # Get similarity score
            similarity = get_similarity(resume_text, job_description)
            
            # Extract skills from resume
            skills = extract_skills(resume_text)
            skills_match = len(skills) / len(skills_list) * 100
            
            # Extract job titles and organizations (Experience)
            job_titles, organizations = extract_experience_and_education(resume_text)
            experience_match = len(job_titles) / 5 * 100  # Example: Matching experience with job titles
            
            # Analyze sentiment of the resume (if cover letter exists in the resume text)
            sentiment_score = analyze_sentiment(resume_text)  # Simplified sentiment calculation
            
            # Combine scores
            total_score = (similarity + skills_match + experience_match + (sentiment_score * 100)) / 4
            
            scores.append({
                'Candidate': filename,
                'Match %': similarity,
                'Skills Match': skills_match,
                'Experience Match': experience_match,
                'Sentiment Score': sentiment_score * 100,
                'Total Score': round(total_score, 2)
            })
        else:
            print(f"Empty or unreadable resume: {filename}")
    
    df = pd.DataFrame(scores)
    df = df.sort_values(by='Total Score', ascending=False).reset_index(drop=True)
    return df

# 7. Run the Program
if __name__ == "__main__":
    job_description = """
    We are looking for a Data Scientist with experience in Python, Machine Learning,
    Natural Language Processing, and resume parsing tools like spaCy or NLTK.
    """

    resume_folder = "C:\\Users\\Golla Krishna\\OneDrive\\Documents\\resume"  # Place your .pdf/.docx files inside this folder
    print("Processing resumes...\n")
    ranked_df = score_resumes(resume_folder, job_description)

    print("\n--- Ranked Candidates ---\n")
    print(ranked_df)

    # Optional: save to Excel or CSV
    ranked_df.to_csv("ranked_candidates.csv", index=False)
    print("\nResults saved to 'ranked_candidates.csv'")
