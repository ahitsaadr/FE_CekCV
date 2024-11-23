# cv_processing.py
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import nltk
import google.generativeai as genai
import numpy as np

# Fungsi untuk memuat model
def load_models():
    sentence_embed_model = SentenceTransformer("firqaaa/indo-sentence-bert-base")
    print("MODEL TRANSFORMER: indo-sentence-bert-base ------------> READY")
    return sentence_embed_model

# Fungsi utama pemrosesan CV
def bert(path_cv_csv, path_jd_txt):
    nltk.download('punkt')
    API_KEY = "YOUR_API_KEY"  # Ganti dengan API Key kamu
    genai.configure(api_key=API_KEY)
    
    cv_df = pd.read_csv(path_cv_csv)
    cvs = cv_df['cv_pelamar'].tolist()
    
    with open(path_jd_txt, 'r') as file:
        job_requirements = file.read()
    
    model = load_models()
    cv_embeddings = model.encode(cvs, convert_to_tensor=True)
    job_embedding = model.encode(job_requirements, convert_to_tensor=True)
    
    similarities = util.pytorch_cos_sim(job_embedding, cv_embeddings).cpu().numpy()
    top_n = 3
    top_n_indices = np.argsort(similarities[0])[-top_n:][::-1]
    
    rouge = Rouge()
    results_summary = []
    
    for index in top_n_indices:
        cv_text = cvs[index]
        similarity_score = similarities[0][index]
        
        candidate_tokens = nltk.word_tokenize(cv_text)
        reference_tokens = nltk.word_tokenize(job_requirements)
        bleu_score = sentence_bleu([reference_tokens], candidate_tokens)
        
        rouge_scores = rouge.get_scores(cv_text, job_requirements)[0]
        rouge_2_f = rouge_scores['rouge-2']['f']
        
        result_summary = f"Index: {index}, Similarity: {similarity_score}, ROUGE-2 F-Score: {rouge_2_f}"
        results_summary.append(result_summary)
    
    return results_summary
