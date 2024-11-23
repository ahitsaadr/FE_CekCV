import pandas as pd
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import nltk
import google.generativeai as genai
import numpy as np

# Fungsi untuk memastikan bahwa tokenizer 'punkt' terinstal
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading 'punkt' tokenizer...")
        nltk.download('punkt')
        print("'punkt' tokenizer downloaded successfully!")

# Fungsi untuk memuat model tanpa caching streamlit
def load_models():
    sentence_embed_model = SentenceTransformer("firqaaa/indo-sentence-bert-base")
    print("MODEL TRANSFORMER: indo-sentence-bert-base ------------> READY")
    return sentence_embed_model

def bert(saved_cv_paths, job_description_path):
    # Pastikan NLTK data (punkt tokenizer) terunduh
    download_nltk_resources()

    # Configure Gemini API with your API key
    API_KEY = "YOUR_API_KEY"  # Ganti dengan kunci API yang sesuai
    genai.configure(api_key=API_KEY)

    # Load job requirements from text file
    with open(job_description_path, 'r', encoding='ISO-8859-1', errors='ignore') as file:
        job_requirements = file.read()

    # Initialize Sentence Transformer model for BERT embeddings
    model = load_models()

    # Membaca semua CV dari daftar path
    cvs = []
    for cv_path in saved_cv_paths:
        with open(cv_path, 'r', encoding='ISO-8859-1', errors='ignore') as file:
            cvs.append(file.read())

    # Compute embeddings for CVs and job requirements
    cv_embeddings = model.encode(cvs, convert_to_tensor=True)
    job_embedding = model.encode(job_requirements, convert_to_tensor=True)

    # Calculate cosine similarities
    similarities = util.pytorch_cos_sim(job_embedding, cv_embeddings).cpu().numpy()

    # Get the indices of the top N most similar CVs
    top_n = 3
    top_n_indices = np.argsort(similarities[0])[-top_n:][::-1]

    # Initialize ROUGE scorer
    rouge = rouge_scorer.RougeScorer(['rouge-1', 'rouge-2', 'rouge-l'], use_stemmer=True)

    # Collect results for response generation
    results = []
    results_summary = []
    top_n_similarities = []
    top_n_bleu_scores = []
    top_n_rouge_scores = []

    # Evaluate and display the most relevant CVs
    for rank, index in enumerate(top_n_indices, start=1):
        cv_text = cvs[index]
        similarity_score = similarities[0][index]
        top_n_similarities.append(similarity_score)

        # Tokenize texts
        candidate_tokens = nltk.word_tokenize(cv_text)
        reference_tokens = nltk.word_tokenize(job_requirements)
        
        # Calculate BLEU score
        bleu_score = sentence_bleu([reference_tokens], candidate_tokens)
        top_n_bleu_scores.append(bleu_score)
        
        # Calculate ROUGE score
        rouge_scores = rouge.score(cv_text, job_requirements)
        rouge_2_f = rouge_scores['rouge-2'].fmeasure
        top_n_rouge_scores.append(rouge_2_f)
        
        result_summary = f"Index: {index}\nSimilarity: {similarity_score}\nROUGE-2 F-Score: {rouge_2_f}"
        
        result = f"""
    Index: {index}
    Similarity: {similarity_score}
    ROUGE-2 F-Score: {rouge_2_f}
    CV: {cv_text}

    ========================================================

    JD: {job_requirements}"""
        results.append(result)
        results_summary.append(result_summary)

    # Calculate the average similarity, BLEU, and ROUGE scores
    average_similarity = np.mean(top_n_similarities)
    average_bleu_score = np.mean(top_n_bleu_scores)
    average_rouge_score = np.mean(top_n_rouge_scores)

    # Combine results into a single string
    results_combined = "\n\n".join(results)
    result_sum = "\n\n".join(results_summary)

    # Prepare the prompt for the Gemini API
    prompt = f"""
    {results_combined}

    =======================================================

    Berdasarkan hasil penyaringan CV berikut, kenapa dipilih dan mengapa?
    dan berikan index datanya keberapa.
    """

    # Generate the response with Gemini API
    response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
    
    return response.text, result_sum, top_n_indices, average_similarity, average_bleu_score, average_rouge_score