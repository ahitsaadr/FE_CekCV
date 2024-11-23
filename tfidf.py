import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
# from rouge import Rouge
from rouge_score import rouge_scorer
import google.generativeai as genai
import nltk

def tfidf(path_cv_csv, path_jd_txt):
    # Download NLTK data if not already present
    nltk.download('punkt')
    API_KEY = "AIzaSyAjVjTEYhTrCwIL_-Q0rBCV7HAOrqTuLd8"

    # Load CV data from CSV
    cv_df = pd.read_csv(path_cv_csv, encoding="utf-8")
    cvs = cv_df['cv_pelamar'].tolist()

    # Load job requirements from text file
    with open(path_jd_txt, 'r') as file:
        job_requirements = file.read()

    # Combine job requirements with CVs for TF-IDF vectorization
    documents = [job_requirements] + cvs

    # Create TF-IDF vectorizer and transform the documents
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Compute cosine similarity between job requirements and CVs
    job_tfidf = tfidf_matrix[0]
    cv_tfidfs = tfidf_matrix[1:]
    similarities = cosine_similarity(cv_tfidfs, job_tfidf)

    # Get the indices of the top N most similar CVs
    top_n = 3
    top_n_indices = np.argsort(similarities[:, 0])[-top_n:][::-1]

    # Initialize ROUGE scorer
    rouge = rouge_scorer()

    # Collect results for response generation
    results = []
    results_summary = []
    bleu_scores = []
    rouge_2_f_scores = []

    # Evaluate and display the most relevant CVs
    for rank, index in enumerate(top_n_indices, start=1):
        cv_text = cvs[index]

        # Calculate BLEU score
        reference = nltk.word_tokenize(job_requirements)
        candidate = nltk.word_tokenize(cv_text)
        bleu_score = sentence_bleu([reference], candidate)
        bleu_scores.append(bleu_score)

        # Calculate ROUGE score
        rouge_scores = rouge.get_scores(cv_text, job_requirements)[0]
        rouge_2_f = rouge_scores['rouge-2']['f']
        rouge_2_f_scores.append(rouge_2_f)

        result_summary = f"Index: {index}\nSimilarity: {similarities[index, 0]}\nROUGE-2 F-Score: {rouge_2_f}"

        # Collect results
        result = f"""
    Index: {index}
    Similarity: {similarities[index, 0]}
    ROUGE-2 F-Score: {rouge_2_f}
    CV: {cv_text}

    ========================================================

    JD: {job_requirements}"""
        results.append(result)
        results_summary.append(result_summary)

    # Calculate the average similarity, BLEU score, and ROUGE-2 F-Score of the top N CVs
    average_similarity = np.mean([similarities[index, 0] for index in top_n_indices])
    average_bleu_score = np.mean(bleu_scores)
    average_rouge_2_f_score = np.mean(rouge_2_f_scores)

    # Combine results into a single string
    results_combined = "\n\n".join(results)
    result_sum = "\n\n".join(results_summary)

    # Prepare the prompt for the Gemini API
    prompt = f"""
    {results_combined}

    =======================================================

    Berdasarkan hasil penyaringan CV berikut, kenapa dipilih dan mengapa?
    dan berikan index datanya keberapa serta nama pemilik cv nya itu siapa.

    """

    # Set up the API key and client for the Gemini API
    api_key = API_KEY
    genai.configure(api_key=api_key)
    model_answer = genai.GenerativeModel('gemini-1.5-flash')

    # Generate the response
    response = model_answer.generate_content(prompt)

    return response.text, result_sum, top_n_indices, cv_df, average_similarity, average_bleu_score, average_rouge_2_f_score
