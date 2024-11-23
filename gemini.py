import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
# from rouge import Rouge
from rouge_score import rouge_scorer
import google.generativeai as genai
import nltk
from sklearn.metrics.pairwise import cosine_similarity

def gemini(path_cv_csv, path_jd_txt):
    # Download NLTK data if not already present
    nltk.download('punkt')

    # Configure Gemini API with your API key
    API_KEY = "AIzaSyAjVjTEYhTrCwIL_-Q0rBCV7HAOrqTuLd8"
    genai.configure(api_key=API_KEY)

    # Function to embed text using Gemini API
    def embed_text(text, max_length=2048):
        if len(text) > max_length:
            text = text[:max_length]
        return genai.embed_content(model="models/text-embedding-004", content=text)["embedding"]

    # Load CV data from CSV
    cv_df = pd.read_csv(path_cv_csv, encoding="utf-8")
    cvs = cv_df['cv_pelamar'].tolist()

    # Load job requirements from text file
    with open(path_jd_txt, 'r') as file:
        job_requirements = file.read()

    # Compute embeddings for CVs and job requirements
    cv_embeddings = [embed_text(cv) for cv in cvs]
    job_embedding = embed_text(job_requirements)

    # Convert embeddings to numpy arrays for further processing
    cv_embeddings = np.array(cv_embeddings)
    job_embedding = np.array(job_embedding).reshape(1, -1)

    # Compute cosine similarity between job requirements and CV embeddings
    similarities = cosine_similarity(cv_embeddings, job_embedding)

    # Get the indices of the top N most similar CVs
    top_n = 3
    top_n_indices = np.argsort(similarities[:, 0])[-top_n:][::-1]

    # Initialize ROUGE scorer
    rouge = rouge_scorer()

    # Collect results for response generation
    results = []
    results_summary = []

    # Lists to store BLEU and ROUGE scores
    bleu_scores = []
    rouge_2_f_scores = []

    # Evaluate and display the most relevant CVs
    for rank, index in enumerate(top_n_indices, start=1):
        cv_text = cvs[index]

        # Tokenize texts
        candidate = nltk.word_tokenize(cv_text)
        reference = nltk.word_tokenize(job_requirements)

        # Calculate BLEU score
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

    # Combine results into a single string
    results_combined = "\n\n".join(results)
    result_sum = "\n\n".join(results_summary)

    # Calculate average similarity, BLEU score, and ROUGE-2 F-score for top N
    average_similarity = np.mean(similarities[top_n_indices, 0])
    average_bleu = np.mean(bleu_scores)
    average_rouge_2_f = np.mean(rouge_2_f_scores)

    # Prepare the prompt for the Gemini API
    prompt = f"""
    {results_combined}

    =======================================================

    Berdasarkan hasil penyaringan CV berikut, kenapa dipilih dan mengapa?
    dan berikan index datanya keberapa serta nama pemilik cv nya itu siapa.
    """

    # Set up the API key and client for the Gemini API
    model_answer = genai.GenerativeModel('gemini-1.5-flash')

    # Generate the response
    response = model_answer.generate_content(prompt)

    return response.text, result_sum, top_n_indices, cv_df, average_similarity, average_bleu, average_rouge_2_f
