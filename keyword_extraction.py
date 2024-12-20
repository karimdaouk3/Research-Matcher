import openai
from openai import OpenAI
import os
from flask import Flask, render_template, request
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import json
from collections import Counter
import numpy as np
from tenacity import retry, wait_exponential, stop_after_attempt
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

#comment this out
#load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def tokenize(text):
    tokens = word_tokenize(text)
    return tokens


@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
def get_openai_response(prompt, model="gpt-4", max_tokens=20, temperature=0.3):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

def get_keywords_batched(titles, batch_size=20):
    all_keywords = []

    for i in range(0, len(titles), batch_size):
        batch = titles[i:i + batch_size]
        batch_text = " ".join([title if title.endswith('.') else title + '.' for title in batch])
        
        prompt = f"Based on the following research article titles, extract the main research areas or topics related to the professor's work. Provide a comma-separated list of these research areas:\n\n{batch_text}\n\nResearch areas:"

        try:
            response = get_openai_response(prompt)
            batch_keywords = response.split(',')
            all_keywords.extend([keyword.strip().capitalize() for keyword in batch_keywords])
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")

    return sorted(set(all_keywords))


def top_keywords(keywords, n):
    from sklearn.metrics.pairwise import cosine_similarity

    keyword_freq = Counter(keywords)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(keywords)

    relevance_scores = []
    for i, keyword in enumerate(keywords):
        freq = keyword_freq[keyword]
        avg_similarity = np.mean(cosine_similarity([embeddings[i]], embeddings)[0])
        relevance_scores.append(freq * avg_similarity)

    sorted_keywords_with_scores = sorted(zip(relevance_scores, keywords, embeddings), reverse=True)

    selected_keywords = []
    selected_embeddings = []
    for _, keyword, embedding in sorted_keywords_with_scores:
        if len(selected_keywords) >= n:
            break

        is_too_similar = False
        for selected_embedding in selected_embeddings:
            similarity = cosine_similarity([embedding], [selected_embedding])[0][0]
            if similarity > 0.8:  
                is_too_similar = True
                break

        if not is_too_similar:
            selected_keywords.append(keyword)
            selected_embeddings.append(embedding)

    return selected_keywords

def prof_keywords(json_data, output_data):
    for prof_id, prof_data in json_data.items():
        if "sorted_articles" in prof_data and prof_data["sorted_articles"]:
            titles = [article["title"] for article in prof_data["sorted_articles"]]
            titles_str = " ".join([title if title.endswith('.') else title + '.' for title in titles])
            
            keywords = get_keywords_batched(titles,20)
            #print(keywords)
            best_keywords = top_keywords(keywords, 5)
            #print("Top 5 keywords:", best_keywords)
            
            prof_data["keywords"] = best_keywords
            
    with open(output_data, "w") as outfile:
        json.dump(json_data, outfile, indent=4)
    


def main():
    filepath = "C:/Users/misss/OneDrive/Desktop/Research-Matcher/faculty_info_complete.json"
    output = "C:/Users/misss/OneDrive/Desktop/Research-Matcher/faculty_info_uniquekeywords.json"
    with open(filepath, "r") as json_file:
        data = json.load(json_file)
    prof_keywords(data,output)
        
if __name__ == "__main__":
    main()


