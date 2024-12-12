from flask import Flask, render_template, request
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from openai import OpenAI
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
from waitress import serve

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

# Load the JSON file back into a DataFrame
with open('C:/Users/misss/OneDrive/Desktop/Research-Matcher/faculty_data.json', 'r') as json_file:
    loaded_data_list = json.load(json_file)
    
#NEW
with open('C:/Users/misss/OneDrive/Desktop/Research-Matcher/faculty_info_keywords.json', 'r') as json_file:
    faculty_keywords = json.load(json_file)
    
df = pd.DataFrame(loaded_data_list)

#NEW
#keywords_df = pd.DataFrame(faculty_keywords)
keywords_df = pd.DataFrame.from_dict(faculty_keywords, orient='index')
keywords_df.reset_index(inplace=True)
keywords_df.rename(columns={'index': 'id'}, inplace=True)

# Initialize the sentence embedder model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', device='cpu')

logging.basicConfig(filename='app_log.txt', level=logging.INFO)

@app.route('/', methods=['GET', 'POST'])
def index():
    global df
    global keywords_df
    generate_title_checkbox_checked = False

    if request.method == 'POST':
        
        log_entry = {'text_content': request.form['proposal'], 'timestamp': str(datetime.now())}
        logging.info(json.dumps(log_entry))

        proposal = request.form['proposal']
        result_count = int(request.form['resultCount'])
        generate_title_checkbox = 'generateTitle' in request.form
        generate_title_checkbox_checked = generate_title_checkbox

        # Assuming 'proposal_embedding' contains the 1D array of the embedded proposal vector
        proposal_embedding = model.encode([proposal])[0]

        # Calculate dot product for each row
        df['Dot Product'] = df['Research Embedding'].apply(lambda x: np.dot(proposal_embedding, x) if x is not None else None)

        # Sort DataFrame based on dot product scores in descending order
        df_sorted = df.sort_values(by='Dot Product', ascending=False)

        # Select top N rows based on the result count
        top_n_rows = df_sorted.head(result_count)
        
        #NEW
        top_n_rows['Name_lower'] = top_n_rows['Name'].str.lower()
        keywords_df['Name_lower'] = keywords_df['name'].str.lower()
        top_n_rows = pd.merge(top_n_rows, keywords_df[['Name_lower', 'keywords']], on='Name_lower', how='left')

        # Extract relevant information for rendering
        top_n_rows.loc[:, 'Name'] = top_n_rows['Name'].apply(lambda x: x.title() if x else x)
        top_n_rows.loc[:, 'Title'] = top_n_rows['Title'].apply(lambda x: x.title() if x else x)

        
        df['Name_lower'] = df['Name'].str.lower()
        keywords_df['Name_lower'] = keywords_df['name'].str.lower()
        df = pd.merge(df, keywords_df[['Name_lower', 'keywords']], on='Name_lower', how='left')
        # Convert to dictionary
        #print(df)
        #NEW
        results = []
        for _, row in top_n_rows.iterrows():
            result = {
                'Name': row['Name'],
                'Title': row['Title'],
                'Research Summary': row['Research Summary']
            }
            if 'keywords' in row and isinstance(row['keywords'], list) and len(row['keywords']) > 0:
                result['keywords'] = row['keywords']  # Keep it as a list
            else:
                result['keywords'] = None
            results.append(result)
        
        """
        results = top_n_rows[['Name', 'Title', 'Research Summary', 'keywords']].to_dict(orient='records')
        for result in results:
            result['keywords'] = result.get('keywords', [])
            if isinstance(result['keywords'], str):
                result['keywords'] = [keyword.strip() for keyword in result['keywords'].split(',')]
        """
        generated_title = None

        if generate_title_checkbox:
            # Generate a title using OpenAI chat-based language model
            generated_title = generate_title(proposal)

        # Render the template
        return render_template('index.html', proposal=proposal, results=results, resultCount=result_count, generatedTitle=generated_title, generateTitleCheckbox=generate_title_checkbox_checked)

    # For GET requests, set the checkbox to unchecked by default
    return render_template('index.html', generateTitleCheckbox=generate_title_checkbox_checked)

def generate_title(proposal):
    # Call the OpenAI API to generate a title using chat-based language model
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant that generates a simple title for a research paper given a research proposal. Do not include the word title in the response."},
            {"role": "user", "content": proposal}
        ],
        temperature=0.5,
        max_tokens=50,
    )

    return response.choices[0].message.content.strip()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
    # serve(app, host='0.0.0.0')
