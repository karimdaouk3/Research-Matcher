from flask import Flask, render_template, request
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from openai import OpenAI
import logging
from datetime import datetime
import os
# from waitress import serve

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

app = Flask(__name__)

# Load the JSON file back into a DataFrame
with open('faculty_data.json', 'r') as json_file:
    loaded_data_list = json.load(json_file)

df = pd.DataFrame(loaded_data_list)

# Initialize the sentence embedder model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

logging.basicConfig(filename='app_log.json', level=logging.INFO)

@app.route('/', methods=['GET', 'POST'])
def index():
    generate_title_checkbox_checked = False

    if request.method == 'POST':

        log_entry = {'text_content': request.form['proposal'], 'timestamp': str(datetime.now())}
        logging.info(json.dumps(log_entry))

        proposal = request.form['proposal']
        result_count = int(request.form['resultCount'])
        generate_title_checkbox = 'generateTitle' in request.form
        generate_title_checkbox_checked = generate_title_checkbox

        proposal_embedding = model.encode([proposal])[0]

        # Calculate dot product for each row
        df['Dot Product'] = df['Research Embedding'].apply(lambda x: np.dot(proposal_embedding, x) if x is not None else None)

        df_sorted = df.sort_values(by='Dot Product', ascending=False)

        top_n_rows = df_sorted.head(result_count)

        top_n_rows.loc[:, 'Name'] = top_n_rows['Name'].apply(lambda x: x.title() if x else x)
        top_n_rows.loc[:, 'Title'] = top_n_rows['Title'].apply(lambda x: x.title() if x else x)

        results = top_n_rows[['Name', 'Title', 'Research Summary']].to_dict(orient='records')

        generated_title = None

        if generate_title_checkbox:
            generated_title = generate_title(proposal)

        return render_template('index.html', proposal=proposal, results=results, resultCount=result_count, generatedTitle=generated_title, generateTitleCheckbox=generate_title_checkbox_checked)

    return render_template('index.html', generateTitleCheckbox=generate_title_checkbox_checked)

def generate_title(proposal):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that generates a simple title for a research paper given a research proposal. Do not include the word title in the response."},
            {"role": "user", "content": proposal}
        ],
        temperature=0.5,
        max_tokens=50,
    )

    return response.choices[0].message.content.strip()

if __name__ == '__main__':
    app.run(port=7070, debug=False)
    # serve(app, host='0.0.0.0', port=80)