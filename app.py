from flask import Flask, request, render_template, jsonify
import re
import nltk
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from transformers import pipeline
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
nlp = spacy.load("en_core_web_sm")


nltk.download('punkt')
nltk.download('stopwords')


summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="pt")#summarization pipeline

def txt_processing(doc_text):
    doc_text = doc_text.lower()  # Convert to lower case
    doc_text = BeautifulSoup(doc_text, "html.parser").text  # Remove HTML tags
    doc_text = re.sub(r'\d+', '', doc_text)  # Remove digits
    doc_text = re.sub(r'http\S+|www\S+|https\S+', '', doc_text, flags=re.MULTILINE)  # Remove links
    doc_text = re.sub(r'\b\w\.\b', '', doc_text)
    doc_text = re.sub(r'\(.*?\)', '', doc_text)  # Removes () and their contents

    tokens = word_tokenize(doc_text)  # Tokenize

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords

    cleaned_doc = ' '.join(tokens)  # Join as a string

    return tokens, cleaned_doc

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        data = request.get_json()
        text = data.get('text')
        max_length = int(data.get('max_length'))
        min_length = int(data.get('min_length'))

        logging.debug(f"Received text: {text}")
        logging.debug(f"Max length: {max_length}, Min length: {min_length}")

        tokens, cleaned_doc = txt_processing(text)

        # Process the text with spaCy
        doc = nlp(cleaned_doc)

        # Allowed parts of speech
        allowed_pos = ['ADJ', 'VERB', 'NOUN', 'INTJ']

        tokens_1 = [token.text for token in doc if token.pos_ in allowed_pos and token.text not in punctuation]

        result_text = " ".join(tokens_1)

        # Summarize the text
        summary = summarizer(result_text, max_length=max_length, min_length=min_length, do_sample=False)
        logging.debug(f"Summary: {summary}")

        return jsonify({"summary": summary[0]['summary_text']})

    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        return jsonify({"error": "An error occurred while summarizing the text."}), 500

if __name__ == '__main__':
    app.run(debug=False)
