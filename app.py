import numpy as np
import spacy
import pandas as pd
from flask import Flask, render_template, request, send_file
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
vect = pickle.load(open('vec.pkl', 'rb'))
model = pickle.load(open('mdel.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict1', methods=['GET', 'POST'])
def predict1():

    message = request.form.get('msg')
    data = [message]
    inputOfData = vect.transform(data).toarray()

    predictIt = model.predict(inputOfData)
    return render_template('index.html', pred='Your entered message is {}'.format(predictIt))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        fileUpload = request.files['file']
        
       
        if not fileUpload:
            return render_template('index.html', error_message='File not uploaded !! Please upload it..')
        
        try:
            df = pd.read_csv(fileUpload, encoding="ISO-8859-1")
        except:
            return render_template('index.html', error_message='File format is invalid upload CSV here')
        
        en = spacy.load('en_core_web_sm')
        sw_spacy = en.Defaults.stop_words
        df['clean'] = df['v2'].apply(lambda x: ' '.join(
            [word for word in x.split() if word.lower() not in (sw_spacy)]))
        
        count = vect.transform(df['clean'])
        
        
        
        predictions = model.predict(count)
        
        df['Prediction'] = predictions
        
        df.to_csv('result.csv', index=False)
        
        return send_file('result.csv', as_attachment=True)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
