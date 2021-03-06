import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('modellr.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print(request.form.values())
    int_features = [float(x) for x in request.form.values()]
    final_features = [int_features]
    prediction = model.predict(final_features)
    if(prediction==1):
        return render_template('index.html', prediction_text='Employee Will leave the company')
    else:
       return render_template('index.html', prediction_text='Employee Will be in the company')

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=False)