# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:25:13 2022

@author: no1ca
"""

from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

def load_models():
    file_name = "RF_model.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model

@app.route("/", methods=['GET'])
def predict():
    #if request.method == "POST":
        #deviceID = request.form("deviceID")
    return render_template("index.html")

@app.route("/sub", methods = ['POST'])
def submit():
    if request.method == "POST":
        request_json = request.get_json()
        x = [1,2.22045e-16,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	1]
        x_in = np.array(x).reshape(1,-1)
        # load model
        model = load_models()
        prediction = model.predict(x_in)[0]    
        print(prediction)
    return render_template("sub.html", pred = prediction)

if __name__ == "__main__":
    app.run()