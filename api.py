# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 17:27:02 2021

@author: artpe
"""

from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json


app = Flask(__name__)


@app.route('/api/', methods=['POST'])
def makecalc():
    data = request.get_json()
    prediction = np.array2string(model.predict(data))

    return jsonify(prediction)

if __name__ == '__main__':
    modelfile = 'lgbm.pkl'
    model = p.load(open(modelfile, 'rb'))
    app.run(debug=True, host='0.0.0.0')