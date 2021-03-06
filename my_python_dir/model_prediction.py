from sklearn import datasets
from joblib import load
import numpy as np
import json
from flask import send_file
import os
from flask import request

UPLOAD_FOLDER='.'

#load the model

my_model = load('log_model.pkl')

def upload(filename):
    f = request.files['file']
    f.save(filename)
    return "Upload success"

def download(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    return send_file(path, as_attachment=True)

def heartfailure(arg1):
    dummy = np.array(arg1)
    dummyT = dummy.reshape(1,-1)
    r = dummy.shape
    t = dummyT.shape
    r_str = json.dumps(r)
    t_str = json.dumps(t)
    prediction = my_model.predict(dummyT)
    #name_str = json.dumps(prediction)
    str = [t_str, r_str, np.array_str(prediction)]
    return str
    
