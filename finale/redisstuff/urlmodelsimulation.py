import os
import sys
import csv
import re
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import seaborn

from urllib.parse import urlparse

from ast import literal_eval as make_tuple

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error as mse
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler


vectorizer = HashingVectorizer(stop_words='english',analyzer='word',n_features=2**10)

def url_tokenizer():
    pass


def getUrlPath(url):
    parsed = urlparse(url)
    path = parsed.path
    return path


def getUrlQuery(url):
        parsed = urlparse(url)
        query = parsed.query
        if query:
                return query
        else:
                return ' '


def extract_features(tuple_list):
    '''
    df = pd.DataFrame(columns=['amount_parameters','query_length','url_length','path_length','average_deviation','average_ratio'])
    url_length = len(url)
    url_path = getUrlPath(url)
    path_length = len(url_path)

#    titel_length = 0
#    tokens = url_path.split('/')
#    tokens = [t for t in tokens if  t]
#    if len(tokens) > 1:
#        comp = re.split('[-.,!?]',tokens[1])
#        if len(comp) > 2:
#            titel_length = len(tokens[1])
        
    query = getUrlQuery(url)
    amount_parameters = 0
    query_length = 0
    if query:
        parameters = query.split('&')
        amount_parameters = len(parameters)
        query_length = sum(len(p) for p in parameters)

    dev = url_length - float(avg)
    ratio = url_length / float(avg)

    df.loc[0] = [amount_parameters,query_length,url_length,path_length,dev,ratio]
    '''

    docs = []
    for t in tuple_list:
        doc = t[0] + ' ' + t[1] + ' ' + t[2] # url anchor and text
        docs.append(doc.strip())
    doc_term_matrix = vectorizer.fit_transform(docs)
    #df_final = pd.concat([df,pd.DataFrame(doc_term_matrix.toarray(),index=df.index)],axis=1)
    return doc_term_matrix
