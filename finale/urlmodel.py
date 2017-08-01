import os
import sys
import csv
import re
import sys
import pandas as pd
import numpy as np

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


def extract_features(df):
    url_lengths = []
    path_length = []
    for i,row in df.iterrows():
        url = row[0]
        try:
            url_lengths.append(len(url))
        except:
            print (url)
        url_path = .getUrlPath(url)
        df.loc[i,'url_path'] = url_path
        path_length.append(len(url_path))

        titel_length = 0
        tokens = url_path.split('/')
        tokens = [t for t in tokens if  t]
        if len(tokens) > 1:
            comp = re.split('[-.,!?]',tokens[1])
            if len(comp) > 2:
                titel_length = len(tokens[1])
        df.loc[i,'title_length'] = titel_length
        
        query = getUrlQuery(url)
        amount_parameters = 0
        query_length = 0
        if query:
            parameters = query.split('&')
            amount_parameters = len(parameters)
            query_length = sum(len(p) for p in parameters)
        df.loc[i,'amount_parameters'] = amount_parameters
        df.loc[i, 'query_length'] = query_length

    df['url_length'] = pd.Series(url_lengths,index=df.index)
    df['path_length'] = pd.Series(path_length, index=df.index)

    avg = sum(l for l in url_lengths) / len(url_lengths)
    dev = [l - avg for l in url_lengths]
    df['average_deviation'] = pd.Series(dev,index=df.index)

    ratio = [l/avg for l in url_lengths]
    df['average_ratio'] = pd.Series(ratio, index=df.index)

    #vectorizer = HashingVectorizer(stop_words='english',analyzer='word',n_features=2**15)
    #doc_term_matrix = vectorizer.fit_transform(df['url'])
    #df_final = pd.concat([df,pd.DataFrame(doc_term_matrix.toarray(),index=df.index)],axis=1)
    return df


#pd.options.display.max_colwidth = 100
experiment_name = sys.argv[1]
path = experiment_name + '/data/'
output = []

for file in os.listdir(path):
    print(file)
    domain = file[0:file.index('.')]

    df = pd.DataFrame.from_csv(path + file)
    df = df.dropna()

    df = extract_features(df)
    df = df.dropna()
    df.loc[df['prediction'] >= 0.5,'label'] = 1
    df.loc[df['prediction'] < 0.5, 'label'] = 0
    print (len(df[df['prediction'] >= 0.5]))
    print (len(df[df['prediction'] < 0.5]))
    X = df.drop(['url','prediction','domain','label','url_path'],axis=1)
    y = df['label']
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    for train_index, test_index in sss.split(X, y):
        '''
        calculate url features based on training data of positive class
        '''
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = df['prediction'].iloc[train_index], df['prediction'].iloc[test_index]

        model = lr()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

    joblib.dump(model, experiment_name + '/pickles/' + domain + '.pkl')
    
    error = mse(y_test,prediction)
    output.append([domain,len(df[(df.index.isin(test_index)) & (df['label'] == 1)]),len(df[(df.index.isin(test_index)) & (df['label'] == 0)]),error])

f = open(experiment_name + '/results.csv','w')
writer = csv.writer(f,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)

writer.writerow(['domain','#products','#nonproducts','mse'])

for out in output:
    writer.writerow(out)
f.close()
