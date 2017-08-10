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

import balancedsample

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
        url_path = getUrlPath(url)
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

path = 'dataframes/'

for file in os.listdir(path):
    if 'target' in file or 'shop' in file:
        continue
    print(file)
    domain = file[0:file.rindex('.')]

    df = pd.DataFrame().from_csv(path + file)
    df = df.dropna()
    steps = np.arange(0.01, 1.0, 0.01)
    errors = np.array([]) # collects all mse per domain
    abs_sample_size_pos = np.array([]) # collects samplesize for final statistics
    abs_sample_size_neg = np.array([])
    for size in steps:
        print ('current step ' + str(size))
        sample = balancedsample.balanced_subsample(df.drop(['label'],axis=1).as_matrix(),df['label'].as_matrix(),subsample_size=size)
        data = np.hstack((sample[0],sample[1].reshape(len(sample[1]),1)))
        df_sample = pd.DataFrame(data=data,columns=df.columns)
        df_sample = extract_features(df_sample)
        df_sample = df_sample.dropna()
        
        df_sample.loc[df_sample['prediction'] >= 0.5,'label'] = 1
        df_sample.loc[df_sample['prediction'] < 0.5, 'label'] = 0

        abs_sample_size_pos = np.append(abs_sample_size_pos, len(df_sample['prediction'] >= 0.5))
        abs_sample_size_neg = np.append(abs_sample_size_neg, len(df_sample['prediction'] < 0.5))

        X = df_sample.drop(['url','prediction','domain','label','url_path'],axis=1)
        y = df_sample['label']
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        robust_scaler = RobustScaler()
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = df['prediction'].iloc[train_index], df['prediction'].iloc[test_index]
            #y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = lr()
            Xtr_r = robust_scaler.fit_transform(X_train)
            model.fit(Xtr_r, y_train)
            Xte_r = robust_scaler.transform(X_test)
            prediction = model.predict(Xte_r)

        joblib.dump(model, experiment_name + '/pickles/' + domain  + '.pkl')

        error = mse(y_test,prediction)
        errors = np.append(errors, error)
    data_final = np.column_stack((np.array([domain for s in steps]),steps,errors,abs_sample_size_pos,abs_sample_size_neg))
    df_error = pd.DataFrame(data=data_final,columns=['domain','size','error','size_pos','size_neg'])
    df_error.to_csv(experiment_name + '/' + domain + '-error.csv')
    data_plot = np.column_stack((steps,errors))
    error_plot = seaborn.lmplot(x='size',y='error',data=pd.DataFrame(data=data_plot,columns=['size','error']), fit_reg=False, palette=seaborn.xkcd_rgb["pale red"], markers='X')
    #seaborn.plt.ylim(-0.5,1.5)
    error_plot.savefig(experiment_name + '/' + domain + '-error.png')
