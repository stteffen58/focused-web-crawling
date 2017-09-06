import os
import sys
import csv
import re
import sys
import csv
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import seaborn
import ast
import time

from urllib.parse import urlparse

from ast import literal_eval as make_tuple

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
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


def extract_features(df,domain):
    '''
    url_lengths = []
    path_length = []
    df_url = pd.DataFrame(columns=['amount_parameters','query_length','url_length','path_length','average_deviation','average_ratio'])
    index = df.columns.get_loc('url')
    for i,row in df.iterrows():
        url = row[index]
        try:
            url_lengths.append(len(url))
        except:
            print (url)
        url_path = getUrlPath(url)
        #df.loc[i,'url_path'] = url_path
        path_length.append(len(url_path))

#        titel_length = 0
#        tokens = url_path.split('/')
#        tokens = [t for t in tokens if  t]
#        if len(tokens) > 1:
#            comp = re.split('[-.,!?]',tokens[1])
#            if len(comp) > 2:
#                titel_length = len(tokens[1])
#        df.loc[i,'title_length'] = titel_length
        
        query = getUrlQuery(url)
        amount_parameters = 0
        query_length = 0
        if query:
            parameters = query.split('&')
            amount_parameters = len(parameters)
            query_length = sum(len(p) for p in parameters)
        df_url.loc[i,'amount_parameters'] = amount_parameters
        df_url.loc[i, 'query_length'] = query_length

    df_url['url_length'] = pd.Series(url_lengths,index=df.index)
    df_url['path_length'] = pd.Series(path_length, index=df.index)

    avg = float(float(sum(l for l in url_lengths)) / float(len(url_lengths)))
    # save avg to file
    with open('misc/' + domain + '-avg','w') as f:
        f.write(str(avg))

    dev = [l - avg for l in url_lengths]
    df_url['average_deviation'] = pd.Series(dev,index=df.index)

    ratio = [float(l)/float(avg) for l in url_lengths]
    df_url['average_ratio'] = pd.Series(ratio, index=df.index)
    '''
    df['text'] = df.apply(lambda row: ast.literal_eval(row['text'])[0],axis=1)    
    #combined = df['url'] + ' ' + df['text']
    vectorizer = HashingVectorizer(stop_words='english',analyzer='word',n_features=2**10)
    doc_term_matrix = vectorizer.fit_transform(df['url'])
    #df_final = pd.concat([df[['prediction','label','url']],pd.DataFrame(doc_term_matrix.toarray(),index=df.index)],axis=1)
    return doc_term_matrix
    #return pd.concat([df,df_url],axis=1)#,pd.DataFrame(doc_term_matrix.toarray(),index=df.index)],axis=1)


#pd.options.display.max_colwidth = 100
experiment_name = sys.argv[1]

path = 'dataframes/' + experiment_name + '/'

for file in os.listdir(path):
    if 'target' in file or 'shop' in file:
        continue
    print(file)
    domain = file[0:file.rindex('.')]
    df = pd.read_csv(path + file, quotechar='|',)
    df = df.dropna()
    df = df.drop(['Unnamed: 0'],axis=1)
   # steps = np.arange(0.3, 0.4, 0.1)
    steps = [0.3]
    errors = np.array([]) # collects all mse per domain
    abs_sample_size_pos = np.array([]) # collects samplesize for final statistics
    abs_sample_size_neg = np.array([])
    for size in steps:
        print ('current step ' + str(size))
        sample = balancedsample.balanced_subsample(df.drop(['label'],axis=1).as_matrix(),df['label'].as_matrix(),subsample_size=size)
        data = np.hstack((sample[0],sample[1].reshape(len(sample[1]),1)))
        df_sample = pd.DataFrame(data=data,columns=df.columns)
        doc_term_matrix = extract_features(df_sample,domain)
        df_sample = df_sample.dropna()
        
        #df_sample.loc[df_sample['prediction'] > 0.5,'label'] = 1
        #df_sample.loc[df_sample['prediction'] <= 0.5, 'label'] = 0

        abs_sample_size_pos = np.append(abs_sample_size_pos, len(df_sample['prediction'] >= 0.5))
        abs_sample_size_neg = np.append(abs_sample_size_neg, len(df_sample['prediction'] < 0.5))
        #X = df_sample.drop(['url','prediction','label'],axis=1)
        X = df_sample.drop(['url','prediction','domain','label','name', 'rating_count', 'rating_value', 'has_image', 'identifier_count', 'list_length',
               'list_rows', 'table_length', 'table_rows', 'description_length', 'price', 'text'],axis=1)
        y = df_sample['label']
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
#        robust_scaler = StandardScaler()
        for train_index, test_index in sss.split(X, y):
            # X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            X_train, X_test = doc_term_matrix[train_index], doc_term_matrix[test_index]
#            y_train, y_test = df['prediction'].iloc[train_index], df['prediction'].iloc[test_index]
            y_train, y_test = df['label'].iloc[train_index], df['label'].iloc[test_index]
            urls = df_sample.iloc[train_index]['url']
            model = GaussianNB()
            model.fit(X_train.toarray(), y_train)
            prediction = model.predict(X_test.toarray())
            probas = model.predict_proba(X_test.toarray())
#            for i,p in enumerate(probas):
#                print (p)
#                time.sleep(1)
#                if i == 10:
#                    break
            # urls which are ignored during simulation
            urls = df_sample.iloc[train_index]['url']
            f = open('misc/' + domain + '-train.csv','w')
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(urls)
            f.close()

        joblib.dump(model, experiment_name + '/pickles/np-url-' + domain  + '.pkl')
        
        #error = mse(y_test,prediction)
        #errors = np.append(errors, error)
        '''
        predict_labels = []
        print (prediction)
        for e in prediction:
            if e[1] > 0.5:
                predict_labels.append(1)
            else:
                predict_labels.append(0)
        print (predict_labels)
        '''
        precision = precision_score(df['label'].iloc[test_index],prediction)
        recall = recall_score(df['label'].iloc[test_index],prediction)
        print (precision,recall)

#    data_final = np.column_stack((np.array([domain for s in steps]),steps,errors,abs_sample_size_pos,abs_sample_size_neg))
#    df_error = pd.DataFrame(data=data_final,columns=['domain','size','error','size_pos','size_neg'])
#    df_error.to_csv(experiment_name + '/' + domain + '-error.csv')
#    data_plot = np.column_stack((steps,errors))
#    error_plot = seaborn.lmplot(x='size',y='error',data=pd.DataFrame(data=data_plot,columns=['size','error']), fit_reg=False, palette=seaborn.xkcd_rgb["pale red"], markers='X')
    #seaborn.plt.ylim(-0.5,1.5)
#    error_plot.savefig(experiment_name + '/' + domain + '-error.png')
