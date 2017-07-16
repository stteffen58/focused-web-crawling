import os
import csv
import re
import pandas as pd
import numpy as np

from urllib.parse import urlparse

import extractor

from ast import literal_eval as make_tuple

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib


def url_tokenizer():
    pass

            
def extract_features(df):
    url_lengths = []
    path_length = []
    for i,row in df.iterrows():
        url = row[0]
        try:
       	    url_lengths.append(len(url))
        except:
            print (url)
        url_path = extractor.getUrlPath(url)
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
        
        query = extractor.getUrlQuery(url)
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

    vectorizer = HashingVectorizer(stop_words='english',analyzer='word',n_features=2**15)
    doc_term_matrix = vectorizer.fit_transform(df['url'])
    df_final = pd.concat([df,pd.DataFrame(doc_term_matrix.toarray(),index=df.index)],axis=1)
    return df


#pd.options.display.max_colwidth = 100
path = 'data/dataframes/'
output = []
for file in os.listdir(path + 'train'):
   # if 'shop' not in file:
   #     continue
    print(file)
    df = pd.DataFrame.from_csv(path + 'train/' + file)
    df = df.dropna()
    print (df['url'].head())
    if df.shape[0] < 500:
        continue
    df = extract_features(df)
    df = df.dropna()
    nb = GaussianNB()
    y_train = df['label']
#    print (df.head()[['url','title']])
    drop_columns = ['label','url','url_path','path_length','average_deviation','average_ratio','amount_parameters','query_length','title_length']
    df.drop(drop_columns,inplace=True,axis=1) # use inplace, otherwise memory error
    nb.fit(df,y_train)

    domain = file[0:file.rindex('.')]
    joblib.dump(nb,'pickles/' + domain + '.pkl')

    df_test = pd.DataFrame.from_csv(path + 'test/' + file)
    df_test = df_test.dropna()
    df_test = extract_features(df_test)
    df_test = df_test.dropna()
    y_test = df_test['label']
#    print (df_test.head()[['url','title','label']])
    df_test.drop(drop_columns,inplace=True,axis=1)
    prediction = nb.predict(df_test)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    accuracy = accuracy_score(y_test, prediction)
    cm = confusion_matrix(y_test, prediction)
    tp = cm[1][1]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[0][0]
    print (len(y_test[y_test == 1]), len(y_test[y_test == 0]),precision,recall,f1,accuracy)
    output.append([domain,len(y_test[y_test == 1]),len(y_test[y_test == 0]),tp,fp,fn,tn,precision,recall,f1,accuracy])
    
f = open(path + 'results_svm_terms.csv','w')
writer = csv.writer(f,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
#if os.path.getsize(path + 'results_svm.csv') == 0:
writer.writerow(['domain','#products','#nonproducts','TP','FP','FN','TN','precision','recall','f1','accuracy'])

for out in output:
    writer.writerow([domain,len(y_test[y_test == 1]),len(y_test[y_test == 0]),tp,fp,fn,tn,precision,recall,f1,accuracy])
f.close()

'''columns = ['predict','expect']
columns.extend(df_test.columns)
inter = pd.concat([y_test,df_test],axis=1)
result_plot = pd.DataFrame(data=np.transpose(np.vstack((prediction,inter['label'].as_matrix(),inter['title_length'].as_matrix()))),columns=['predict','label','title_length'],index=y_test.index)
result_plot.loc[(result_plot['predict'] == 1) & (result_plot['label'] == 1),'class'] = 'tp'
result_plot.loc[(result_plot['predict'] == 1) & (result_plot['label'] == 0),'class'] = 'fp'
result_plot.loc[(result_plot['predict'] == 0) & (result_plot['label'] == 0),'class'] = 'tn'
result_plot.loc[(result_plot['predict'] == 0) & (result_plot['label'] == 1),'class'] = 'fn'

print (result_plot[result_plot['class'] == 'tn'].head())
print (result_plot[result_plot['class'] == 'tp'].head())
print (df_test.head())
print (df.head())
'''
