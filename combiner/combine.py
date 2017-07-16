import re
import os
import pandas as pd
import numpy as np

from combiner import extractor

from urllib.parse import urlparse

from sklearn.externals import joblib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def extract_url_features(df):
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
    df_finale = pd.concat([df,pd.DataFrame(doc_term_matrix.toarray(),index=df.index)],axis=1)
    df_finale = df_finale.drop(['url_path','path_length','average_deviation','average_ratio','amount_parameters','query_length','title_length'],axis=1)
    return df_finale


'''
Load models from pickle
'''
entity_model = joblib.load('data/entity-model.pkl')
url_pickel_path = 'data/pickles/'
url_models = {}

for file in os.listdir(url_pickel_path):
    domain = file[0:file.rindex('.')]
    url_model = joblib.load(url_pickel_path + file)
    url_models[domain] = url_model

df = pd.DataFrame.from_csv('data/dataframe.csv')
df.fillna(0,inplace=True)


'''
Predict URL model by choosing correct classifier per site. Does URL lead to product - yes/no?
'''
X_url = extract_url_features(pd.DataFrame(df['url'],index=df.index)) # data + features for URL model
df_url_pred = pd.DataFrame(index=df.index, columns=['prediction']) # saves prediction of url model
avg_proba = pd.DataFrame().from_csv('data/avg_proba.csv') # contains avg domain scores

for i,r in X_url.iterrows():
    netloc = urlparse(r['url']).netloc
    domain = netloc[netloc.index('.') + 1:len(netloc)]
    if domain in url_models:
        url_clf = url_models[domain]
        url_pred_score = url_clf.predict_proba(r.drop('url'))
        url_pred = url_clf.predict(r.drop('url'))
        df_domain_score = avg_proba[avg_proba['domain'] == domain[0:domain.index('.')]]
        df_url_pred.loc[i, 'predict_proba'] = (url_pred_score[0][1] + df_domain_score['avg_proba'].iloc[0]) / 2 # combine predicted probability with average probability per domain
        df_url_pred.loc[i, 'prediction'] = url_pred[0] # predicted label

#df_url_pred.fillna(0, inplace=True)
#df_url_pred.loc[df_url_pred['predict_proba'] > 0.5,'prediction'] = 1
#df_url_pred.loc[df_url_pred['predict_proba'] <= 0.5,'prediction'] = 0


'''
Apply entity model on remaining URLs. Based on output from URL model.
'''
drop_columns = ['label', 'domain', 'name', 'url']
X_test = df[df_url_pred['prediction'] == 1]
entity_pred = pd.DataFrame(entity_model.predict(X_test.drop(drop_columns,axis=1)),columns=['prediction'])
inter = df_url_pred[df['label'] == 1]
inter = pd.DataFrame(inter[inter['prediction'] == 0]['prediction'])
final_pred = pd.concat([entity_pred,inter],axis=0)
final_pred.replace(to_replace=0.0, value=0,inplace=True)
#final_pred.reset_index(drop=True,inplace=True)

inter = df[df_url_pred['prediction'] == 0]
inter = inter[inter['label'] == 1]['label']
y_test = np.concatenate((X_test['label'], inter))
recall = recall_score(y_test, final_pred)
f1 = f1_score(y_test, final_pred)
accuracy = accuracy_score(y_test, final_pred)
precision = precision_score(y_test, final_pred)
cm = confusion_matrix(y_test,final_pred)
tn = cm[0][0]
fn = cm[1][0]
tp = cm[1][1]
fp = cm[0][1]
print ('Precision ' + str(precision) + ' Recall ' + str(recall) + ' F1 ' + str(f1) + ' Accuracy ' + str(accuracy))
