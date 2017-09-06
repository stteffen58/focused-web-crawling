import numpy as np
import pandas as pd
import sys
import csv
import redis
import ast
import entitypredictor
import os

from sklearn.externals import joblib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import RobustScaler


'''
loads complete database into dataframe and samples afterwards. parameters : (1) domain (2) experiment name (3) redis port
'''

def run_experiment(experiment_name):
    entity_pickel = experiment_name + 'pickles/entity-model.pkl'
    entitymodel = joblib.load(entity_pickel)
    scaler_pickel = experiment_name + '/pickles/scaler.pkl'
    
    prediction = entitymodel.predict_proba(Xtr_r)
    df_full['prediction'] = np.array([p[1] for p in prediction])
    df_full.loc[df_full['prediction'] > 0.5,'label'] = 1
    df_full.loc[df_full['prediction'] <= 0.5,'label'] = 0
    print (df_full.head())
    print (df_full.shape)
    df_full.to_csv('dataframes/' + experiment_name + domain + '.csv', quoting=csv.QUOTE_ALL, quotechar='|')


domain = sys.argv[1]

# load database
r = redis.StrictRedis(host='localhost', port=int(sys.argv[2]), db=0)
columns = ['domain','url','name','rating_count','rating_value','rating_ratio','has_image','identifier_count','list_length',
            'list_rows','table_length','table_rows','description_length','price','text','prediction','label']

df_full = pd.DataFrame(columns=columns)
vectorizer = HashingVectorizer(stop_words='english',analyzer='word',n_features=2**15)
ser = pd.Series()
error = 0
i = 0
for key in r.scan_iter():
    entry = r.lrange(key,0,3)
    row = ast.literal_eval(entry[0].decode('utf-8'))
    if row:
        sys.stdout.write('\r' + str(i))
        sys.stdout.flush()
        try:
            anchor = str(entry[2].decode('utf-8')).strip()
            text = str(entry[3].decode('utf-8')).strip()
        except:
            error += 1
            print (key)
            print (entry)
            break
        doc = [anchor+text]
        row = row + [text,0,0] # set column prediction and label to 0
        df = pd.DataFrame(data=[row],columns=columns)

        #doc_term_matrix = vectorizer.fit_transform(doc)
        #df = pd.concat([df,pd.DataFrame(doc_term_matrix.toarray(),index=df.index)],axis=1)
        df_full = pd.concat([df_full,df],axis=0)
        i += 1


scaler = RobustScaler(with_centering=False)
Xtr_r = scaler.fit_transform(df_full[['name','rating_count','rating_value','rating_ratio','has_image','identifier_count','list_length',
            'list_rows','table_length','table_rows','description_length','price']])

print ('\nerrors ' + str(error))
df_full.reset_index(drop=True, inplace=True)

print ('productcatalog')
run_experiment('productcatalog/')
print ('price')
run_experiment('price/')
print ('rating')
run_experiment('rating/')
