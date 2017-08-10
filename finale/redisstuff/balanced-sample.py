import numpy as np
import pandas as pd
import sys
import csv
import redis
import ast
import entitypredictor
import os
from sklearn.externals import joblib


'''
loads complete database into dataframe and samples afterwards. parameters : (1) domain (2) experiment name (3) redis port
'''

def balanced_subsample(x,y,subsample_size=0.1):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys


def sample(df_full,sample_size,name,domain):
    data_product = df_full[df_full['prediction'] >= 0.5].index
    data_nonproduct = df_full[df_full['prediction'] < 0.5].index
    labels1 = np.array([1 for i in data_product[0:len(data_product)]])
    labels0 = np.array([0 for i in data_nonproduct[0:len(data_nonproduct)]])
    data = np.hstack((data_product,data_nonproduct))
    label = np.hstack((labels1,labels0))
    sample = balanced_subsample(data,label,subsample_size=sample_size)

    outputpath = experiment_name + '/' + domain  + '/samples/' + name + '/'
    products = np.hstack((sample[0][sample[1] == 1],sample[0][sample[1] == 0]))
    labels = np.hstack((sample[1][sample[1] == 1],sample[1][sample[1] == 0]))
    data = np.vstack((products,labels))
    df_final = df_full.iloc[products]
    df_final.reset_index(drop=True,inplace=True)
    df_final.to_csv(outputpath + domain + '.csv')


experiment_name = sys.argv[2]
domain = sys.argv[1]

# check if directory exists
if not os.path.isdir(experiment_name + '/' + domain  + '/samples/big/'):
    print ('big')
elif not os.path.isdir(experiment_name + '/' + domain  + '/samples/medium/'):
    print ('medium')
elif not os.path.isdir(experiment_name + '/' + domain  + '/samples/small/'):
    print ('small')
elif not os.path.isdir('dataframes/'):
    print ('dataframes')
else:

    # load database
    r = redis.StrictRedis(host='localhost', port=int(sys.argv[3]), db=0)
    columns = ['domain','url','name','rating_count','rating_value','has_image','identifier_count','list_length',
                'list_rows','table_length','table_rows','description_length','price','prediction','label']
    df_full = pd.DataFrame(columns=columns)
    entity_pickel = experiment_name + '/pickles/entity-model.pkl'
    entitymodel = joblib.load(entity_pickel)

    for i,key in enumerate(r.scan_iter()):
        sys.stdout.write('\r'+str(i))
        sys.stdout.flush()
        row = ast.literal_eval(r.lrange(key,0,0)[0].decode('utf-8'))
        row = row + [0,0] # set column prediction and label to 0
        df = pd.DataFrame(data=[row],columns=columns)
        prediction = entitypredictor.predict(df,entitymodel)
        df['prediction'] = prediction
        df.loc[df['prediction'] >= 0.5,'label'] = 1
        df.loc[df['prediction'] < 0.5,'label'] = 0
        df_full = df_full.append(df,ignore_index=True)

    print (df_full.head())
    df_full.to_csv('dataframes/' + domain + '.csv')

    sample(df_full,0.01,'small',domain)
    sample(df_full,0.1,'medium',domain)
    sample(df_full,0.3,'big',domain)
