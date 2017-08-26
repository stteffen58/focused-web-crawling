import redis
import ast
import os
import sys
import time
import tldextract
import multiprocessing
import csv

import pandas as pd

import entitypredictor

from copy import deepcopy,copy

from urlmodelsimulation import extract_features

from sklearn.externals import joblib
from sklearn.preprocessing import RobustScaler,StandardScaler


INDEX_ENTITY = 0
INDEX_LINKS = 1

def get_connection(port):
    r = redis.StrictRedis(host='localhost', port=port, db=0)
    return r


def get_seed_keys(n_seeds, r):
    seed_list = [(r.randomkey().decode(),-1) for i in range(0, n_seeds)]
    return seed_list


def crawl_step(url,r):
    if not ('http' in url[0] or 'https' in url[0]):
        return []
    return r.lrange(url[0], 0, 1)


def crawl_bfs(total_requests, seeds, domain, port, train_urls):
    n_requests = 0
    r = get_connection(port)
    seeds = deepcopy(seed_list)
    crawled_entities = []
    keys_not_found = 0
    already_crawled = set()
    while seeds and n_requests < total_requests:
        sys.stdout.write('\r Number of requests: '+str(n_requests))
        sys.stdout.flush()
        seed = seeds.pop(0)
        if seed[0] in already_crawled:
            continue
        if seed[0] not in train_urls:
            n_requests += 1
        try:
            page = crawl_step(seed,r)
            already_crawled.add(seed[0])
            entity = ast.literal_eval(page[INDEX_ENTITY].decode('utf-8'))
            entity = (entity + [0, 0],seed[0]) # set column prediction and label to 0
            crawled_entities.append(entity)
            extracted_links = ast.literal_eval(page[INDEX_LINKS].decode('utf-8'))
            for link in extracted_links:
                if link and link not in already_crawled:
                    seeds.append((link,-1))
        except (KeyError,IndexError) as e:
            keys_not_found += 1
    return crawled_entities,keys_not_found


def order_seeds(seed_list,urlmodel,avg):
    current_seeds = []
    start = time.time()
    for seed,pred in seed_list:
        if pred == -1: # -1 means no prediction has been made for this one
            df = extract_features(seed,avg)
            pred = urlmodel.predict(df)
#            print (seed,pred)
#            print (df)
        current_seeds.append((seed,pred))
    end = time.time()
#    print (end-start)
    return current_seeds


def crawl_focused(total_requests, seed_list, domain, port, experiment_name, train_urls, avg):
    n_requests = 0
    r = get_connection(port)
    crawled_entities = []
    keys_not_found = 0
    already_crawled = set()
    
    for file in os.listdir(experiment_name + '/pickles/'): 
        if domain in file:
            urlmodel = joblib.load(experiment_name + '/pickles/' + file)
    seeds = order_seeds(seed_list,urlmodel,avg)
    print (total_requests)
    while seeds and n_requests < total_requests:
        sys.stdout.write('\r Number of requests: ' + str(n_requests))
        sys.stdout.flush()
        seed = seeds.pop(seeds.index(max(seeds,key=lambda tup:tup[1])))
        if seed[0] in already_crawled:
            continue
        if seed[0] not in train_urls:
            n_requests += 1

        try:
            page = crawl_step(seed,r)
            already_crawled.add(seed[0])
            entity = ast.literal_eval(page[INDEX_ENTITY].decode('utf-8'))
            entity = (entity + [0, 0],seed[0]) # set column prediction and label to 0
            crawled_entities.append(entity)
            extracted_links = ast.literal_eval(page[INDEX_LINKS].decode('utf-8'))
            for link in extracted_links:
                if link and link not in already_crawled:
                    seeds.append((link,-1))
            seeds = order_seeds(seeds,urlmodel,avg)
        except (KeyError,IndexError) as e:
            keys_not_found += 1
    return crawled_entities,keys_not_found


def eval_bfs(crawled_entities,experiment_name,train_urls):
    entity_pickel = experiment_name + '/pickles/entity-model.pkl'
    scaler_pickel = experiment_name + '/pickles/scaler.pkl'
    entitymodel = joblib.load(entity_pickel)
    scaler = joblib.load(scaler_pickel)
    columns = ['domain', 'url', 'name', 'rating_count', 'rating_value', 'has_image', 'identifier_count', 'list_length',
               'list_rows', 'table_length', 'table_rows', 'description_length', 'price', 'prediction', 'label']
    n_rel_pages = 0
    n_irrel_pages = 0
    print ('\n evaluate')
    for i,entity in enumerate(crawled_entities):
        sys.stdout.write('\r Number of entities evaluated: '+str(i))
        sys.stdout.flush()
        if entity[1] in train_urls:
            continue
        df = pd.DataFrame(data=[entity[0]],columns=columns)
        Xtr_r = scaler.transform(df.drop(['label','prediction','domain','url'],axis=1))
#        print (Xtr_r)
#        print (df)
#        time.sleep(1)
        prediction = entitypredictor.predict(Xtr_r, entitymodel)
        if prediction > 0.5:
             n_rel_pages += 1
        else:
            n_irrel_pages += 1
    return n_rel_pages,n_irrel_pages


def save_to_df(domain,experiment_name,n_seeds,current_request,result1,evaluation0,evaluation1,ellapsed_time,columns):
    fname = experiment_name + '/' + domain + '-simulated-result.csv'
    df = pd.DataFrame(data=[[domain,experiment_name,n_seeds,current_request,result1,evaluation0,evaluation1,ellapsed_time]],columns=columns)
    if not os.path.isfile(fname):
        df_finale = df
    else:
        df_finale = pd.DataFrame.from_csv(fname)
        df_finale = pd.concat([df_finale,df])
    print ('\n')
    print (df_finale)   
    df_finale.to_csv(fname)


def start(current_request,seed_list,domain,port,train_urls,avg):
    seeds = deepcopy(seed_list)
    start = time.time()
    result = crawl_bfs(current_request,seeds,domain,port,train_urls)
    end = time.time()
    evaluation = eval_bfs(result[0],'base',train_urls)
    save_to_df(domain,'base',n_seeds,current_request,result[1],evaluation[0],evaluation[1],end-start,columns)

    start = time.time()
    result = crawl_focused(current_request,seed_list,domain,port,'productcatalog',train_urls,avg)
    end = time.time()
    evaluation = eval_bfs(result[0],'productcatalog',train_urls)
    save_to_df(domain,'productcatalog',n_seeds,current_request,result[1],evaluation[0],evaluation[1],end-start,columns)


domain = sys.argv[1]
port = int(sys.argv[2])
n_seeds = 10
total_requests = [10000,20000,30000]
#total_requests = [1000]
columns = ['domain','experiment-name','#seeds','#requests','#notfound','#rel-pages','#irrel-pages','time']
seed_list = get_seed_keys(n_seeds,get_connection(port)) # make sure all experiments use the same seeds

pool = multiprocessing.Pool(len(total_requests))
results = []
train_urls = []
with open('misc/' + domain + '-train.csv','r') as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')
    train_urls = next(reader)
with open('misc/' + domain + '-avg','r') as f:
    avg = f.readline().strip()

for current_request in total_requests:
    result = pool.apply_async(start,(current_request,seed_list,domain,port,train_urls,avg))
    results.append(result)
for r in results:
    r.get()
    pass
print ('done')
