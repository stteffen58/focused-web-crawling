import gzip
import warc
import redis
import os
import random
import sys
import csv
import entityfeatureextractor
from bs4 import BeautifulSoup
#from _thread import start_new_thread, allocate_lock
#from threading import Thread
from multiprocessing.pool import Pool


import time

'''
creates a random balanced sample by using extracted html according to schema.org/Product. First inserts only products into the database until half of the sample size is reached.
'''

def create(file_path,i):
    entries = []
    r = redis.StrictRedis(host='localhost', port=6379, db=0)
    with gzip.open(file_path, mode='rb') as gzf:
        for record in warc.WARCFile(fileobj=gzf):
            url = record['WARC-Target-URI'].strip()
            html = record.payload.read()
            soup = BeautifulSoup(html,'lxml')
            links = [link.get('href') for link in soup.find_all('a')]
            row = entityfeatureextractor.extract_row(domain,url,html)
            r.rpush(url,row)
            r.rpush(url,links)


path = '/home/eckel/data/dataset/archive/'
database_path = '/home/eckel/master/finale/database/'
domain = sys.argv[1]
files = [f for f in os.listdir(path) if domain.lower() in f]

async_results = []
pool = Pool(processes=len(files))
start = time.time()
for i,file in enumerate(files):
    file_path = path + file
    async_results.append(pool.apply_async(create, (file_path,i)))
    print (file)

for i,result in enumerate(async_results):
    entries = result.get()
    print (str(i) + ' done')
end = time.time()

print ('\ndone')
