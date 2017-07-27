import gzip
import warc
from tinydb import TinyDB
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware
import os
import random
import sys
import csv
import entityfeatureextractor
from bs4 import BeautifulSoup


'''
creates a random balanced sample by using extracted html according to schema.org/Product. First inserts only products into the database until half of the sample size is reached.
'''

path = '/home/eckel/data/dataset/archive/'
database_path = '/home/eckel/master/finale/database/'
domain = sys.argv[1]
files = [f for f in os.listdir(path) if domain.lower() in f]
#print (files)

db = TinyDB(database_path + domain + '.json', storage=CachingMiddleware(JSONStorage))
CachingMiddleware.WRITE_CACHE_SIZE = 10000
current_sample_size = 0

for file in files:
    print (file)
    file_path = path + file
    with gzip.open(file_path, mode='rb') as gzf:
        for record in warc.WARCFile(fileobj=gzf):
            url = record['WARC-Target-URI'].strip()
            html = record.payload.read()
            soup = BeautifulSoup(html,'lxml')
            links = [link.get('href') for link in soup.find_all('a')]
            row = entityfeatureextractor.extract_row(domain,url,html)
            db.insert({'url':url,'entity':row,'links':links})
            current_sample_size += 1
            sys.stdout.write('\r'+str(current_sample_size))
            sys.stdout.flush()

db.close()
print ('\n')
