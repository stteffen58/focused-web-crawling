import gzip
import warc
from tinydb import TinyDB
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware
import os
import random
import sys
import csv


'''
creates a random balanced sample by using extracted html according to schema.org/Product. First inserts only products into the database until half of the sample size is reached.
'''

path = '/home/eckel/data/dataset/archive/'
database_path = '/home/eckel/master/finale/database/'
domain = sys.argv[1]
files = [f for f in os.listdir(path) if domain.lower() in f]
print (files)

f = open('/home/eckel/data/quad-stats/producturls.csv','r')
reader = csv.reader(f,delimiter=',')
product_urls = {line[0]:line[1:len(line)] for line in reader}

max_sample_size = 10000
current_sample_size = 0
number_of_files = len(files)
samples_per_file = max_sample_size/number_of_files
db = TinyDB(database_path + domain + '.json', storage=CachingMiddleware(JSONStorage))
CachingMiddleware.WRITE_CACHE_SIZE = 500

for file in files:
    print (file)
    file_path = path + file
    cur_num = 0
    curr_products = product_urls[domain.lower()+'.com']
    curr_products_html = {}
    curr_nonproducts_html = {}
    with gzip.open(file_path, mode='rb') as gzf:
        for record in warc.WARCFile(fileobj=gzf):
            url = record['WARC-Target-URI'].strip()
            html = record.payload.read()
            if url in curr_products:
                curr_products_html[url] = html
            else:
                curr_nonproducts_html[url] = html
    #print (len(curr_products_html))
    #print (len(curr_nonproducts_html))

    for url in curr_products_html:
        db.insert({'url':url,'html':str(curr_products_html[url])})
        current_sample_size += 1
        cur_num += 1

        if int(samples_per_file/2) == cur_num:
            break
        sys.stdout.write('\r'+str(current_sample_size))
        sys.stdout.flush()

    for url in curr_nonproducts_html:
        db.insert({'url':url,'html':str(curr_nonproducts_html[url])})
        current_sample_size += 1
        cur_num += 1

        if int(samples_per_file) == cur_num:
            break
        sys.stdout.write('\r'+str(current_sample_size))
        sys.stdout.flush()

print ('\n')
db.close()
