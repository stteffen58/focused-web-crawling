import gzip
import warc
from tinydb import TinyDB
import os
import random
import sys


path = '/home/eckel/data/dataset/archive/'
database_path = '/home/eckel/master/finale/database/'
max_sample_size = 1000
current_sample_size = 0
domain = sys.argv[1]
number_of_files = int(sys.argv[2])
samples_per_file = max_sample_size/number_of_files
db = TinyDB(database_path + domain + '.json')
files = [f for f in os.listdir(path) if domain in f]

for file in files:
    file_path = path + file
    cur_num = 0
    with gzip.open(file_path, mode='rb') as gzf:
        for record in warc.WARCFile(fileobj=gzf):
            if random.randint(0,1) == 1:
                url = record['WARC-Target-URI'].strip()
                html = record.payload.read()
                db.insert({url:str(html)})
                current_sample_size += 1
                cur_num += 1

            if current_sample_size >= 1000 or samples_per_file == cur_num:
                break

            sys.stdout.write('\r'+str(current_sample_size))
            sys.stdout.flush()

    if current_sample_size >= 1000:
        break


