import gzip
import warc
import redis
import os
import random
import sys
import csv
import re
import entityfeatureextractor
from bs4 import BeautifulSoup
from bs4 import NavigableString
#from _thread import start_new_thread, allocate_lock
#from threading import Thread
from multiprocessing.pool import Pool
import time
from urllib.parse import urljoin


'''
creates redis database for each domain. each entry is of the form (key,[entityfeatures,links,anchor text, sourrounding text]) where the key is the URL of the web page
'''

def create(file_path,i,port,base_url):
    join_errors = 0
    r = redis.StrictRedis(host='localhost', port=port, db=0)
    with gzip.open(file_path, mode='rb') as gzf:
        for record in warc.WARCFile(fileobj=gzf):
            url = record['WARC-Target-URI'].strip()
            html = record.payload.read()
            soup = BeautifulSoup(html,'lxml')
            links = []
            
            for link in soup.find_all('a'):
                href = link.get('href')
                links.append(href)
                anchor_text = re.sub('[^A-Za-z0-9]+',' ',link.text.strip())
                s = ''
                tag = link.parent
                while len(s) < 30:
                    if tag:
                        if not isinstance(tag, NavigableString):
                            t = tag.text.strip()
                            t = ' '.join(t.split())
                            if len(s+t) < 30:
                                s += t
                            else:
                                s += t[-(30-len(s)+1):]
                        tag = tag.parent
                    else:
                        break
                
                tag = link.nextSibling
                siblings = True
                while len(s) < 60:
                    if tag:
                        if not isinstance(tag, NavigableString):
                            t = tag.text.strip()
                            t = ' '.join(t.split())
                            if len(s+t) < 60:
                                s += t         
                            else:
                                s += t[-(60-len(s)+1):]
                        tag = tag.next_sibling
                        continue
                    else:
                        if not siblings:
                            break
                        siblings = False
                        tag = link.parent
                        if tag:
                            tag = tag.next_sibling
                s = re.sub(' +',' ',s)

                try:
                    href_join = urljoin(base_url, href)
                    if r.exists(href_join):
#                        r.rpush(href_join,anchor_text)
                        r.lset(href_join,3,s)
#                    else:
#                        r.rpush(href_join,[])
#                        r.rpush(href_join,[])
#                        r.rpush(href_join,anchor_text)
#                        r.rpush(href_join,s)
                except:
                    join_errors += 1
                    continue
            #row = entityfeatureextractor.extract_row(domain,url,html)
            #if r.exists(url):
            #r.lset(url,0,row)
                #r.lset(url,1,links)
            #else:
            #    r.rpush(url,row)
            #    r.rpush(url,links)
    print ('join errors: ' + str(join_errors))


path = '/home/eckel/data/dataset/archive/'
database_path = '/home/eckel/master/finale/database/'
domain = sys.argv[1]
port = int(sys.argv[2])
files = [f for f in os.listdir(path) if domain.lower() in f]
#files = ['target.com0.warc.gz']
async_results = []
pool = Pool(processes=len(files))
start = time.time()
base_url = 'http://www.'+domain.lower()+'.com'
print (base_url)
for i,file in enumerate(files):
    file_path = path + file
    async_results.append(pool.apply_async(create, (file_path,i,port,base_url)))
    print (file)

for i,result in enumerate(async_results):
    entries = result.get()
    print (str(i) + ' done')
end = time.time()

print ('\ndone')
