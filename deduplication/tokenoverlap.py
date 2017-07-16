import csv
from urllib.parse import urlparse,parse_qsl,urlsplit
import re
from nltk.stem import PorterStemmer


def preprocess_url(url):
    parse = urlparse(url)
    snake_case_p = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', parse.path).lower()
    stemmer = PorterStemmer()
    path_split = [t for t in re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', snake_case_p) if t and not t.isdigit()] #and 'store' not in t and 'jsp' not in t]
    path = set(stemmer.stem(t) for t in path_split[1:len(path_split)])

    query_dict = dict(parse_qsl(urlsplit(url).query))
    query_values = set(query_dict[k] for k in query_dict if query_dict[k])
    query_keys = set(k for k in query_dict if k and 'id' not in k)
    snake_case_q = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', '?'.join(query_values)).lower()
    query_split = set(stemmer.stem(t) for t in re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', snake_case_q) if t)
#    snake_case_q = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', '?'.join(query_keys)).lower()
#    key_split = set(stemmer.stem(t) for t in re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', snake_case_q) if t)
#    query_split.update(key_split)
    query_split.update(path_split)
    return query_split


def is_duplicate(url1,url2):
    tokens1 = preprocess_url(url1)
    tokens2 = preprocess_url(url2)
    inter_path = set.intersection(tokens1,tokens2)
    if not inter_path:
        return False
    combined = set.union(tokens1,tokens2)
    ratio = len(inter_path) / len(combined)

    if ratio > 0.73:
         return True
    return False


def is_path_duplicate(path1,path2):
    inter_path = set.intersection(path1,path2)
    if not inter_path:
        return False
    ratio = 0
    if len(path1) > len(path2):
        ratio = len(inter_path) / len(path1)
    else:
        ratio = len(inter_path) / len(path2)

    if ratio > 0.5:
         return True
    return False


def is_query_duplicate(query1,query2):
    inter_query = set.intersection(query1,query2)
    if not inter_query:
        return False
    ratio = 0
    if len(query1) > len(query2):
        ratio = len(inter_query) / len(query1)
    else:
        ratio = len(inter_query) / len(query2)

    if ratio > 0.5:
         return True
    return False


fp = []
tp = []
fn = []
tn = []
total = 0

with open('data/walmart.csv','r') as f:
    reader = csv.reader(f,delimiter=',')

    for line in reader:
        print (total,end='\r')
        total += 1
        url1 = line[0]
        url2 = line[1]
        if is_duplicate(url1,url2):
            # duplicate found
            #if is_query_duplicate(tokens1[1],tokens2[1]):
            # duplicate found
            if line[3] == 'true':
                tp.append(line)
            else:
                fp.append(line)
        else:
            # no duplicate found
            if line[3] == 'true':
                fn.append(line)
            else:
                tn.append(line)
        if total == 100000:
             break

fps = len(fp)
print ('fp ' + str(fps))
tps = len(tp)
print ('tp ' + str(tps))
tns = len(tn)
print ('tn ' + str(tns))
fns = len(fn)
print ('fn ' + str(fns))
precision = 0
recall = 0
try:
    precision = tps/(tps+fps)
    recall = tps/(tps+fns)
except:
    pass

print ('total ' + str(total))
print ('Precision ' + str(precision))
print ('Recall ' + str(recall))
print (fn[0:20])
