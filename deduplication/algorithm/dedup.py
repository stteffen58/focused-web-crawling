from deduplication.algorithm.rule import Rule
import csv
import os
from urllib.parse import urlparse
from collections import defaultdict
from copy import copy
import sys


urls = set()

def generalize_rules(rules,minS,fan_out):
    gen_rules = rules
    d_empty = False
    while not d_empty:
        D = calc_distance(gen_rules)
        D_ = filter_fan_out(D,fan_out)
        if not D_:
            d_empty = True
        reduction = (len(D) - len(D_))
        for r in D_:
            for k in r.delta_keys:
                r_ = r
                if k in r.c and r.c.get(k,'') != '*':
                    r_.c[k] = '*'
                else:
                    r_.a[k] = '*'
                #gen_rules.append(r_)
    final_rules = []
    dups = []
    for r in gen_rules:
        dup = False
        for r1 in dups:
            if r1 == r:
                dup = True
        if not dup:
            dups.append(r)
            support = supp(r)
            r.support = support
            if support > minS:
                final_rules.append(r)
    return final_rules

def calc_distance(gen_rules):
    D = []
    for r1 in gen_rules:
        for r2 in gen_rules:
            if r1 == r2:
                continue

            delta_keys = set()
            values = set()
            keyset = set.intersection(set(k for k in r1.c),set(k_ for k_ in r2.c))
            keyset.update(set.intersection(set(k for k in r1.a), set(k_ for k_ in r2.a)))

            for k in keyset:
                    if r1.c.get(k,'') != r2.c.get(k,'') and r1.c.get(k,'') != '*':
                            delta_keys.add(k)
                            values.add(r2.c[k])
                    elif r1.a.get(k,'') != r2.a.get(k,'') and r1.a.get(k,'') != '*':
                        delta_keys.add(k)
                        values.add(r2.a[k])

            if len(delta_keys) == 1:
                r1.delta_keys.update(delta_keys)
                r1.delta_key_values.update(values)
                D.append(r1)
    return D

def filter_fan_out(D,f):
    filtered = []
    for r in D:
        if len(r.delta_key_values) >= f:
            filtered.append(r)
    return filtered

def supp(rule):
    pairs = []
    support = 0
    for url1 in urls:
        transformed_url = apply_rule(rule,url1)
        for url2 in urls:
            url = get_url(url2)
            match = True
            for k in url:
                if url.get(k,'') != transformed_url.get(k,'') and transformed_url.get(k,'') != '*':
                    match = False
            if match:
                support += 1
    return support

def apply_rule(rule,url_string):
    c = rule.c
    url = get_url(url_string)
    for key in c:
        if c.get(key,'') != url.get(key,'') and c.get(key,'') != '*':
            # url does not match context of rule
            return url

    # context matches, thus apply transformation
    a = rule.a
    canonical_url = {}
    for k in url:
        k_ = a[k]
        if k_ in c:
            canonical_url[k] = c[k]
        else:
            canonical_url[k] = k_
    return canonical_url

def generate_all_rules(cluster):
    rules = []
    urls.update(cluster)
    for el1 in cluster:
        for el2 in cluster:
            #if el1 != el2: # do not create rule for equal URLs
            r = generate_rule(el1, el2)
            rules.append(r)
    return rules

def generate_rule(url1, url2):
    u = get_url(url1)
    v = get_url(url2)
    r = Rule()
    r.c = u
    keyset = set(v.keys())
    keyset.update(u.keys())
    for k in keyset:
        for k_ in keyset:
            if k in v:
                if  u.get(k_,'') == v.get(k,''):
                    r.a[k] = k_
                    break
                r.a[k] = v[k]
            else:
                r.a[k] = ''
    return r

def get_url(url):
    parsed = urlparse(url)
    url_map = defaultdict(str)
    url_map = get_protocol(parsed,url_map)
    url_map = get_netloc(parsed,url_map)
    url_map = get_path(parsed,url_map)
    url_map = get_query(parsed,url_map)
    return url_map

def get_query(parsed,url_map):
    if parsed.query:
        for token in parsed.query.split('&'):
            try:
                key = token[0:token.rindex('=')].strip()
                value = token[token.rindex('=') + 1:len(token)].strip()
                url_map[key] = value
            except:
                continue # sometimes key with no value (walmart)
    return url_map

def get_path(parsed,url_map):
    index = max(url_map.keys()) + 1
    for token in parsed.path.split('/'):
        if token:
            url_map[index] = token
            index += 1
    return url_map

def get_netloc(parsed,url_map):
    netloc = parsed.netloc
    if not url_map:
        url_map[0] = netloc
    else:
        index = max(url_map.keys()) + 1
        url_map[index] = netloc
    return url_map

def get_protocol(parsed,url_map):
    protocol = parsed.scheme
    if not url_map:
        url_map[0] = protocol
    else:
        index = max(url_map.keys()) + 1
        url_map[index] = protocol
    return url_map

def create_dup_cluster(filepath):
    temp_dict = defaultdict(list)
    csv_file = open(filepath,'r')
    reader = csv.reader(csv_file, delimiter=';')
    count = 0
    for line in reader:
        url1 = line[0]
        url2 = line[1]
        if line[3] == 'true':
            count += 1
            temp_dict[url1].append(url2)
    dup_cluster = defaultdict(set)
    index = 0
    for k in temp_dict:
        match = False
        for k1 in dup_cluster:
            if k in dup_cluster[k1]:
                match = True
                break
        if not match:
            dup_cluster[index].add(k)
            dup_cluster[index].update(temp_dict[k])
            index += 1
    return dup_cluster

path_train = '../goldstandard-url/train/'
path_out = '../goldstandard-url/output/'
fan_out_dict = {'overstock.csv':2,'walmart.csv':10,'searsoutlet.csv':2,'selection.alibaba.csv':10,'shop.csv':10,'target.csv':10}
# for file in os.listdir(path_train):
#     dup_cluster = create_dup_cluster(path_train + file)
#     total_cluster = len(dup_cluster)
dup_cluster = {0:['http://www.xyz.com/show.php?page=11',
                    'http://www.xyz.com/show.php?page=12',
                    'http://www.xyz.com/show.php?page=13',
                    'http://www.xyz.com/13/index.html']}
# dup_cluster = {0:['http://www.xyz.com/show.php?sid=71829',
#    'http://www.xyz.com/show.php?sid=72930',
#    'http://www.xyz.com/show.php?sid=17628']}
#     csvfile = open(path_out+file,'w')
#     writer = csv.writer(csvfile,delimiter=';')

for i,d in enumerate(dup_cluster):
    rules = generate_all_rules(dup_cluster[d])
    #gen_rules = generalize_rules(rules,10,fan_out_dict[file])
    gen_rules = generalize_rules(rules,0,0)
    print (gen_rules)
    # if gen_rules:
    #     writer.writerow(gen_rules)
    urls = set()
    # sys.stdout.write(str(i) + ' of ' + str(total_cluster) + ' from ' + file + '\n')
    # sys.stdout.flush()
    # csvfile.close()

print ('done')

#r = ({0: 'http', 1: 'www.amazon.com', 2: '*', 3: 'dp', 4: '0545010225'},
#     {0: 0, 1: 1, 2: '*', 3: 3, 4: 4})

#print (apply_rule(r,'http://www.amazon.com/Deathly-Hallows/dp/0545010225'))

