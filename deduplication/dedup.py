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
        print (reduction)
        for x in D_:
            r = x[0]
            c = r[0]
            a = r[1]
            k = x[1]
            a_ = a
            c_ = c
            if k in c and c.get(k,'') != '*':
                c_[k] = '*'
            else:
                a_[k] = '*'

            #r_ = (c_,a_)
            #gen_rules.append(r_)

    final_rules = []
    dups = []
    for r in gen_rules:
        dup = False
        t1 = r[0].items()
        t2 = r[1].items()
        for r1 in dups:
            t1_ = r1[0].items()
            t2_ = r1[1].items()
            if t1_ == t1 and t2_ == t2:
                dup = True
        if not dup:
            dups.append(r)
            support = len(supp(r))
            if support > minS:
                final_rules.append((r,support))
    return final_rules

def filter_fan_out(D,f):
    fan_out = defaultdict(list)
    for x in D:
        r = x[0]
        c = r[0]
        k = x[1]
        fan_out[k].append(c[k])

    filtered = []
    for x in D:
        r = x[0]
        k = x[1]
        if(len(fan_out[k]) >= f):
            filtered.append(x)
    return filtered

def supp(rule):
    pairs = []
    for url1 in urls:
        transformed_url = apply_rule(rule,url1)
        for url2 in urls:
            if url1 != url2:
                if transformed_url == apply_rule(rule,url2):
                    pairs.append((url1,url2))
    return pairs

def apply_rule(rule,url_string):
    c = rule[0]
    url = get_url(url_string)
    for key in c:
        if c[key] != url[key] and c[key] != '*':
            # url does not match context of rule
            return url

    # context matches, thus apply transformation
    a = rule[1]
    canonical_url = {}
    for k in url:
        k_ = a[k]
        if k_ in c:
            canonical_url[k] = c[k]
        else:
            canonical_url[k] = k_
    return canonical_url

def calc_distance(gen_rules):
    D = []
    for r1 in gen_rules:
        c = r1[0]
        a = r1[1]
        for r2 in gen_rules:
            differ_keys = set()
            c_ = r2[0]
            a_ = r2[1]

            keyset = set.intersection(set(k for k in c),set(k_ for k_ in c_))
            for k in keyset:
                if c.get(k,'') != c_.get(k,'') and c.get(k,'') != '*':
                        differ_keys.add(k)

            keyset = set.intersection(set(k for k in a), set(k_ for k_ in a_))
            for k in keyset:
                if a.get(k,'') != a_.get(k,'') and a.get(k,'') != '*':
                    differ_keys.add(k)

            if len(differ_keys) == 1:
                t = (r1,differ_keys.pop())
                D.append(t)
    return D

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
    print (count)
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
    a = defaultdict(str)
    keyset = set(v.keys())
    keyset.update(u.keys())
    for k in keyset:
        for k_ in keyset:
            if k_ in u and k in v:
                if  u.get(k_,'') == v.get(k,''):
                    a[k] = k_
                    break
                a[k] = v[k]
    return u,a

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

path_train = 'goldstandard-url/train/'
path_out = 'goldstandard-url/output/'
fan_out_dict = {'overstock.csv':2,'walmart.csv':10,'searsoutlet.csv':2,'selection.alibaba.csv':10,'shop.csv':10,'target.csv':10}
# for file in os.listdir(path_train):
#     print (file)
#     dup_cluster = create_dup_cluster(path_train + file)
dup_cluster = {0:['http://www.xyz.com/show.php?page=NUM',
                     'http://www.xyz.com/NUM/index.html']}
# dup_cluster = {0:['http://www.xyz.com/show.php?sid=71829',
#                     'http://www.xyz.com/show.php?sid=72930',
#                     'http://www.xyz.com/show.php?sid=17628']}
#    csvfile = open(path_out+file,'w')
#    writer = csv.writer(csvfile,delimiter=';')

for d in dup_cluster:
    print ('generate')
    rules = generate_all_rules(dup_cluster[d])
    print ('generalize')
    #gen_rules = generalize_rules(rules,10,fan_out_dict[file])
    gen_rules = generalize_rules(rules,0,0)

    print ('sort')
    gen_rules = sorted(gen_rules, key=lambda x: x[1])
    print (gen_rules)
#        if gen_rules:
#            writer.writerow(gen_rules)
    urls = set()
    #    csvfile.close()

print ('done')

#r = ({0: 'http', 1: 'www.amazon.com', 2: '*', 3: 'dp', 4: '0545010225'},
#     {0: 0, 1: 1, 2: '*', 3: 3, 4: 4})

#print (apply_rule(r,'http://www.amazon.com/Deathly-Hallows/dp/0545010225'))