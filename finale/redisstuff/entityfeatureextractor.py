from bs4 import BeautifulSoup as bs
from collections import defaultdict
import re
import os
import csv
import sys
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from ast import literal_eval as make_tuple
from tinydb import TinyDB
from sklearn.externals import joblib
from decimal import Decimal


EVALUATOIN_CSV = 'feature-evaluation.csv'

ITEMPROP = 'itemprop'

VALUE = 'http://schema.org/Product'

ITEMTYPE = 'itemtype'

'''
Classes to extract features for the product entity model using different microdata properties, <table>, <li>, etc.
'''

class StandardMicrodataExtractor:
    LIST_KEYWORDS = ''

    TABLE_KEYWORDS = ''

    def __init__(self,html_doc):
        self.name = ''
        self.description_length = 0
        self.rating_value = 0
        self.rating_count = 0
        self.has_image = 0
        self.id_count = 0
        self.table_length = tuple()
        self.list_length = tuple()

        self.soup = bs(html_doc, 'lxml') # lxml is faster than html.parser
        for tag in self.soup.findAll(['script','meta','style']):
            tag.extract()
        self.product = self.soup.find(re.compile('.+'), attrs={ITEMTYPE:VALUE})
        if not self.product:
            raise ValueError('no ' + VALUE + ' found.')
    
    def get_price(self):
        tags = self.product.find(re.compile('.+'), attrs={ITEMPROP: 'price'})
        price = 0.0
        if tags:
            price = ''.join(tags.strings).replace('\n',' ').strip()
            price = re.sub(' +', ' ', price)  # remove double whitespaces
            price = float(Decimal(re.sub(r'[^\d.]', '', price)))
        return price

    def get_name(self):
        tags = self.product.find(re.compile('.+'), attrs={ITEMPROP: 'name'})

        name = ''
        if tags:
            name = ''.join(tags.strings).replace('\n',' ').strip()
            name = re.sub(' +',' ',name) # remove double whitespaces
        if name:
            return 1
        return 0

    def get_description_length(self):
        tags = self.product.find(re.compile('.+'), attrs={ITEMPROP: 'description'})

        description = ''
        if tags:
            unwanted = tags.find_all(['table','ul','li','dl'])  # remove lists from product descriptions as this is a separate feature
            for u in unwanted:
                u.extract()
            description = ''.join(tags.strings).replace('\n',' ').replace('\t','').replace('\r','').strip()
        return len(description)

    def get_rating_value(self):
        '''
        TODO requires normalization
        '''
        tags = self.product.find(re.compile('.+'), attrs={ITEMPROP: 'ratingValue'})
        rating = ''
        if tags:
            rating = ''.join(tags.strings).strip()
            if not rating:
                rating = tags['content']
        num_rating = '' # make sure rating contains only numbers
        if rating:
            for s in rating:
                if s.isnumeric() or s == '.':
                    num_rating += s
        if num_rating:
            return float(num_rating)
        return 0.0


    def has_product_image(self):
        tags = self.product.find(re.compile('.+'), attrs={ITEMPROP: 'image'})
        if not tags:
            return 0
        return 1


    def get_rating_count(self):
        tags = self.product.find(re.compile('.+'), attrs={ITEMPROP:['ratingCount', 'reviewCount']})

        rating_count = ''
        if tags:
            rating_count = ''.join(tags.strings).strip()
            if not rating_count:
                rating_count = tags['content']
        count = rating_count.replace(',','.')
        if count:
            return float(count)
        return 0.0

    def get_identifier_count(self):
        tags = self.product.find(re.compile('.+'), attrs={ITEMPROP:[re.compile('gtin.*'), 'mpn', 'model']})

        count = 0
        if tags:
            for tag in tags:
                count += 1
        return count

    def get_table_length(self):
        '''
        first search in itemprop=description element, if table is not found search complete page for keywords
        '''
        tags = self.product.find(re.compile('.+'), attrs={ITEMPROP: 'description'})

        table_length = tuple()  # first element is length of table string, second amount of rows
        if tags and tags.find('table'):
            table = ''.join(tags.strings).replace('\n', ' ').replace('\t','').strip()
            table = re.sub(' +',' ',table) # remove double whitespaces
            table_length += (len(table),)
            rowCount = 0
            for row in tags.findAll('tr'):
                rowCount += 1
            table_length += (rowCount,)
        else:
            '''
            table may be outside of schema.org/Product, in this case search for description key word in whole page
            '''
            # first search in product
            tags = self.product.find(re.compile('h[0-5]+'), string=re.compile(self.TABLE_KEYWORDS))

            # then search in whole page
            if not tags:
                tags = self.soup.find(re.compile('h[0-5]+'), string=re.compile(self.TABLE_KEYWORDS))

            if tags:
                sibling = tags # search in all siblings from matching keyword tag
                while sibling and not (sibling.name == 'table' or sibling.name == 'tbody' or sibling.find('table')):
                    sibling = sibling.next_sibling
                    while sibling == '\n':
                        sibling = sibling.next_sibling

                if sibling:
                    table_length = self.extract_table(sibling)
        if not table_length:
            table_length = (0,0)
        return table_length

    def extract_table(self, sibling):
        table_length = tuple()
        if sibling.find('table') or sibling.name == 'table' or sibling.name == 'tbody':
            tags = sibling.find('table')  # check if table is current element or if table is child element
            if not tags:
                tags = sibling  # table is current element
            try:
                table = ''.join(tags.strings).replace('\n', ' ').replace('\t', '').strip()
            except:
                return (0,0)
            table = re.sub(' +', ' ', table)  # remove double whitespaces
            table_length += (len(table),)
            rowCount = 0
            for row in tags.findAll('tr'):
                rowCount += 1
            table_length += (rowCount,)
        return table_length

    def get_list_length(self):
        '''
        first search in itemprop=description for lists, if no list found or element not available search complete page for keywords
        '''
        tags = self.product.find(re.compile('.+'), attrs={ITEMPROP: 'description'})

        list_length = tuple()
        if tags and tags.find_all(['dl','li']):
            tags = tags.find_all(['dl','li'])
            rowCount = 0
            l = ''
            for tag in tags:
                temp = ''.join(tag.strings).replace('\n', ' ').replace('\t','').strip()
                l += re.sub(' +', ' ', temp)  # remove double whitespaces
                rowCount += 1
            list_length += (len(l),rowCount)
        else:
            '''
            some lists may be outside of schema.org/Product
            '''
            tags = self.soup.find(re.compile('h[0-5]+'), string=re.compile(self.LIST_KEYWORDS))
            if tags:
                tags = tags.parent
                if tags.find_all(['dl','ul','li']):
                    tags_list = tags.find_all(['dl','ul','li'])
                    list_length = self.extract_list(tags_list)
        if not list_length:
            list_length = (0,0)
        return list_length

    def extract_list(self, tags_list):
        l = ''
        for tag in tags_list:
            l += ''.join(tag.strings).replace('\n', ' ').replace('\t', '').replace('\r', '').strip()

        rowCount = 0
        if l:
            l = re.sub(' +', ' ', l)  # remove double whitespaces
            for tag in tags_list:
                for row in tag.find_all(['dt', 'li']):
                    rowCount += 1
        list_length = (len(l),rowCount)
        return list_length

    def execute(self):
        self.name = self.get_name()
        self.table_length = self.get_table_length()
        self.description_length = self.get_description_length()
        self.has_image = self.has_product_image()
        self.id_count = self.get_identifier_count()
        self.rating_count = self.get_rating_count()
        self.rating_value = self.get_rating_value()

class Aliexpress(StandardMicrodataExtractor):
    LIST_KEYWORDS = 'Item specifics'

    TABLE_KEYWORDS = 'Descriptions'

    def get_table_length(self):
        tags = self.soup.find(re.compile('h[0-5]+'), string=re.compile(self.TABLE_KEYWORDS))
        if tags:
            table_tag = tags.parent.next_sibling.next_sibling.next_sibling
            table_length = self.extract_table(table_tag)
            return table_length
        return (0,0)

class Apple(StandardMicrodataExtractor):
    LIST_KEYWORDS = ''

    TABLE_KEYWORDS = ''

class Bestbuy(StandardMicrodataExtractor):
    LIST_KEYWORDS = ''

    TABLE_KEYWORDS = ''

class Bjs(StandardMicrodataExtractor):
    LIST_KEYWORDS = '.*Features'

    TABLE_KEYWORDS = ''

    def get_list_length(self):
        tags = self.soup.find(re.compile('h[0-5]+'), string=re.compile(self.LIST_KEYWORDS))
        list_length = (0,0)
        if tags:
            tags = tags.parent.next_sibling
            if tags.find_all(['li']):
                tags = tags.find_all(['li'])
                list_length = self.extract_list(tags)
        return list_length

    def extract_list(self, tags_list):
        list_length = tuple()
        l = ''
        for tag in tags_list:
            l += ''.join(tag.strings).replace('\n', ' ').replace('\t', '').replace('\r', '').strip()
        l = re.sub(' +', ' ', l)  # remove double whitespaces
        list_length += (len(l),)
        rowCount = 0
        for tag in tags_list:
            for row in tag:
                rowCount += 1
        list_length += (rowCount,)
        return list_length

class Conns(StandardMicrodataExtractor):
    LIST_KEYWORDS = ''

    TABLE_KEYWORDS = ''

class Ebay(StandardMicrodataExtractor):
    LIST_KEYWORDS = ''

    TABLE_KEYWORDS = 'Item specifics'

    def get_name(self):
        tags = self.product.find('h1', attrs={ITEMPROP: 'name'})

        name = ''
        if tags:
            name = ''.join(tags.strings).replace('\n',' ').strip()
            name = re.sub(' +',' ',name) # remove double whitespaces
        if name:
            return 1
        return 0

class Flipkart(StandardMicrodataExtractor):
    LIST_KEYWORDS = 'Specifications'

    TABLE_KEYWORDS = 'Specifications'

    def get_table_length(self):
        # first search in product
        tags = self.product.find(re.compile('h[0-5]+'), string=re.compile(self.TABLE_KEYWORDS))

        # then search in whole page
        if not tags:
            tags = self.soup.find(re.compile('h[0-5]+'), string=re.compile(self.TABLE_KEYWORDS))

        table_length = tuple()
        if tags:
            length = 0
            rowCount = 0
            for tag in tags.find_next_siblings('table'):
                table_length = self.extract_table(tag)
                length += table_length[0]
                rowCount += table_length[1]
            table_length = (length,rowCount)

        if not table_length:
            table_length = (0,0)
        return table_length

class Frontierpc(StandardMicrodataExtractor):
    LIST_KEYWORDS = ''

    TABLE_KEYWORDS = ''

    def get_table_length(self):
        tags = self.soup.find('div',attrs={'class':'tabs-content'})
        table_length = (0,0)
        if tags:
            rowCount = 0
            for tag in tags.find_all('div',attrs={'class':'row','itemprop':'feature'}):
                rowCount += 1
            table = ''.join(tags.strings).replace('\n', ' ').strip()
            table = re.sub(' +', ' ', table)  # remove double whitespaces
            table_length = (len(table),rowCount)
        return table_length

class Newegg(StandardMicrodataExtractor):
    LIST_KEYWORDS = '"Learn more.*'

    TABLE_KEYWORDS = ''

    def get_list_length(self):
        #tags = self.soup.find('h2')#, #string=re.compile(self.LIST_KEYWORDS))
        tags = self.soup.find('h2',attrs={'class':'sectionTitle'})
        list_length = tuple()
        if tags:
            sibling = tags.next_sibling
            return (0,0)
            tags_list = sibling.find_all(['dl','ul','li'])
            list_length = self.extract_list(tags_list)
        if not list_length:
            list_length = (0,0)
        return list_length

class Overstock(StandardMicrodataExtractor):
    LIST_KEYWORDS = ''

    TABLE_KEYWORDS = 'Specs'

class Samsclub(StandardMicrodataExtractor):
    LIST_KEYWORDS = 'Description'

    TABLE_KEYWORDS = 'Specifications'

class Searsoutlet(StandardMicrodataExtractor):
    LIST_KEYWORDS = ''

    TABLE_KEYWORDS = 'Specifications'

    def get_table_length(self):
        # first search in product
        tags = self.product.find(re.compile('h[0-5]+'), string=re.compile(self.TABLE_KEYWORDS))

        # then search in whole page
        if not tags:
            tags = self.soup.find(re.compile('h[0-5]+'), string=re.compile(self.TABLE_KEYWORDS))

        if tags:
            sibling = tags.next_sibling.next_sibling
            if sibling:
                table_length = self.extract_table(sibling)
        if not table_length:
            table_length = (0,0)
        return table_length

    def extract_table(self, sibling):
        table_length = tuple()
        if sibling.find('div',attrs={'class':'row-fluid'}):
            tags = sibling.find_all('div',attrs={'class':'row-fluid'})
            table = ''
            rowCount = 0
            for tag in tags:
                table += ''.join(tag.strings).replace('\n', ' ').replace('\t', '').strip()
                rowCount += 1
            table = re.sub(' +', ' ', table)  # remove double whitespaces
            table_length = (len(table),rowCount)
        return table_length

class Shop(StandardMicrodataExtractor):
    LIST_KEYWORDS = ''

    TABLE_KEYWORDS = ''

    def get_table_length(self):
        return (0,0)

class ShopLenovo(StandardMicrodataExtractor):
    LIST_KEYWORDS = 'Tech Specs'

    TABLE_KEYWORDS = 'Tech Specs'

    def get_description_length(self):
        pass

class Target(StandardMicrodataExtractor):
    LIST_KEYWORDS = ''

    TABLE_KEYWORDS = ''

class Techspot(StandardMicrodataExtractor):
    LIST_KEYWORDS = ''

    TABLE_KEYWORDS = ''

class Tesco(StandardMicrodataExtractor):
    LIST_KEYWORDS = ''

    TABLE_KEYWORDS = 'specifications'

    def get_table_length(self):
        '''
        table may be outside of schema.org/Product, in this case search for description key word in whole page
        '''
        # first search in product
        tags = self.product.find(re.compile('h[0-5]+'), string=re.compile(self.TABLE_KEYWORDS))

        # then search in whole page
        if not tags:
            tags = self.soup.find(re.compile('h[0-5]+'), string=re.compile(self.TABLE_KEYWORDS))

        if tags:
            sibling = tags.next_sibling.next_sibling # search in all siblings from matching keyword tag

            if sibling:
                table_length = self.extract_table(sibling)
        if not table_length:
            table_length = (0,0)
        return table_length

    def extract_table(self, sibling):
        table_length = tuple()
        if sibling.find_all('div',attrs={'class':'product-spec-cell product-spec-label'}):
            tags = sibling.find_all('div',attrs={'class':'product-spec-cell product-spec-label'})
            table = ''.join(sibling.strings).replace('\n', ' ').replace('\t', '').strip()
            table = re.sub(' +', ' ', table)  # remove double whitespaces
            rowCount = 0
            for tag in tags:
                rowCount += 1
            table_length = (len(table),rowCount)
        return table_length

class Walmart(StandardMicrodataExtractor):
    LIST_KEYWORDS = 'About this item'

    TABLE_KEYWORDS = 'Specifications'

    def get_table_length(self):
        '''
                    table may be outside of schema.org/Product, in this case search for description key word in whole page
                    '''
        # first search in product
        tags = self.product.find(re.compile('h[0-5]+'), string=re.compile(self.TABLE_KEYWORDS))

        # then search in whole page
        if not tags:
            tags = self.soup.find(re.compile('h[0-5]+'), string=re.compile(self.TABLE_KEYWORDS))

        table_length = tuple()
        if tags:
            sibling = tags.next_sibling.next_sibling

            if sibling:
                table_length = self.extract_table(sibling)

        if not table_length:
            table_length = (0, 0)
        return table_length

    def get_list_length(self):
        '''
        first search in itemprop=description for lists, if no list found or element not available search complete page for keywords
        '''
        tags = self.product.find(re.compile('h[0-5]+'), attrs={ITEMPROP: 'description'})

        list_length = tuple()
        if tags and tags.find_all(['dl','li']):
            tags = tags.find_all(['dl','li'])
            rowCount = 0
            l = ''
            for tag in tags:
                temp = ''.join(tag.strings).replace('\n', ' ').replace('\t','').strip()
                l += re.sub(' +', ' ', temp)  # remove double whitespaces
                rowCount += 1
            list_length += (len(l),rowCount)
        else:
            '''
            some lists may be outside of schema.org/Product
            '''
            tags = self.soup.find_all(re.compile('h[0-5]+'), string=re.compile(self.LIST_KEYWORDS))
            for tag in tags:
                tag = tag.next_sibling.next_sibling
                if tag:
                    if tag.find_all(['dl','ul','li']):
                        tags_list = tag.find_all(['dl','ul','li'])
                        list_length = self.extract_list(tags_list)
                        break
        if not list_length:
            list_length = (0,0)
        return list_length

def covers(p,p1):
	if p == p1:
		return True

	indexMap = {}
	i = 0
	j = 0
	if len(p) == 0:
		return False
	elif len(p1) == 0:
		return True

	wildcard = False
	while j < len(p1):
		if p1[j] == p[i]:
			wildcard = False
			indexMap[j] = i
			if i < len(p) - 1:
				i += 1
			j += 1
		elif p[i] == '*':
			if i < len(p)-1:
				if p1[j] == p[i+1]: # check special case if character after * match, as * could also be the empty string, e.g. abc*dc covers abcdc
					i += 1
			indexMap[j] = i
			j += 1
			if i < len(p) - 1:
				i += 1
			wildcard = True
		elif wildcard:
			indexMap[j] = i-1
			j += 1
		else:
			return False

	for i,e in enumerate(p):
		if e == '*':
			continue
		count = 0
		if i not in indexMap.values():
			return False

		for k in indexMap:
			if indexMap[k] == i:
				count += 1
		if count > 1:
			return False
	return True

def seqCovers(seq1,seq2):
	if len(seq1) > len(seq2):
		return False

	isCovered = False
	for s2 in seq2:
		for s1 in seq1:
			if not covers(s1,s2):
				isCovered = False
			else:
				isCovered = True
				break
	return isCovered

def getUrlPath(url):
    parsed = urlparse(url)
    path = parsed.path
    return path

def getUrlQuery(url):
	parsed = urlparse(url)
	query = parsed.query
	if query:
		return query
	else:
		return ' '

def getUrlFilename(url):
	parsed = urlparse(url)
	filename = parsed.path.split('/')
	for f in filename:
		if '.' in f:
			return f
	return ' '

def patternCover(url,patterns):
    countCover = 0
    countNot = 0
    urlPath = getUrlPath(url).split('/')
    query = getUrlQuery(url)
    filename = getUrlFilename(url)
    if len(urlPath) < 2:
        return 0
    t = tuple([e for e in urlPath if e and '.' not in e])
    urlSeq = (t, filename, query)
    for pattern in patterns:
        p = make_tuple(pattern)
        if seqCovers(p[0],urlSeq[0]):
            if covers(p[1],urlSeq[1]):
                if covers(p[2],urlSeq[2]):
                    countCover += 1
                    break
    return countCover

def evaluate_feautures():
    path = '/home/steffen/Uni/masterthesis/label/products/'
    folders = sorted(os.listdir(path))
    for folder in folders:
        page_count = 0
        missing_feat = defaultdict(int)
        feature_vect = ['name', 'description_length', 'rating_value', 'rating_count', 'is_image', 'identifier_count',
                        'table_length', 'list_length']
        print (folder)
        for filepath in os.listdir(path + folder):
            f = open(path + folder + '/' + filepath, 'rb') # rb prevents decode error

            try:
                c = eval(folder)
                me = c(f.read())
            except:
                continue

            name = me.get_name()
            list_length = me.get_list_length()
            rating_value = me.get_rating_value()
            rating_count = me.get_rating_count()
            is_image = me.has_product_image()
            identifier_count = me.get_identifier_count()
            table_length = me.get_table_length()
            description_length = me.get_description_length() # called at the end because otherwise it may remove tables and lists
            page_count += 1

            for feat in feature_vect:
                if feat == 'list_length':
                    if eval(feat)[0] != 0:
                        missing_feat[feat] += 1
                elif feat == 'table_length':
                    if eval(feat)[0] != 0:
                        missing_feat[feat] += 1
                elif feat == 'rating_value':
                    if eval(feat) != '':
                        missing_feat[feat] += 1
                elif feat == 'rating_count':
                    if eval(feat) != '':
                        missing_feat[feat] += 1
                elif feat == 'is_image':
                    if eval(feat) != 0:
                        missing_feat[feat] += 1
                elif feat == 'identifier_count':
                    if eval(feat) != 0:
                        missing_feat[feat] += 1
                elif feat == 'description_length':
                    if eval(feat) != 0:
                        missing_feat[feat] += 1
                elif feat == 'name':
                    if eval(feat) != '':
                        missing_feat[feat] += 1
                else:
                    missing_feat[feat] += 1

        for key in missing_feat:
            value = missing_feat[key]
            value = value / page_count
            missing_feat[key] = value

        with open(EVALUATOIN_CSV, 'a') as csvfile:
            feature_vect = ['domain'] + feature_vect
            writer = csv.DictWriter(csvfile, fieldnames=feature_vect)
            if os.stat(EVALUATOIN_CSV).st_size == 0:
                writer.writeheader()
            missing_feat['domain'] = folder
            writer.writerow(missing_feat)

def normalize_rating(df):
    domains = set()
    rows = pd.DataFrame()
    for i,row in enumerate(df.iterrows()):
        domain_new = row[1][0]
        if domain_new not in domains:
            domains.add(domain_new)
            rows = pd.DataFrame(data=df[df['domain'] == domain_new], index=df[df['domain'] == domain_new].index)
            min = rows['rating_value'].min()
            max = rows['rating_value'].max()
            for index in rows.index:
                if not pd.isnull(rows.loc[index,'rating_value']):
                    try:
                        rating = (rows.loc[index,'rating_value'] - min) / (max - min)
                        df.loc[index,'rating_value'] = rating
                    except:
                        df.loc[index,'rating_value'] = 0
    return df

def build_dataframe():
    #columns = ['domain','url','isCovered','coverCount','name','rating_count','rating_value','has_image','identifier_count','list_length',
    #            'list_rows','table_length','table_rows','description_length','label']
    columns = ['domain','url','name','rating_count','rating_value','has_image','identifier_count','list_length',
                'list_rows','table_length','table_rows','description_length','price','prediction']

    df_finale = pd.DataFrame(columns=columns)
    path = 'database/'

    domain = sys.argv[1]
    page_count = 0
    print(domain)
    file_path = path + domain + '.json'

    '''
    Entity model features
    '''
    db = TinyDB(file_path)
    entity_model = joblib.load('misc_data/entity-model.pkl')

    c = eval(domain)
    for entry in db:
        url = entry['url']
        row = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        try:
            me = c(entry['html'])
            name = me.get_name()
            list_length = me.get_list_length()
            rating_value = me.get_rating_value()
            rating_count = me.get_rating_count()
            has_image = me.has_product_image()
            identifier_count = me.get_identifier_count()
            table_length = me.get_table_length()
            description_length = me.get_description_length()  # called at the end because otherwise it may remove tables and lists
            row = [name, rating_count, rating_value, has_image, identifier_count, list_length[0],
                   list_length[1], table_length[0], table_length[1], description_length]
        except Exception as err:
            print (err)
            pass
        row = [0,0] + row + [0]
        row = [r if r else 0 for r in row]
        np_array = np.array([row],dtype=np.float64)
        df = pd.DataFrame(data=np_array,columns=columns)
        df = df.fillna(value=0)
        df = normalize_rating(df)
        df = df.fillna(value=0)
        prediction = entity_model.predict_proba(df.drop(['domain','url','prediction'],axis=1))
        df['prediction'] = prediction[0][1]
        df['domain'] = domain
        df['url'] = url
        df_finale = pd.concat([df_finale,df])
        page_count += 1
        sys.stdout.write('\r' + str(page_count))
        sys.stdout.flush()
    #df_final = pd.DataFrame(data=data,columns=columns)
    df_finale.to_csv('results/'+ domain.lower()  +'.csv')
    print ('done')
    return df_finale


def extract_row(domain,url,html):
    row = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    c = eval(domain)
    try:
        me = c(html)
        name = me.get_name()
        list_length = me.get_list_length()
        rating_value = me.get_rating_value()
        rating_count = me.get_rating_count()
        rating_ratio = 0.0
        if rating_count and rating_value:
            rating_ratio = rating_count/rating_value
        else:
            rating_ratio = 0.0
        has_image = me.has_product_image()
        identifier_count = me.get_identifier_count()
        table_length = me.get_table_length()
        description_length = me.get_description_length()  # called at the end because otherwise it may remove tables and lists
        price = me.get_price()
        row = [name, rating_count, rating_value, rating_ratio, has_image, identifier_count, list_length[0],
        list_length[1], table_length[0], table_length[1], description_length, price]
    except Exception as err:
#        print (err)
         pass

    row = [domain.lower(),url] + row
    row = [r if r else 0 for r in row]
    return row

#build_dataframe()
