from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from urllib.parse import urlparse
from ast import literal_eval as make_tuple
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
import seaborn as sns
import csv


from urlmodel import extractor


'''
deprecated! old classifier for global URL model
'''

def getUrlPath(url):
    parsed = urlparse(url.strip())
    path = parsed.path
    return path

def nestingCount(url):
    path = getUrlPath(url)
    summe = 0
    if path:
        if path[len(path) - 1] == '/':
            path = path[0:len(path) - 1]
        s = path.split('/')
        summe = sum(1 for e in s if e)
    return summe

def calc_url_features(df,train_index):
    urls = [row for row in df.iloc[train_index]['url']]
    abs_ranges = extractor.absolute(urls)
    nest_ranges = extractor.nesting(urls)
    avg_value = extractor.average(urls)
    for i,row in df.iterrows():
        url = row[1]
        urlLength = len(getUrlPath(url))
        for range in abs_ranges:
            t = make_tuple(range)
            if t[0] <= urlLength <= t[1]:
                df.loc[i,range] = 1
            else:
                df.loc[i,range] = 0

        count = nestingCount(url)
        for range in nest_ranges:
            t = make_tuple(range)
            if t[0] <= count <= t[1]:
                df.loc[i,range] = 1
            else:
                df.loc[i,range] = 0

        for value in avg_value:
            df.loc[i,'averageDeviation'] = (float(len(url)) - float(value))
            df.loc[i,'averageRatio'] = (float(len(url)) / float(value))
    return df

df = pd.DataFrame.from_csv('results/dataframe.csv', sep=',')

#model = GaussianNB()
model = SVC(probability=True)
drop_columns = ['domain','url','isCovered','coverCount','name','rating_count','rating_value','has_image','identifier_count','list_length',
               'list_rows','table_length','table_rows','description_length','label']
X = df.drop(drop_columns, axis=1)
y = df['label']

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
for train_index, test_index in sss.split(X, y):
    '''
    calculate url features based on training data of positive class
    '''
    df = calc_url_features(df,train_index)
    X_url = df.drop(drop_columns, axis=1)
    X_train, X_test = X_url.iloc[train_index], X_url.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    precision = precision_score(y_test, prediction)

'''
Evaluate
'''
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
accuracy = accuracy_score(y_test, prediction)
print ('Precision ' + str(precision) + ' Recall ' + str(recall) + ' F1 ' + str(f1) + ' Accuracy ' + str(accuracy))
with open('results/results.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(
        ['# good products', '# bad products', 'TP', 'FN', 'FP', 'TN', 'precision', 'recall', 'f1',
         'accuracy'])
    cm = confusion_matrix(y_test, prediction)
    tp = cm[1][1]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[0][0]
    writer.writerow([len(y_test[y_test == 1]), len(y_test[y_test == 0]), tp, fn, fp, tn, precision, recall, f1, accuracy])

p = model.decision_function(X_test)
proba_1 = np.array([item[1] for item in model.predict_proba(X_test)])

result_plot = pd.DataFrame(data=np.transpose(np.vstack((prediction,y_test.as_matrix(),p))),columns=['predict','expect','distance to hyperplane'],index=y_test.index)
result_plot.loc[(result_plot['predict'] == 1) & (result_plot['expect'] == 1),'class'] = 'tp'
result_plot.loc[(result_plot['predict'] == 1) & (result_plot['expect'] == 0),'class'] = 'fp'
result_plot.loc[(result_plot['predict'] == 0) & (result_plot['expect'] == 0),'class'] = 'tn'
result_plot.loc[(result_plot['predict'] == 0) & (result_plot['expect'] == 1),'class'] = 'fn'

plot = sns.lmplot('index','distance to hyperplane',data=result_plot.reset_index(),
                  fit_reg=False,hue='class',legend=True,palette=dict(tp='#4878CF',fp='#6ACC65',tn='#D65F5F',fn='#56B4E9'))

plot.savefig('results/urlmodel-plot.png')
sns.plt.show()
