import pandas as pd
import seaborn as sns
import numpy as np
import csv

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression as logreg
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.externals import joblib

from urlmodel import extractor
from urllib.parse import urlparse
from ast import literal_eval as make_tuple


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
                    rating = (rows.loc[index,'rating_value'] - min) / (max - min)
                    df.loc[index,'rating_value'] = rating
    return df

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

'''
Load dataset from csv
'''
result_path = 'entitymodel/results/'
df = pd.DataFrame.from_csv('%sdataframe.csv' % result_path, sep=',')
df = normalize_rating(df)
df = df.fillna(value=0)
print (df.head())
drop_columns = ['label', 'domain', 'url']
X = df.drop(drop_columns, axis=1)
y = df['label']

'''
Fit model
'''
model = logreg()
#model = SVC(probability=True)
#model = GaussianNB()
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
for train_index, test_index in sss.split(X, y):
    '''
    calculate url features based on training data of positive class
    '''
    #df = calc_url_features(df,train_index)

    #X_url = df.drop(drop_columns, axis=1)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    print ('shape')
    print (X_train.shape)
    print (model.coef_)
    #model.coef_ = np.array([[0.1,0.1,0.1,0.1,0.2,0,0.4,0,0,0]])
    prediction = model.predict(X_test)

joblib.dump(model, '%sentity-model.pkl' % result_path)

'''
Evaluate
'''
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
print ('Precision ' + str(precision) + ' Recall ' + str(recall) + ' F1 ' + str(f1) + ' Accuracy ' + str(accuracy))
with open('%sresults.csv' % result_path, 'w') as csvfile:
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

proba_1 = np.array([item[1] for item in model.predict_proba(X_test)])

result_plot = pd.DataFrame(data=np.transpose(np.vstack((prediction,y_test.as_matrix(),proba_1))),columns=['predict','expect','probability of class 1'],index=y_test.index)
result_plot.loc[(result_plot['predict'] == 1) & (result_plot['expect'] == 1),'class'] = 'tp'
result_plot.loc[(result_plot['predict'] == 1) & (result_plot['expect'] == 0),'class'] = 'fp'
result_plot.loc[(result_plot['predict'] == 0) & (result_plot['expect'] == 0),'class'] = 'tn'
result_plot.loc[(result_plot['predict'] == 0) & (result_plot['expect'] == 1),'class'] = 'fn'

plot = sns.lmplot('index','probability of class 1',data=result_plot.reset_index(),fit_reg=False,hue='class',legend=True,
                  palette=dict(tp='#4878CF',fp='#6ACC65',tn='#D65F5F',fn='#56B4E9'))
plot.savefig('%sentitymodel-plot.png' % result_path)

'''
pca plot
'''

df_pca = PCA(n_components=2).fit_transform(X_test)
result_plot_pca = pd.DataFrame(data=np.transpose(np.vstack((df_pca[:,0],df_pca[:,1],prediction,y_test.as_matrix()))),columns=['pca_1','pca_2','predict','expect'],index=y_test.index)
result_plot_pca.loc[(result_plot['predict'] == 1) & (result_plot['expect'] == 1),'class'] = 'tp'
result_plot_pca.loc[(result_plot['predict'] == 1) & (result_plot['expect'] == 0),'class'] = 'fp'
result_plot_pca.loc[(result_plot['predict'] == 0) & (result_plot['expect'] == 0),'class'] = 'tn'
result_plot_pca.loc[(result_plot['predict'] == 0) & (result_plot['expect'] == 1),'class'] = 'fn'

plot_pca = sns.lmplot('pca_1','pca_2',data=result_plot_pca,hue='class',fit_reg=False,
                      palette=dict(tp='#4878CF',fp='#6ACC65',tn='#D65F5F',fn='#56B4E9'))
sns.plt.ylim(-600,1500)
sns.plt.xlim(-300,100)
plot_pca.savefig('%spca-plot.png' % result_path)

sns.plt.show()

'''
ROC curve
'''

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

fpr,tpr,thresholds = roc_curve(y_test,proba_1,pos_label=1)
df_threshold = pd.DataFrame(data=np.transpose(np.vstack((tpr, thresholds))))
#print (df_threshold)
plot_roc = sns.jointplot('false positives','true positives',data=pd.DataFrame(data=np.transpose(np.vstack((fpr,tpr))),
                                                                   columns=['false positives','true positives']))
sns.plt.xticks([0.0,1.0])
sns.plt.show()


'''
Calculate average value per domain
'''

df = pd.concat([df,pd.DataFrame(proba_1,columns=['probability'])],axis=1)
df = df.dropna()
avg_proba = df.groupby(df['domain'])['probability'].mean()
a = avg_proba.to_frame()
a.reset_index(inplace=True)
a.columns = ['domain','avg_proba']
a.to_csv('%savg_proba.csv' % result_path)

print ('tp ' + str(tp), 'tn ' + str(tn), 'fn ' + str(fn), 'fp ' + str(fp))
