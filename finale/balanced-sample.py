import numpy as np
import csv
import pandas as pd
import sys


'''
takes a balanced sample of 10 percent of the complete URLs for each domain
'''

def balanced_subsample(x,y,subsample_size=0.1):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys


path = 'dataframes/'
datapath = 'samples/'
domain = sys.argv[1]
df = pd.DataFrame().from_csv(path + domain + '.csv')
df.reset_index(drop=True,inplace=True) # index has to be reset because every row had the same index for some reason
print (domain)

data_product = df[df['prediction'] >= 0.5].index
data_nonproduct = df[df['prediction'] < 0.5].index
labels1 = np.array([1 for i in data_product[0:len(data_product)]])
labels0 = np.array([0 for i in data_nonproduct[0:len(data_nonproduct)]])
data = np.hstack((data_product,data_nonproduct))
label = np.hstack((labels1,labels0))
sample = balanced_subsample(data,label,subsample_size=1)

products = np.hstack((sample[0][sample[1] == 1],sample[0][sample[1] == 0]))
labels = np.hstack((sample[1][sample[1] == 1],sample[1][sample[1] == 0]))
data = np.vstack((products,labels))
df_final = df.iloc[products]
df_final.reset_index(drop=True,inplace=True)
df_final.to_csv(datapath + domain + '.csv')
