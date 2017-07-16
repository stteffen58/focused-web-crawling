import numpy as np
import csv
import pandas as pd


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


path = '/home/eckel/data/quad-stats'
fproduct = open(path+'/producturls.csv','r')
fnonproduct = open(path+'/nonproducturls.csv','r')

for line in fproduct:
    nonproductLine = fnonproduct.readline()
    data_product = np.array(line.split(','))
    data_nonproduct = np.array(nonproductLine.split(','))
    print (data_product[0])
    print (data_nonproduct[0])
    labels1 = np.array([1 for i in data_product[1:len(data_product)]])
    labels0 = np.array([0 for i in data_nonproduct[1:len(data_nonproduct)]])
    sample = balanced_subsample(np.hstack((data_product,data_nonproduct)),np.hstack((labels1,labels0)))

    datapath = '/home/eckel/master/patternmining/data/'
    csvfile = open('%sproducturls.csv' % datapath, 'a')
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    out_product = [data_product[0]]
    out_product.extend(sample[0][sample[1] == 1])
    writer.writerow(out_product) # select products from sample, i.e. row where label == 1
    csvfile.close()

    csvfile = open('%snonproducturls.csv' % datapath, 'a')
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    out_nonproduct = [data_nonproduct[0]]
    out_nonproduct.extend(sample[0][sample[1] == 0])
    writer.writerow(out_nonproduct) # select non products from sample, i.e. row where label == 0
    csvfile.close()

    csvfile = open('%ssample-summary.csv' % datapath,'a')
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([data_product[0],len(out_product),len(out_nonproduct)])
    csvfile.close()

    products = np.hstack((out_product[1:len(out_product)],out_nonproduct[1:len(out_nonproduct)]))
    labels = np.hstack((sample[1][sample[1] == 1],sample[1][sample[1] == 0]))
    data = np.vstack((products,labels))
    df = pd.DataFrame(data=np.transpose(data),columns=['url','label'])
    df.to_csv('%surl-dataframe.csv' % datapath)
