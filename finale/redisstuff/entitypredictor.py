import os
import sys
import pandas as pd

#from sklearn.externals import joblib


'''
predicts score using entity model. sys.argv[1] = <experiment name>, for example productcatalog
'''

#experiment_name = sys.argv[1]
#entity_pickel = experiment_name  + '/pickles/entity-model.pkl'
#entitymodel = joblib.load(entity_pickel)
#inputpath = 'balanced-samples/'
#outputpath = experiment_name + '/data/'
#for file in os.listdir(inputpath):
#    if os.path.isdir(inputpath + file):
#        continue
#    print (file)
#    df = pd.DataFrame.from_csv(inputpath + file)
def predict(df,entitymodel):
    prediction = entitymodel.predict_proba(df)
    return prediction[0][1]
    #df['prediction'] = prediction
    #df.to_csv(outputpath + file)
    
