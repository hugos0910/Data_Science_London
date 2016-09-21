
import numpy as np
import pandas as pd
import util
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Import data
print('Importing data...')
train = pd.read_csv('train.csv', header = None)
test = pd.read_csv('test.csv', header = None)
label = pd.read_csv('trainLabels.csv', header = None)
label = np.ravel(label)

# Cleaning data
print('Sanitizing data...')
pca = PCA(n_components = 12, whiten = True)
train = pca.fit_transform(train)
test = pca.transform(test)

# # Obtain best parameters
# num_processor = -1
# util.obtain_parameters('RF', train, label, num_processor)
# util.obtain_parameters('ET', train, label, num_processor)
# util.obtain_parameters('SVM', train, label, num_processor)
# util.obtain_parameters('KNN', train, label, num_processor)
# util.obtain_parameters('LR', train, label, num_processor)

# Training classifier

'''
classifier abbreviations:
RF - Random Forest
ET - Extra Trees
SVM - Support Vector Machine
KNN - K Nearest Neighbors
LR - Logistic Regression
'''

classifier_name = 'SVM'
print('Training and prediction with %s classifier...' %classifier_name)
prediction = util.classify(classifier_name, train, label, test)

# Exporting solution
index = list(range(1,len(test) +1))
print('Writing data to CSV file...')
df_prediction = pd.DataFrame(data = prediction, index = index, columns = ['Solution'])
df_prediction_csv = df_prediction.to_csv('prediction_%s.csv' % classifier_name, index_label = ["Id"])
