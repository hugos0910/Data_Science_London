import numpy as np
import pandas as pd
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def obtain_parameters(classifier_name, X_train, y, num_processor):
  if classifier_name == 'RF':
    classifier = RandomForestClassifier()
    param_grid = dict(max_depth = [10, 20, 30, None], min_samples_split = [2,4,6], n_estimators = [500])
  elif classifier_name == 'ET':
    classifier = ExtraTreesClassifier()
    param_grid = dict(criterion = ['gini', 'entropy'],
                      max_depth = [10, 20, None], 
                      min_samples_split = [2,4,6], 
                      min_samples_leaf = [1,2,3], 
                      n_estimators = [500])
  elif classifier_name == 'SVM':
    steps = [('scl', StandardScaler()), 
             ('clf', SVC())]
    classifier = Pipeline(steps)
    param_grid = dict(clf__C = [1,5,10,15,20,25,30], 
                      clf__kernel = ['rbf'], 
                      clf__gamma = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1])
  elif classifier_name == 'KNN':
    steps = [('scl', StandardScaler()), 
             ('clf', KNeighborsClassifier())]
    classifier = Pipeline(steps)
    param_grid = dict(clf__n_neighbors = list(range(1,31)))
  elif classifier_name == 'LR':
    steps = [('scl', StandardScaler()), 
         ('clf', LogisticRegression())]
    classifier = Pipeline(steps)
    param_grid = dict(clf__penalty = ['l1', 'l2'], clf__C = [0.1,1,5,6,7,8,10,15,20,25,30])
  # grid = RandomizedSearchCV(classifier, param_grid, cv = 10, scoring = 'accuracy', n_iter = num_iter, n_jobs = num_processor)
  grid = GridSearchCV(classifier, param_grid, cv = 10, scoring = 'accuracy', n_jobs = num_processor)

  grid.fit(X_train,y)
  grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
  print(grid_mean_scores)
  print(grid.best_estimator_)
  print(grid.best_params_)
  print('The best CV score using %s classifier is %f.' %(classifier_name, grid.best_score_))

def classify(classifier_name, X_train, y, X_test):
  if classifier_name == 'RF':
    clf = RandomForestClassifier(n_estimators = 500, min_samples_split = 2, max_depth = None)
  elif classifier_name == 'ET':
    clf = ExtraTreesClassifier(criterion = 'gini', n_estimators = 500, min_samples_leaf = 1, min_samples_split = 4, max_depth = None)
  elif classifier_name == 'KNN':
    clf = KNeighborsClassifier(n_neighbors = 5)
  elif classifier_name == 'SVM':
    clf = SVC(kernel = 'rbf', gamma = 0.1, C = 5)
  elif classifier_name == 'LR':
    clf = LogisticRegression(C = 0.1, penalty = 'l1')
  prediction = clf.fit(X_train, y).predict(X_test)
  return prediction