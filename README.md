# Data Science London + Scikit-learn

## Goal and Motivation 
Data Science London is hosting a meetup on Scikit-learn.  This competition is a practice ground for trying, sharing, and creating examples of sklearn's classification abilities.  

## Cleaning Data
The data given for this competition contains no NA.  The training data consists of 1000 entries, each with 40 features.  The testing data consists of 9000 entries, also with 40 features.  All of the features are numerical, non of them are categorical features.  Principle component analysis (PCA) was applied to the data to reduce the number of features from 40 to 12 to improve prediction accuracy.

## Feature Engineering
Since the features are meaningless, there is no feature engineering applied to this dataset.

## Choosing Classifiers
The cross-validation accuracy obtained from GridSearch with 10-folds for five of the chosen classifiers are listed as followed in decreasing order:
* K Nearest Neighbors - 0.955
* Support Vector Machine - 0.948
* Extra Trees - 0.936
* Random Forest - 0.911
* Logistic Regression -  0.833

## Result
Since the classifier training and test set prediction time for this dataset is low, I decided to submit data predicted using all 5 classifiers and see how they score on Kaggle.  The accuracy obtained as listed as followed, in decreasing order:
* Support Vector Machine - 0.94981
* K Nearest Neighbors - 0.94158
* Extra Trees - 0.92084
* Random Forest - 0.90437
* Logistic Regression -  0.81333
The cross-validation accuracy closely resembles the prediction accuracy of the test data.  It is not surprising that the SVM outperforms the KNN for the test set, since they had extremely close cross-validation accruacy.
