import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import f1_score, classification_report
import pdb
from sklearn.feature_extraction.text import CountVectorizer
from bag_of_words import *
from functions import *
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

train_samples_file = np.genfromtxt('train_samples.txt',encoding='utf-8',delimiter='\t',dtype=None,names=('id','text'),comments=None)
train_samples = train_samples_file['text']


train_labels_file = np.genfromtxt('train_labels.txt',encoding='utf-8',delimiter='\t',dtype=None,names=('id','label'),comments=None)
train_labels = train_labels_file['label']

validation_samples_file = np.genfromtxt('validation_samples.txt',encoding='utf-8',delimiter='\t',dtype=None,names=('id','text'),comments=None)
validation_samples = validation_samples_file['text']

validation_labels_file = np.genfromtxt('validation_labels.txt',encoding='utf-8',delimiter='\t',dtype=None,names=('id','label'),comments=None)
validation_labels = validation_labels_file['label']

test_samples_file = np.genfromtxt('test_samples.txt',encoding='utf-8',delimiter='\t',dtype=None,names=('id','text'),comments=None)
test_samples = test_samples_file['text']
test_ID = test_samples_file['id']

#bow_model = Bag_of_words()
#bow_model.build_vocabulary(train_samples) 

Vectorizer = CountVectorizer()
Vectorizer.fit(train_samples)
Vectorizer.fit(test_samples)
test_features = Vectorizer.transform(test_samples)
train_features = Vectorizer.transform(train_samples)

#train_features = bow_model.get_features(train_samples)
#test_features = bow_model.get_features(test_samples) 
print(test_features.shape)

scaled_train_data, scaled_test_data = normalize_data(train_features, test_features, type='l2')


#scaled_train_data, scaled_test_data, train_samples, test_samples = train_test_split(train_features, np.ravel(test_features), test_size = 0.10, random_state = 101)
#param_grid = {'C': [0.1, 1, 10, 100, 1000],  'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']} 
#svm_model = svm.SVC()
#grid = GridSearchCV( svm_model, param_grid, refit = True, verbose = 3) 
#grid.fit(scaled_train_data,train_labels)
#print(grid.best_params_)
#print(grid.best_estimator_) 

svm_model = svm.SVC(C=1000, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1, kernel='linear', max_iter=-1,
    probability=False, random_state=None, shrinking=True, tol=0.001,
    verbose=False)
svm_model.fit(scaled_train_data, train_labels)
predicted_labels_svm = svm_model.predict(scaled_test_data) 
print(predicted_labels_svm.shape)
print(scaled_test_data.shape)
slicer = slice(2623)
validation_labels = validation_labels[slicer]


model_accuracy_svm = compute_accuracy(validation_labels, predicted_labels_svm)
print('f1 score', f1_score(validation_labels, predicted_labels_svm))

print("SVM model accuracy: ", model_accuracy_svm * 100)
print(classification_report(validation_labels, predicted_labels_svm))
cf_mtx = confusion_matrix(validation_labels, predicted_labels_svm)
print(cf_mtx)
with open("submission5.txt","w+") as f:
    f.write("id,label" + '\n')
    for i in range(len(predicted_labels_svm)):
        _y = str(predicted_labels_svm[i])
        _x = str(test_ID[i])
        f.write(_x + "," + _y + "\n")
