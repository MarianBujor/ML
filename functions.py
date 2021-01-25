from sklearn import preprocessing
import numpy as np 

def compute_accuracy(gt_labels, predicted_labels):
    accuracy = np.sum(predicted_labels == gt_labels) / len(predicted_labels)
    return accuracy

def normalize_data(train_data, test_data, type=None):
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler()

    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()

    elif type == 'l1':
        scaler = preprocessing.Normalizer(norm='l1')

    elif type == 'l2':
        scaler = preprocessing.Normalizer(norm='l2')

    if scaler is not None:
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data) 
        return (scaled_train_data, scaled_test_data)
    else:
        print("No scaling was performed. Raw data is returned.")
        return (train_data, test_data)
