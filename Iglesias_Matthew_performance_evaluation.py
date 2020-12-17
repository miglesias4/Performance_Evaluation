##Base Code by: Olac Fuentes
##Modified by: Matthew Iglesias

import numpy as np
import matplotlib.pyplot as plt
import time 


def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/y_true.shape[0]

def confusion_matrix(y_true,y_pred):
    cm = np.zeros((np.amax(y_true)+1,np.amax(y_true)+1),dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i]][y_pred[i]] += 1
    return cm

# Precision = TP/(TP + FP)
def precision(y_true,y_pred,positive_class):
    cm = confusion_matrix(y_true,y_pred)
    true_pos = np.diag(cm)
    false_pos = cm.sum(axis=0) - true_pos
    cm_precision = .1 * np.sum(true_pos / (true_pos + false_pos))
    return cm_precision

# Recall = TP/(TP+FN)
def recall(y_true,y_pred,positive_class):
    cm = confusion_matrix(y_true,y_pred)
    true_pos = np.diag(cm)
    false_neg = cm.sum(axis=1) - true_pos
    cm_recall = .1 * np.sum(true_pos / (true_pos + false_neg))
    return cm_recall



if __name__ == '__main__':
    print('\nEvaluating Algorithm 1')
    y_test_a1 =  np.load('y_test_a1.npy')
    pred_a1 =  np.load('pred_a1.npy')
    print('Accuracy:  {:.4}'.format(accuracy(y_test_a1,pred_a1)))
    print('Confusion matrix:')
    print(confusion_matrix(y_test_a1,pred_a1))
    print('Precision and recall:')
    for i in range(np.amax(y_test_a1)+1):
        print('Positive class {}, precision: {:.4}, recall: {:.4}'.format(i, precision(y_test_a1,pred_a1,i),recall(y_test_a1,pred_a1,i)))
    
    print('\nEvaluating Algorithm 2')
    y_test_a2 =  np.load('y_test_a2.npy')
    pred_a2 =  np.load('pred_a2.npy')
    print('Accuracy:  {:.4}'.format(accuracy(y_test_a2,pred_a2)))
    print('Precision and recall:')
    for i in range(np.amax(y_test_a2)+1):
        print('Positive class {}, precision: {:.4}, recall: {:.4}'.format(i, precision(y_test_a2,pred_a2,i),recall(y_test_a2,pred_a2,i)))
    
    
    