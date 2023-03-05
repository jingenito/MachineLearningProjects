#creating a module

import numpy

def confusion_matrix(y_true, y_pred) :
    C = numpy.zeros((2,2))
    for i in range(len(y_true)) :
        C[0,0] += 1 if y_true[i] == 0 and y_pred[i] == 0 else 0 #true negatives
        C[0,1] += 1 if y_true[i] == 0 and y_pred[i] == 1 else 0 #false positives
        C[1,0] += 1 if y_true[i] == 1 and y_pred[i] == 0 else 0 #false negatives
        C[1,1] += 1 if y_true[i] == 1 and y_pred[i] == 1 else 0 #true positives
    return C

def accuracy_score(y_true, y_pred) :
    """Computes the accuracy of the data."""
    C = confusion_matrix(y_true, y_pred)
    return (C[0,0] + C[1,1]) / (C[0,0] + C[0,1] + C[1,0] + C[1,1])

def precision_score(y_true, y_pred) :
    """Computes the precision of the data."""
    C = confusion_matrix(y_true, y_pred)
    return C[1,1] / (C[0,1] + C[1,1])

def recall_score(y_true, y_pred) :
    """Computes the recall of the data."""
    C = confusion_matrix(y_true, y_pred)
    return C[1,1] / (C[1,0] + C[1,1])

def f1_score(y_true, y_pred) :
    """Computes the f1_score of the data."""
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2*((prec*recall)/(prec + recall))