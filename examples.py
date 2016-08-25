import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
import math
import RobustLinearProgramming
import PolyhedralConicFunctions


""""
    #Polyhedral Conic Functions Example
# PCF algorithm requires both data clusters
# This function seperates data to clusters according to their labels
def seperatetoAB(data, labels, indexes):
    A = []
    B = []
    for i in indexes:
        if labels[i] == -1:
            A.append(data[i])
        elif labels[i] == 1:
            B.append(data[i])
    A = np.array(A)
    B = np.array(B)
    return A, B


# This function converts given labels (l1,l2) to 1, -1
def convertLabels(labelData, l1, l2):
    lbls = np.zeros(len(labelData))
    for i in range(len(labelData)):
        if labelData[i] == l1:
            lbls[i] = -1
        elif labelData[i] == l2:
            lbls[i] = 1
    return lbls

f = open('PimaDiabetes.txt')
X = []
labels = []
for line in f:
    row = []
    line = line.split(',')
    labels.append(int(line[-1]))
    X.append([float(line[i]) for i in range(len(line)-1)])
labels = np.array(labels)
X = np.array(X)

acc = []
labels = convertLabels(labels, 1, 0)
skf = StratifiedKFold(labels, 10)
for train, test in skf:
    pModel = PolyhedralConicFunctions.PCF_iterative()
    sepData = seperatetoAB(X, labels, train)
    pModel.fit(sepData[0], sepData[1])
    acc.append(accuracy_score(labels[test], pModel.predict(X[test])))
print "Accuracy over iterations:", acc
print "Average accuracy: ", sum(acc)/10

"""
# *********************
""""
   #Polyhedral Conic Functions Example
# PCF algorithm requires both data clusters
# This function seperates data to clusters according to their labels
def seperatetoAB(data, labels, indexes):
    A = []
    B = []
    for i in indexes:
        if labels[i] == -1:
            A.append(data[i])
        elif labels[i] == 1:
            B.append(data[i])
    A = np.array(A)
    B = np.array(B)
    return A, B


# This function converts given labels (l1,l2) to 1, -1
def convertLabels(labelData, l1, l2):
    lbls = np.zeros(len(labelData))
    for i in range(len(labelData)):
        if labelData[i] == l1:
            lbls[i] = -1
        elif labelData[i] == l2:
            lbls[i] = 1
    return lbls

f = open('Heart.txt')
X = []
labels = []
for line in f:
    row = []
    line = line.split(' ')
    labels.append(int(line[-1]))
    X.append([float(line[i]) for i in range(len(line)-1)])
labels = np.array(labels)
X = np.array(X)

acc = []
labels = convertLabels(labels, 1, 2)
skf = StratifiedKFold(labels, 10)
for train, test in skf:
    pModel = PolyhedralConicFunctions.PCF_iterative()
    sepData = seperatetoAB(X, labels, train)
    pModel.fit(sepData[0], sepData[1])
    acc.append(accuracy_score(labels[test], pModel.predict(X[test])))
print "Accuracy over iterations:", acc
print "Average accuracy: ", sum(acc)/10
"""
# *********************
"""
    #Polyhedral Conic Functions Example
# PCF algorithm requires both data clusters
# This function seperates data to clusters according to their labels
def seperatetoAB(data, labels, indexes):
    A = []
    B = []
    for i in indexes:
        if labels[i] == -1:
            A.append(data[i])
        elif labels[i] == 1:
            B.append(data[i])
    A = np.array(A)
    B = np.array(B)
    return A, B


# This function converts given labels (l1,l2) to 1, -1
def convertLabels(labelData, l1, l2):
    lbls = np.zeros(len(labelData))
    for i in range(len(labelData)):
        if labelData[i] == l1:
            lbls[i] = -1
        elif labelData[i] == l2:
            lbls[i] = 1
    return lbls

f = open('Ionosphere.txt')
X = []
labels = []
for line in f:
    row = []
    line = line.split(',')
    labels.append(line[-1].replace('\n', ''))
    X.append([float(line[i]) for i in range(len(line)-1)])
labels = np.array(labels)
X = np.array(X)

acc = []
labels = convertLabels(labels, 'b', 'g')
skf = StratifiedKFold(labels, 10)
for train, test in skf:
    pModel = PolyhedralConicFunctions.PCF_iterative()
    sepData = seperatetoAB(X, labels, train)
    pModel.fit(sepData[0], sepData[1])
    acc.append(accuracy_score(labels[test], pModel.predict(X[test])))
print "Accuracy over iterations:", acc
print "Average accuracy: ", sum(acc)/10
"""
# *********************
""""
    #Polyhedral Conic Functions Example
# PCF algorithm requires both data clusters
# This function seperates data to clusters according to their labels
def seperatetoAB(data, labels, indexes):
    A = []
    B = []
    for i in indexes:
        if labels[i] == -1:
            A.append(data[i])
        elif labels[i] == 1:
            B.append(data[i])
    A = np.array(A)
    B = np.array(B)
    return A, B


# This function converts given labels (l1,l2) to 1, -1
def convertLabels(labelData, l1, l2):
    lbls = np.zeros(len(labelData))
    for i in range(len(labelData)):
        if labelData[i] == l1:
            lbls[i] = -1
        elif labelData[i] == l2:
            lbls[i] = 1
    return lbls

f = open('WBCP9Features.txt')
X = []
labels = []
for line in f:
    row = []
    line = line.split(',')
    labels.append(int(line[-1]))
    X.append([float(line[i]) for i in range(1,len(line)-1)])
labels = np.array(labels)
X = np.array(X)

acc = []
labels = convertLabels(labels, 2, 4)
skf = StratifiedKFold(labels, 10)
for train, test in skf:
    pModel = PolyhedralConicFunctions.PCF_iterative()
    sepData = seperatetoAB(X, labels, train)
    pModel.fit(sepData[0], sepData[1])
    acc.append(accuracy_score(labels[test], pModel.predict(X[test])))
print "Accuracy over iterations:", acc
print "Average accuracy: ", sum(acc)/10
"""
# *********************
"""
  #Polyhedral Conic Functions Example
#PCF algorithm requires both data clusters
#This function seperates data to clusters according to their labels
def seperatetoAB(data, labels, indexes):
    A = []
    B = []
    for i in indexes:
        if labels[i] == -1:
            A.append(data[i])
        elif labels[i] == 1:
            B.append(data[i])
    A = np.array(A)
    B = np.array(B)
    return A, B

#This function converts given labels (l1,l2) to 1, -1
def convertLabels(labelData, l1, l2):
    lbls = np.zeros(len(labelData))
    for i in range(len(labelData)):
        if labelData[i] == l1:
            lbls[i] = -1
        elif labelData[i] == l2:
            lbls[i] = 1
    return lbls

f = open('WBCD9Features.txt')
X = []
labels = []
for line in f:
    row = []
    line = line.split(',')
    labels.append(int(line[-1]))
    X.append([float(line[i]) for i in range(1,len(line)-1)])
labels = np.array(labels)
X = np.array(X)

acc = []
labels = convertLabels(labels, 2, 4)
skf = StratifiedKFold(labels, 10)
for train, test in skf:
    pModel = PolyhedralConicFunctions.PCF_iterative()
    sepData = seperatetoAB(X, labels, train)
    pModel.fit(sepData[0], sepData[1])
    acc.append(accuracy_score(labels[test], pModel.predict(X[test])))
print "Accuracy over iterations:", acc
print "Average accuracy: ", sum(acc)/10
"""
# *********************
"""
    #Polyhedral Conic Functions Example

#PCF algorithm requires both data clusters
#This function seperates data to clusters according to their labels
def seperatetoAB(data, labels, indexes):
    A = []
    B = []
    for i in indexes:
        if labels[i] == -1:
            A.append(data[i])
        elif labels[i] == 1:
            B.append(data[i])
    A = np.array(A)
    B = np.array(B)
    return A, B

#This function converts given labels (l1,l2) to 1, -1
def convertLabels(labelData, l1, l2):
    lbls = np.zeros(len(labelData))
    for i in range(len(labelData)):
        if labelData[i] == l1:
            lbls[i] = -1
        elif labelData[i] == l2:
            lbls[i] = 1
    return lbls

f = open('BupaLiver.txt')
X = []
labels = []
for line in f:
    row = []
    line = line.split(',')
    labels.append(int(line[-1]))
    X.append([float(line[i]) for i in range(len(line)-1)])
labels = np.array(labels)
X = np.array(X)

acc = []
labels = convertLabels(labels, 1, 2)
skf = StratifiedKFold(labels, 10)
for train, test in skf:
    pModel = PolyhedralConicFunctions.PCF_iterative()
    sepData = seperatetoAB(X, labels, train)
    pModel.fit(sepData[0], sepData[1])
    acc.append(accuracy_score(labels[test], pModel.predict(X[test])))
print "Accuracy over iterations:", acc
print "Average accuracy: ", sum(acc)/10
"""
# *****************************************************************************************************************
"""
       #Robust Linear Programming example
#RLP predictions  are -1 or 1 but some datasets have different labels.
#This function converts given labels (l1,l2) to 1,-1

def convertLabels(labelData, l1, l2):
    lbls = np.zeros(len(labelData))
    for i in range(len(labelData)):
        if labelData[i] == l1:
            lbls[i] = 1
        elif labelData[i] == l2:
            lbls[i] = -1
    return lbls

f = open('Heart.txt')
X = []
labels = []
for line in f:
    row = []
    line = line.split(' ')
    labels.append(int(line[-1]))
    X.append([float(line[i]) for i in range(len(line)-1)])
labels = np.array(labels)
X = np.array(X)
skf = StratifiedKFold(labels, 10)
acc = 0
labels = convertLabels(labels, 1, 2)

for train, test in skf:
    pModel = RobustLinearProgramming.RLP()
    pModel.fit(X[train], labels[train], 1, 2)
    acc += accuracy_score(labels[test], pModel.predict(X[test]))
print "Ten fold average accuracy:", acc / 10
"""
# *******************
"""
        #Robust Linear Programming example
#RLP predictions  are -1 or 1 but some datasets have different labels.
#This function converts given labels (l1,l2) to 1,-1

def convertLabels(labelData, l1, l2):
    lbls = np.zeros(len(labelData))
    for i in range(len(labelData)):
        if labelData[i] == l1:
            lbls[i] = 1
        elif labelData[i] == l2:
            lbls[i] = -1
    return lbls

f = open('WBCD9Features.txt')
X = []
labels = []
for line in f:
    row = []
    line = line.split(',')
    labels.append(int(line[-1]))
    X.append([float(line[i]) for i in range(1, len(line)-1)])
labels = np.array(labels)
X = np.array(X)
skf = StratifiedKFold(labels, 10)
labels = convertLabels(labels, 2, 4)
acc = 0
for train, test in skf:
    pModel = RobustLinearProgramming.RLP()
    pModel.fit(X[train], labels[train], 2, 4)
    acc += accuracy_score(labels[test], pModel.predict(X[test]))
print "Ten fold average accuracy:", acc/10

"""